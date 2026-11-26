"""In-process background job execution for video extraction.

ADAPTATION NOTE: there is no Celery/Redis in this repository, and the app
is hosted on Railway as a single FastAPI process. Long-running video
processing therefore runs on a small bounded `ThreadPoolExecutor` so it
never blocks the asyncio event loop, with job *state* persisted to SQLite
(backend.video_extraction.db) so status survives a restart even though
in-flight work itself does not (a job that was "processing" when the
process died is left in that status and must be re-submitted with
`/regenerate` -- there is no crash-resume for partially finished frame
extraction). A real distributed queue (Celery+Redis/RQ) would remove that
gap and allow horizontal scaling; this is documented as a known
limitation, not papered over.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

from backend.video_extraction import db
from backend.video_extraction.config import get_settings

logger = logging.getLogger("video_extraction.jobs")


class JobRunner:
    """A tiny bounded worker pool with an in-flight registry."""

    def __init__(self, max_workers: int | None = None):
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers or get_settings().worker_pool_size,
            thread_name_prefix="videox-worker",
        )
        self._lock = threading.Lock()
        self._inflight: dict[str, Future] = {}

    def is_running(self, job_id: str) -> bool:
        with self._lock:
            future = self._inflight.get(job_id)
            return future is not None and not future.done()

    def submit(
        self,
        job_id: str,
        fn: Callable[[Callable[[], bool]], None],
        timeout_seconds: float | None = None,
    ) -> bool:
        """Submit work for job_id. Returns False if already running.

        ``fn`` is called with one argument, ``cancel_check``: a zero-arg
        callable that returns True once ``timeout_seconds`` has elapsed since
        this call. ``fn`` is expected to poll it cooperatively (threads can't
        be force-killed) and raise rather than keep running past it.
        """
        with self._lock:
            existing = self._inflight.get(job_id)
            if existing is not None and not existing.done():
                return False

            deadline = (time.monotonic() + timeout_seconds) if timeout_seconds else None
            cancel_check: Callable[[], bool] = (
                (lambda: time.monotonic() > deadline) if deadline is not None else (lambda: False)
            )

            def _run() -> None:
                started = time.perf_counter()
                try:
                    fn(cancel_check)
                except Exception as exc:  # noqa: BLE001 - never swallow, record it
                    logger.exception("Video extraction job %s failed", job_id)
                    db.update_job(
                        job_id,
                        status="failed",
                        error=f"{exc.__class__.__name__}: {exc}\n{traceback.format_exc()}",
                    )
                finally:
                    elapsed = time.perf_counter() - started
                    logger.info("Video extraction job %s finished in %.2fs", job_id, elapsed)

            future = self._executor.submit(_run)
            self._inflight[job_id] = future
            return True

    def shutdown(self, wait: bool = False) -> None:  # pragma: no cover - process lifecycle
        self._executor.shutdown(wait=wait)


_runner: JobRunner | None = None
_runner_lock = threading.Lock()


def get_job_runner() -> JobRunner:
    global _runner
    with _runner_lock:
        if _runner is None:
            _runner = JobRunner()
        return _runner


def reset_job_runner() -> None:  # pragma: no cover - test hook
    global _runner
    with _runner_lock:
        _runner = None

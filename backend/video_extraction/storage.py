"""Local-filesystem storage backend for video extraction jobs.

ADAPTATION NOTE: the original task brief assumed an existing Wasabi/S3
integration. There is none in this repository (the stale root-level
markdown docs that imply otherwise do not match the actual code). This
module is a thin storage abstraction -- `save`, `open_path`, `delete`,
`local_url` -- so a real object-storage backend (Wasabi/S3 via boto3) can
be dropped in later as a second implementation of the same interface
without touching any caller. Today's implementation keeps every job's
files under a private, non-web-served directory on local disk; nothing
in this tree is ever exposed except through the authenticated,
ownership-checked `/api/video-extraction/{job_id}/files/...` endpoint.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from backend.video_extraction.config import get_settings
from backend.video_extraction.exceptions import StorageError


class VideoExtractionStorage:
    """Save/read/delete files for one video-extraction job tree.

    Layout per job: ``<root>/<job_id>/{original<ext>, extracted/<view>_<n>.png,
    masks/..., overlays/..., cutouts/..., measurements.json,
    extraction_report.json, quality_report.json}``.
    """

    def __init__(self, root: Path | None = None):
        self.root = root or get_settings().storage_root
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            os.chmod(self.root, 0o700)
        except OSError as exc:  # pragma: no cover - environment dependent
            raise StorageError(f"Cannot create storage root {self.root}: {exc}") from exc

    def job_dir(self, job_id: str) -> Path:
        return self.root / self._safe_segment(job_id)

    def save(self, job_id: str, relative_path: str, data: bytes) -> Path:
        """Write ``data`` to ``<job_dir>/<relative_path>`` and return the path."""
        target = self._resolve(job_id, relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(target.parent, 0o700)
        except OSError:  # pragma: no cover - best effort, e.g. on some filesystems
            pass
        tmp = target.with_suffix(target.suffix + ".tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, target)
            os.chmod(target, 0o600)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise StorageError(f"Could not write {relative_path} for job {job_id}: {exc}") from exc
        return target

    def open_path(self, job_id: str, relative_path: str) -> Path:
        """Return the absolute path for an existing stored file, or raise."""
        target = self._resolve(job_id, relative_path)
        if not target.is_file():
            raise StorageError(f"No such file for job {job_id}: {relative_path}")
        return target

    def exists(self, job_id: str, relative_path: str) -> bool:
        try:
            target = self._resolve(job_id, relative_path)
        except StorageError:
            return False
        return target.is_file()

    def delete(self, job_id: str) -> None:
        """Remove the entire job directory tree. Safe to call if absent."""
        job_dir = self.job_dir(job_id)
        shutil.rmtree(job_dir, ignore_errors=True)

    def local_url(self, job_id: str, relative_path: str) -> str:
        """Authenticated-only URL the frontend should use to fetch this file.

        This is NOT a public/static URL -- it is routed through the same
        auth + ownership dependency as every other endpoint. A future
        Wasabi/S3 backend would instead return a short-lived signed URL
        here, with callers unchanged.
        """
        return f"/api/video-extraction/{job_id}/files/{relative_path}"

    @staticmethod
    def _safe_segment(value: str) -> str:
        """Reject path traversal / separators in an id used as a directory name."""
        if not value or value != Path(value).name or ".." in value:
            raise StorageError(f"Unsafe path segment: {value!r}")
        return value

    def _resolve(self, job_id: str, relative_path: str) -> Path:
        job_dir = self.job_dir(job_id)
        candidate = (job_dir / relative_path).resolve()
        job_dir_resolved = job_dir.resolve()
        try:
            candidate.relative_to(job_dir_resolved)
        except ValueError as exc:
            raise StorageError(f"Path escapes job directory: {relative_path!r}") from exc
        return candidate


_storage: VideoExtractionStorage | None = None


def get_storage() -> VideoExtractionStorage:
    global _storage
    if _storage is None:
        _storage = VideoExtractionStorage()
    return _storage


def reset_storage_cache() -> None:
    global _storage
    _storage = None

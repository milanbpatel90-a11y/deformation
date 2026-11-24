"""SQLite-backed job state store for video extraction jobs.

ADAPTATION NOTE: there is no Celery/Redis/distributed-queue in this repo
(Railway hosting + FastAPI only). Job state is therefore kept in a small
SQLite table (same pattern as backend/auth.py's users table) so job
status survives a process restart, instead of only living in memory. The
actual processing work runs via backend.video_extraction.jobs' in-process
thread pool. A real distributed queue (Celery+Redis/RQ) would be a
future upgrade if Redis becomes available on the hosting platform.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from backend.video_extraction.config import get_settings


def _db_path() -> Path:
    path = get_settings().db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(str(_db_path()), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    connection = _connect()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_video_extraction_database() -> None:
    with _db() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS video_extraction_jobs (
                id TEXT PRIMARY KEY,
                owner_user_id TEXT NOT NULL,
                status TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                video_relpath TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                progress_message TEXT,
                error TEXT,
                report_json TEXT,
                selected_frames_json TEXT,
                measurements_json TEXT,
                calibration_json TEXT,
                deform_result_json TEXT
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_videox_jobs_owner ON video_extraction_jobs(owner_user_id)"
        )


initialize_video_extraction_database()


def create_job(job_id: str, owner_user_id: str, original_filename: str) -> dict:
    now = int(time.time())
    with _db() as connection:
        connection.execute(
            """
            INSERT INTO video_extraction_jobs
                (id, owner_user_id, status, original_filename, created_at, updated_at)
            VALUES (?, ?, 'uploaded', ?, ?, ?)
            """,
            (job_id, owner_user_id, original_filename, now, now),
        )
    return get_job(job_id)


def get_job(job_id: str) -> dict | None:
    with _db() as connection:
        row = connection.execute(
            "SELECT * FROM video_extraction_jobs WHERE id = ?", (job_id,)
        ).fetchone()
    return _deserialize(row) if row else None


def list_jobs(owner_user_id: str | None = None) -> list[dict]:
    with _db() as connection:
        if owner_user_id is None:
            rows = connection.execute(
                "SELECT * FROM video_extraction_jobs ORDER BY created_at DESC"
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM video_extraction_jobs WHERE owner_user_id = ? ORDER BY created_at DESC",
                (owner_user_id,),
            ).fetchall()
    return [_deserialize(row) for row in rows]


_JSON_FIELDS = {"report_json", "selected_frames_json", "measurements_json", "calibration_json", "deform_result_json"}
_JSON_TO_PLAIN = {
    "report_json": "report",
    "selected_frames_json": "selected_frames",
    "measurements_json": "measurements",
    "calibration_json": "calibration",
    "deform_result_json": "deform_result",
}


def update_job(job_id: str, **fields: Any) -> dict | None:
    if not fields:
        return get_job(job_id)
    columns: list[str] = []
    values: list[Any] = []
    for key, value in fields.items():
        db_key = key if key in _JSON_FIELDS else (
            f"{key}_json" if f"{key}_json" in _JSON_FIELDS else key
        )
        if db_key in _JSON_FIELDS:
            columns.append(f"{db_key} = ?")
            values.append(json.dumps(value) if value is not None else None)
        else:
            columns.append(f"{key} = ?")
            values.append(value)
    columns.append("updated_at = ?")
    values.append(int(time.time()))
    values.append(job_id)
    with _db() as connection:
        connection.execute(
            f"UPDATE video_extraction_jobs SET {', '.join(columns)} WHERE id = ?", values
        )
    return get_job(job_id)


def delete_job(job_id: str) -> None:
    with _db() as connection:
        connection.execute("DELETE FROM video_extraction_jobs WHERE id = ?", (job_id,))


def _deserialize(row: sqlite3.Row) -> dict:
    data = dict(row)
    for json_field, plain_name in _JSON_TO_PLAIN.items():
        raw = data.pop(json_field, None)
        data[plain_name] = json.loads(raw) if raw else None
    return data

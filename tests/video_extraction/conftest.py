"""Shared fixtures for video-extraction tests.

Isolates every test behind its own temp storage root + SQLite DB path (via
env vars + the package's own cache-reset hooks), so tests never touch the
developer's real `storage/video_extraction` or `data/video_extraction.sqlite3`.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_video_extraction_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEOX_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("VIDEOX_DB_PATH", str(tmp_path / "videox.sqlite3"))

    from backend.video_extraction import config, storage, db

    config.reset_settings_cache()
    storage.reset_storage_cache()
    db.initialize_video_extraction_database()
    yield
    config.reset_settings_cache()
    storage.reset_storage_cache()

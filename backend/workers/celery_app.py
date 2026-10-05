from __future__ import annotations

import os

from celery import Celery

from backend.logging_config import configure_logging

configure_logging()

broker = os.getenv("CELERY_BROKER_URL", os.getenv("REDIS_URL", "redis://redis:6379/0"))
backend = os.getenv("CELERY_RESULT_BACKEND", broker)

celery_app = Celery("deformation", broker=broker, backend=backend)
celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_time_limit=900,
    task_soft_time_limit=840,
)

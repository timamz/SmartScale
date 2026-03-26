import os

from celery import Celery

BROKER_URL = os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@localhost:5672//")

celery_app = Celery("smartscale_worker", broker=BROKER_URL, include=["worker_tasks"])
celery_app.conf.update(task_track_started=True)

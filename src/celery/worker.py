import os
from celery import Celery
from src.config import settings
from src.main import process_video_background


celery = Celery(settings.celery_app_name)
celery.conf.broker_url = settings.celery_broker_url
celery.conf.result_backend = settings.celery_result_backend

# Optional configuration
celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


def get_celery():
    return celery


@celery.task(name="process_video")
def process_video(task_id: int, video_path: str, name: str):
    process_video_background(task_id, video_path, name)
    return "Video processed successfully"

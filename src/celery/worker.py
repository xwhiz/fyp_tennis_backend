import os
import sys
import gc
import threading
import torch
from celery import Celery
from src.config import settings
from src.core.ball_detector import BallDetector
from src.core.bounce_detector import BounceDetector
from src.core.court_detection_net import CourtDetectorNet
from src.core.person_detector import PersonDetector
from src.core.process_video import process_video, cleanup_memory
from src.db.utils import update_task_status


celery = Celery(settings.celery_app_name)
celery.conf.broker_url = settings.celery_broker_url
celery.conf.result_backend = settings.celery_result_backend

# Windows compatibility: use 'threads' pool on Windows, 'prefork' on Unix
# 'threads' allows concurrent task execution using threads (works on Windows)
# 'solo' runs tasks sequentially (one at a time) - not suitable for parallel processing
# 'prefork' uses multiprocessing, which has issues on Windows
# Note: When starting the worker on Windows, use: celery -A src.celery.worker.celery worker --pool=threads --concurrency=4
pool_type = "threads" if sys.platform == "win32" else "prefork"

# Optional configuration
celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    pool_type=pool_type,
)

# Set default pool type and concurrency for Windows compatibility
# This may not always work, so also use --pool=threads --concurrency=2 when starting worker on Windows
if sys.platform == "win32":
    celery.conf.worker_pool = "threads"
# Set concurrency for all platforms
celery.conf.worker_concurrency = settings.celery_worker_concurrency  # Number of concurrent tasks

# Module-level variables for models (loaded once per worker process)
_ball_detector = None
_court_detector = None
_person_detector = None
_bounce_detector = None
_device = None
_load_lock = threading.Lock()  # Lock for thread-safe model loading


def _load_models():
    """Load ML models once per worker process (lazy loading) - thread-safe"""
    global _ball_detector, _court_detector, _person_detector, _bounce_detector, _device

    # Double-checked locking pattern for thread-safe lazy loading
    if _ball_detector is None:
        with _load_lock:
            # Check again after acquiring lock (another thread might have loaded it)
            if _ball_detector is None:
                _device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[CELERY WORKER]: Loading models on device: {_device}")
                _ball_detector = BallDetector("./src/track_net_weights.pt", _device)
                _court_detector = CourtDetectorNet(
                    "./src/model_tennis_court_det.pt", _device
                )
                _person_detector = PersonDetector(_device)
                _bounce_detector = BounceDetector("./src/ctb_regr_bounce.cbm")
                print("[CELERY WORKER]: All models loaded successfully")

    return _ball_detector, _court_detector, _person_detector, _bounce_detector


def get_celery():
    return celery


@celery.task(name="process_video_task", bind=True)
def process_video_task(self, task_id: int, video_path: str, name: str):
    """Celery task to process video"""
    try:
        # Update task status to pending
        update_task_status(task_id, "pending", 0.0, "Waiting in queue to be processed")

        # Load models (lazy loading - only loads once per worker)
        ball_detector, court_detector, person_detector, bounce_detector = _load_models()

        # Process the video
        process_video(
            ball_detector=ball_detector,
            court_detector=court_detector,
            person_detector=person_detector,
            bounce_detector=bounce_detector,
            video_path=video_path,
            task_id=task_id,
            name=name,
        )

        return {
            "success": True,
            "task_id": task_id,
            "message": "Video processed successfully",
        }

    except Exception as e:
        error_msg = f"Error processing video {task_id}: {str(e)}"
        print(f"[CELERY WORKER ERROR]: {error_msg}")
        update_task_status(task_id, "failed", 0.0, error_msg)
        # Re-raise to mark task as failed in Celery
        raise
    finally:
        # Ensure memory cleanup even if task fails
        cleanup_memory(_device)
        gc.collect()
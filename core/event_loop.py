import os
import time

from sqlmodel import Session, select

from core.process_video import process_video
from db.engine import Engine
from db.utils import update_task_status
from models.background_task_model import BackgroundTask


class EventLoop:
    def __init__(self, app):
        self.app = app
        self.tasks = []
        self.is_running = True

        self.load_pending_tasks()

    def load_pending_tasks(self):
        with Session(Engine.instance()) as session:
            statement = select(
                BackgroundTask.id, BackgroundTask.video_path, BackgroundTask.name
            ).where(
                (
                    BackgroundTask.status != "completed"
                    # & (BackgroundTask.status != "failed")
                )
            )
            tasks = session.exec(statement).all()
            tasks = [
                {"id": task.id, "video_path": task.video_path, "name": task.name}
                for task in tasks
            ]
            self.tasks = tasks

    def add_task(self, task):
        self.tasks.append(task)

    def run(self):
        while self.is_running:
            if not self.tasks:
                time.sleep(1)
                continue

            task = self.tasks.pop(0)
            try:
                process_video(self.app, task["video_path"], task["id"], task["name"])
            except Exception as e:
                print(f"Error processing video {task['id']}: {str(e)}")
                update_task_status(task["id"], "failed", 0, "Error processing video")

    def stop(self):
        self.is_running = False
        self.tasks = []

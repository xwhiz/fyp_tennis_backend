import os
import threading
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

        self.number_of_threads = os.getenv("NUMBER_OF_THREADS", 1)

        self.load_pending_tasks()

    def load_pending_tasks(self):
        with Session(Engine.instance()) as session:
            statement = select(
                BackgroundTask.id, BackgroundTask.video_path, BackgroundTask.name
            ).where(
                (BackgroundTask.status != "completed")
                and (BackgroundTask.status != "failed")
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

            threads = []
            number_of_threads = min(self.number_of_threads, len(self.tasks))
            for _ in range(number_of_threads):
                try:
                    task = self.tasks.pop(0)
                    thread = threading.Thread(
                        target=process_video,
                        args=(self.app, task["video_path"], task["id"], task["name"]),
                    )
                    thread.start()
                    threads.append(thread)
                    update_task_status(
                        task["id"], "completed", 9, "Video processed successfully"
                    )
                except Exception as e:
                    print(f"Error processing video {task['id']}: {str(e)}")

            for thread in threads:
                thread.join()

    def stop(self):
        self.is_running = False

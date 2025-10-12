from datetime import datetime
import json
from typing import Annotated
from fastapi import Depends
from sqlmodel import SQLModel, Session, select
from db.engine import Engine
from models.background_task_model import BackgroundTask
from models.ball_track_model import BallTrackModel
from models.bounces_model import BouncesModel
from models.direction_change_indices_model import DirectionChangeIndicesModel
from models.speed_model import SpeedModel
from models.thumbnail_model import ThumbnailModel
from models.video_paths_model import VideoPathsModel


def create_all():
    """Create all tables from the models"""
    SQLModel.metadata.create_all(Engine.instance())


def get_session():
    with Session(Engine.instance()) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


def update_task_status(
    task_id: int,
    status: str,
    progress: int = None,
    description: str = None,
):
    """Update task status in database"""
    engine = Engine.instance()
    with Session(engine) as session:
        statement = select(BackgroundTask).where(BackgroundTask.id == task_id)
        task = session.exec(statement).first()
        if task:
            task.status = status
            task.updated_at = datetime.now()
            if description is not None:
                task.description = description
            if progress is not None:
                task.progress = progress
            session.add(task)
            session.commit()


def to_float(x) -> list:
    if x is None:
        return None
    return [float(i) if i is not None else None for i in x]


def save_ball_track_in_db(task_id: int, ball_track: list):
    processed_track = {i: to_float(ball_track[i]) for i in range(len(ball_track))}
    with Session(Engine.instance()) as session:
        session.add(
            BallTrackModel(task_id=task_id, ball_track=json.dumps(processed_track))
        )
        session.commit()


def save_bounces_in_db(task_id: int, bounces: set):
    with Session(Engine.instance()) as session:
        session.add(BouncesModel(task_id=task_id, bounces=json.dumps(list(bounces))))
        session.commit()


def save_direction_change_indices_in_db(task_id: int, direction_change_indices: list):
    with Session(Engine.instance()) as session:
        session.add(
            DirectionChangeIndicesModel(
                task_id=task_id,
                direction_change_indices=json.dumps(direction_change_indices),
            )
        )
        session.commit()


def save_speed_in_db(task_id: int, speed: dict):
    processed_speed = {k: v.to_dict() for k, v in speed.items()}
    with Session(Engine.instance()) as session:
        session.add(SpeedModel(task_id=task_id, speed=json.dumps(processed_speed)))
        session.commit()


def save_video_paths_in_db(
    task_id: int, name: str, output_path: str, minimap_path: str
):
    with Session(Engine.instance()) as session:
        session.add(
            VideoPathsModel(
                task_id=task_id,
                name=name,
                output_path=output_path,
                minimap_path=minimap_path,
            )
        )
        session.commit()


def save_thumbnail_in_db(task_id: int, thumbnail_path: str):
    with Session(Engine.instance()) as session:
        session.add(ThumbnailModel(task_id=task_id, thumbnail_path=thumbnail_path))
        session.commit()

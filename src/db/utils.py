from datetime import datetime
import json
from typing import Annotated

from fastapi import Depends
from sqlmodel import Session, select

from src.db.engine import Engine
from src.models.background_task import BackgroundTask
from src.models.ball_track import BallTrack
from src.models.bounces import Bounces
from src.models.user import User  # noqa: F401 - registers `users` for owner_id FKs (e.g. Celery worker)
from src.models.direction_change_indices import DirectionChangeIndices
from src.models.homography_matrices import HomographyMatrices
from src.models.player_heatmap_data import PlayerHeatmapData
from src.models.speed import Speed
from src.models.player_positions import PlayerPositions
from src.models.thumbnail import Thumbnail
from src.models.video_paths import VideoPaths


def get_session():
    with Session(Engine.instance()) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


def _owner_id_for_task(task_id: int) -> str:
    with Session(Engine.instance()) as session:
        task = session.exec(select(BackgroundTask).where(BackgroundTask.id == task_id)).first()
        if task is None:
            raise ValueError(f"Background task {task_id} not found")
        return task.owner_id


def update_task_status(
    task_id: int,
    status: str,
    progress: float = None,
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


def update_upload_progress(task_id: int, uploaded_size: int):
    """Update upload progress in database"""
    engine = Engine.instance()
    with Session(engine) as session:
        statement = select(BackgroundTask).where(BackgroundTask.id == task_id)
        task = session.exec(statement).first()
        if task:
            task.uploaded_size = uploaded_size
            task.updated_at = datetime.now()
            session.add(task)
            session.commit()


def to_float(x) -> list:
    if x is None:
        return None
    return [float(i) if i is not None else None for i in x]


def save_ball_track_in_db(task_id: int, ball_track: list):
    processed_track = {i: to_float(ball_track[i]) for i in range(len(ball_track))}
    oid = _owner_id_for_task(task_id)
    with Session(Engine.instance()) as session:
        session.add(
            BallTrack(task_id=task_id, ball_track=json.dumps(processed_track), owner_id=oid),
        )
        session.commit()


def save_bounces_in_db(task_id: int, bounces: dict, serve_frames: set = None):
    if serve_frames is None:
        serve_frames = set()
    processed_bounces = {
        k: {"position": to_float(v), "serve": k in serve_frames}
        for k, v in bounces.items()
    }
    oid = _owner_id_for_task(task_id)
    with Session(Engine.instance()) as session:
        session.add(Bounces(task_id=task_id, bounces=json.dumps(processed_bounces), owner_id=oid))
        session.commit()


def save_direction_change_indices_in_db(task_id: int, direction_change_indices: list):
    processed_direction_change_indices = {
        k: to_float(v) for k, v in direction_change_indices.items()
    }
    oid = _owner_id_for_task(task_id)
    with Session(Engine.instance()) as session:
        session.add(
            DirectionChangeIndices(
                task_id=task_id,
                direction_change_indices=json.dumps(processed_direction_change_indices),
                owner_id=oid,
            ),
        )
        session.commit()


def save_speed_in_db(task_id: int, speed: dict):
    processed_speed = {k: v.to_dict() for k, v in speed.items()}
    oid = _owner_id_for_task(task_id)
    with Session(Engine.instance()) as session:
        session.add(Speed(task_id=task_id, speeds=json.dumps(processed_speed), owner_id=oid))
        session.commit()


def save_video_paths_in_db(
    task_id: int, name: str, output_path: str, minimap_path: str
):
    oid = _owner_id_for_task(task_id)
    with Session(Engine.instance()) as session:
        session.add(
            VideoPaths(
                task_id=task_id,
                name=name,
                output_path=output_path,
                minimap_path=minimap_path,
                owner_id=oid,
            ),
        )
        session.commit()


def save_heatmap_data_in_db(
    task_id: int,
    top_court_points: list,
    bottom_court_points: list,
):
    """Upsert court-space points for player heatmaps (one row per task)."""
    top_json = [[float(x), float(y)] for x, y in top_court_points]
    bottom_json = [[float(x), float(y)] for x, y in bottom_court_points]
    oid = _owner_id_for_task(task_id)
    with Session(Engine.instance()) as session:
        existing = session.exec(
            select(PlayerHeatmapData).where(PlayerHeatmapData.task_id == task_id)
        ).first()
        if existing:
            existing.top_points = top_json
            existing.bottom_points = bottom_json
            session.add(existing)
        else:
            session.add(
                PlayerHeatmapData(
                    task_id=task_id,
                    top_points=top_json,
                    bottom_points=bottom_json,
                    owner_id=oid,
                ),
            )
        session.commit()


def save_homography_matrices_in_db(task_id: int, homography_matrices: list):
    """Upsert per-frame homography (image->court). Each element is 3x3 array or None."""

    def to_json(m):
        if m is None:
            return None
        return [[float(x) for x in row] for row in m.tolist()]

    matrices_json = [to_json(m) for m in homography_matrices]
    oid = _owner_id_for_task(task_id)
    with Session(Engine.instance()) as session:
        existing = session.exec(
            select(HomographyMatrices).where(HomographyMatrices.task_id == task_id)
        ).first()
        if existing:
            existing.matrices = matrices_json
            session.add(existing)
        else:
            session.add(
                HomographyMatrices(task_id=task_id, matrices=matrices_json, owner_id=oid),
            )
        session.commit()


def save_player_positions_in_db(task_id: int, player_top: list, player_bottom: list):
    positions = {}
    for i in range(len(player_top)):
        top_bbox = (
            [float(x) for x in player_top[i][0]]
            if player_top[i] is not None
            else None
        )
        bottom_bbox = (
            [float(x) for x in player_bottom[i][0]]
            if player_bottom[i] is not None
            else None
        )
        positions[i] = {"top": top_bbox, "bottom": bottom_bbox}
    oid = _owner_id_for_task(task_id)
    with Session(Engine.instance()) as session:
        session.add(
            PlayerPositions(task_id=task_id, positions=json.dumps(positions), owner_id=oid),
        )
        session.commit()


def save_thumbnail_in_db(task_id: int, thumbnail_path: str):
    oid = _owner_id_for_task(task_id)
    with Session(Engine.instance()) as session:
        session.add(Thumbnail(task_id=task_id, thumbnail_path=thumbnail_path, owner_id=oid))
        session.commit()

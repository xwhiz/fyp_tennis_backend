from datetime import datetime
import json
from typing import Annotated
from fastapi import Depends
from sqlmodel import SQLModel, Session, select
from src.db.engine import Engine
from src.models.background_task import BackgroundTask
from src.models.ball_track import BallTrack
from src.models.bounces import Bounces
from src.models.direction_change_indices import DirectionChangeIndices
from src.models.speed import Speed
from src.models.player_positions import PlayerPositions
from src.models.thumbnail import Thumbnail
from src.models.video_paths import VideoPaths
from src.models.shot_annotations import ShotAnnotations
from src.models.model_metrics import ModelMetrics


def get_session():
    with Session(Engine.instance()) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


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
    with Session(Engine.instance()) as session:
        session.add(BallTrack(task_id=task_id, ball_track=json.dumps(processed_track)))
        session.commit()


def save_bounces_in_db(task_id: int, bounces: dict, serve_frames: set = None):
    if serve_frames is None:
        serve_frames = set()
    processed_bounces = {
        k: {"position": to_float(v), "serve": k in serve_frames}
        for k, v in bounces.items()
    }

    with Session(Engine.instance()) as session:
        session.add(Bounces(task_id=task_id, bounces=json.dumps(processed_bounces)))
        session.commit()


def save_direction_change_indices_in_db(task_id: int, direction_change_indices: list):
    processed_direction_change_indices = {
        k: to_float(v) for k, v in direction_change_indices.items()
    }
    with Session(Engine.instance()) as session:
        session.add(
            DirectionChangeIndices(
                task_id=task_id,
                direction_change_indices=json.dumps(processed_direction_change_indices),
            )
        )
        session.commit()


def save_speed_in_db(task_id: int, speed: dict):
    processed_speed = {k: v.to_dict() for k, v in speed.items()}
    with Session(Engine.instance()) as session:
        session.add(Speed(task_id=task_id, speeds=json.dumps(processed_speed)))
        session.commit()


def save_video_paths_in_db(
    task_id: int, name: str, output_path: str, minimap_path: str
):
    with Session(Engine.instance()) as session:
        session.add(
            VideoPaths(
                task_id=task_id,
                name=name,
                output_path=output_path,
                minimap_path=minimap_path,
            )
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

    with Session(Engine.instance()) as session:
        session.add(
            PlayerPositions(task_id=task_id, positions=json.dumps(positions))
        )
        session.commit()


def save_thumbnail_in_db(task_id: int, thumbnail_path: str):
    with Session(Engine.instance()) as session:
        session.add(Thumbnail(task_id=task_id, thumbnail_path=thumbnail_path))
        session.commit()


def save_shot_annotation_in_db(
    task_id: int,
    frame_index: int,
    player_position_top: dict = None,
    player_position_bottom: dict = None,
    ball_position: dict = None,
    player_image_path: str = None,
    player_image_paths: dict = None,
    predicted_shot_type: str = "unknown",
):
    """Save shot annotation data to database."""
    with Session(Engine.instance()) as session:
        shot_annotation = ShotAnnotations(
            task_id=task_id,
            frame_index=frame_index,
            player_position_top=player_position_top,
            player_position_bottom=player_position_bottom,
            ball_position=ball_position,
            player_image_path=player_image_path,
            player_image_paths=player_image_paths,
            predicted_shot_type=predicted_shot_type,
            annotated_shot_type="unknown",
            discarded=False,
        )
        session.add(shot_annotation)
        session.commit()
        session.refresh(shot_annotation)
        return shot_annotation.id


def get_unannotated_shots(limit: int = None):
    """Get all shots where annotated_shot_type is 'unknown' and not discarded."""
    with Session(Engine.instance()) as session:
        statement = select(ShotAnnotations).where(
            ShotAnnotations.annotated_shot_type == "unknown",
            ShotAnnotations.discarded == False
        )
        if limit:
            statement = statement.limit(limit)
        shots = session.exec(statement).all()
        return list(shots)


def update_shot_annotation(shot_id: int, annotated_shot_type: str):
    """Update annotated_shot_type for a shot."""
    with Session(Engine.instance()) as session:
        statement = select(ShotAnnotations).where(ShotAnnotations.id == shot_id)
        shot = session.exec(statement).first()
        if shot:
            shot.annotated_shot_type = annotated_shot_type
            shot.updated_at = datetime.now()
            session.add(shot)
            session.commit()
            return True
        return False


def discard_shot_annotation(shot_id: int):
    """Mark a shot annotation as discarded (soft delete)."""
    with Session(Engine.instance()) as session:
        statement = select(ShotAnnotations).where(ShotAnnotations.id == shot_id)
        shot = session.exec(statement).first()
        if shot:
            shot.discarded = True
            shot.updated_at = datetime.now()
            session.add(shot)
            session.commit()
            return True
        return False


def get_all_annotated_shots():
    """Get all shots where annotated_shot_type is not 'unknown' and not discarded."""
    with Session(Engine.instance()) as session:
        statement = select(ShotAnnotations).where(
            ShotAnnotations.annotated_shot_type != "unknown",
            ShotAnnotations.discarded == False
        )
        shots = session.exec(statement).all()
        return list(shots)


def get_model_metrics():
    """Get the latest model metrics."""
    with Session(Engine.instance()) as session:
        statement = select(ModelMetrics).order_by(ModelMetrics.updated_at.desc())
        metrics = session.exec(statement).first()
        return metrics


def update_model_metrics(
    training_status: str,
    accuracy: float = None,
    precision: float = None,
    recall: float = None,
    f1_score: float = None,
    total_samples: int = None,
):
    """Update or create model metrics."""
    with Session(Engine.instance()) as session:
        # Get existing metrics or create new
        statement = select(ModelMetrics).order_by(ModelMetrics.updated_at.desc())
        metrics = session.exec(statement).first()
        
        if metrics is None:
            metrics = ModelMetrics()
            session.add(metrics)
        
        metrics.training_status = training_status
        if accuracy is not None:
            metrics.accuracy = accuracy
        if precision is not None:
            metrics.precision = precision
        if recall is not None:
            metrics.recall = recall
        if f1_score is not None:
            metrics.f1_score = f1_score
        if total_samples is not None:
            metrics.total_samples = total_samples
        
        metrics.updated_at = datetime.now()
        session.add(metrics)
        session.commit()
        return metrics

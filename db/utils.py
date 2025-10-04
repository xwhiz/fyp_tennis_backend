from datetime import datetime
from typing import Annotated
from fastapi import Depends
from sqlmodel import SQLModel, Session, select
from db.engine import Engine
from models.background_task_model import BackgroundTask


def create_all():
    """Create all tables from the models"""
    SQLModel.metadata.create_all(Engine.instance())


def get_session():
    with Session(Engine.instance()) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


def update_task_status(
    engine,
    task_id: int,
    status: str,
    progress: int = None,
    description: str = None,
):
    """Update task status in database"""
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

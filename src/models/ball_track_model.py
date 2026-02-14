from datetime import datetime
from sqlmodel import JSON, Field, SQLModel


class BallTrackModel(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    task_id: int = Field(default=None)
    ball_track: str = Field(sa_type=JSON)
    created_at: datetime = Field(default=datetime.now())
    updated_at: datetime = Field(default=datetime.now())

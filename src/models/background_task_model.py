from datetime import datetime
from sqlmodel import Field, SQLModel


class BackgroundTask(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    progress: int = Field(default=0)
    total_steps: int = Field(default=9)
    status: str = Field(default="pending")
    name: str = Field(default="")
    video_path: str = Field(default="")
    description: str = Field(default="")
    created_at: datetime = Field(default=datetime.now())
    updated_at: datetime = Field(default=datetime.now())

from datetime import datetime
from sqlmodel import Field, SQLModel


class ThumbnailModel(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    task_id: int = Field(default=None)
    thumbnail_path: str = Field()
    created_at: datetime = Field(default=datetime.now())
    updated_at: datetime = Field(default=datetime.now())

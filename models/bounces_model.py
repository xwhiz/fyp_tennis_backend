from datetime import datetime
from sqlmodel import Field, JSON, SQLModel


class BouncesModel(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    task_id: int = Field(default=None)
    bounces: str = Field(sa_type=JSON)
    created_at: datetime = Field(default=datetime.now())
    updated_at: datetime = Field(default=datetime.now())

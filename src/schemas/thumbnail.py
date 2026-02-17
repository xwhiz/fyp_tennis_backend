from datetime import datetime
from pydantic import BaseModel


class ThumbnailSchema(BaseModel):
    """Pydantic schema for Thumbnail model."""
    
    id: int
    task_id: int
    thumbnail_path: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

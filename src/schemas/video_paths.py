from datetime import datetime
from pydantic import BaseModel


class VideoPathsSchema(BaseModel):
    """Pydantic schema for VideoPaths model."""
    
    id: int
    task_id: int
    name: str
    output_path: str
    minimap_path: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

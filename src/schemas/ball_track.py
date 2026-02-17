from datetime import datetime
from typing import Any, Dict
from pydantic import BaseModel


class BallTrackSchema(BaseModel):
    """Pydantic schema for BallTrack model."""
    
    id: int
    task_id: int
    ball_track: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

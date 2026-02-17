from datetime import datetime
from typing import Any, Dict
from pydantic import BaseModel


class BouncesSchema(BaseModel):
    """Pydantic schema for Bounces model."""
    
    id: int
    task_id: int
    bounces: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

from datetime import datetime
from typing import Any, Dict
from pydantic import BaseModel


class DirectionChangeIndicesSchema(BaseModel):
    """Pydantic schema for DirectionChangeIndices model."""
    
    id: int
    task_id: int
    direction_change_indices: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

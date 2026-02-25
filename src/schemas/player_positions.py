from datetime import datetime
from typing import Any, Dict
from pydantic import BaseModel


class PlayerPositionsSchema(BaseModel):
    """Pydantic schema for PlayerPositions model."""

    id: int
    task_id: int
    positions: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

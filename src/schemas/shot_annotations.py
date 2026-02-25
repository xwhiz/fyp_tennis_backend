from pydantic import BaseModel
from typing import Optional


class ShotAnnotationSchema(BaseModel):
    id: int
    task_id: int
    frame_index: int
    player_position_top: Optional[dict] = None
    player_position_bottom: Optional[dict] = None
    ball_position: Optional[dict] = None
    player_image_path: Optional[str] = None  # Kept for backward compatibility
    player_image_paths: Optional[dict] = None  # New: {"top": [paths], "bottom": [paths]}
    predicted_shot_type: str
    annotated_shot_type: str
    discarded: bool = False
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class UpdateAnnotationSchema(BaseModel):
    annotated_shot_type: str


class ModelMetricsSchema(BaseModel):
    id: int
    training_status: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    total_samples: Optional[int] = None
    last_trained_at: Optional[str] = None
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True




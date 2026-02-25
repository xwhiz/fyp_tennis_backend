from sqlalchemy import JSON, BigInteger, String, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base, TimestampMixin


class ShotAnnotations(Base, TimestampMixin):
    """Model for storing shot annotation data."""

    __tablename__ = "shot_annotations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    frame_index: Mapped[int] = mapped_column(BigInteger, nullable=False)
    player_position_top: Mapped[dict] = mapped_column(JSON, nullable=True)
    player_position_bottom: Mapped[dict] = mapped_column(JSON, nullable=True)
    ball_position: Mapped[dict] = mapped_column(JSON, nullable=True)
    player_image_path: Mapped[str] = mapped_column(String, nullable=True)  # Kept for backward compatibility
    player_image_paths: Mapped[dict] = mapped_column(JSON, nullable=True)  # New: stores all player images {"top": [paths], "bottom": [paths]}
    predicted_shot_type: Mapped[str] = mapped_column(String, default="unknown", nullable=False)
    annotated_shot_type: Mapped[str] = mapped_column(String, default="unknown", nullable=False)
    discarded: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)




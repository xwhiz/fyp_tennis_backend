from sqlalchemy import JSON, BigInteger, Boolean, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base, TimestampMixin


class ShotAnnotation(Base, TimestampMixin):
    __tablename__ = "shot_annotations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    frame_index: Mapped[int] = mapped_column(BigInteger, nullable=False)
    player_position_top: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    player_position_bottom: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    ball_position: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    player_image_path: Mapped[str | None] = mapped_column(String, nullable=True)
    predicted_shot_type: Mapped[str] = mapped_column(String, nullable=False, default="unknown")
    annotated_shot_type: Mapped[str] = mapped_column(String, nullable=False, default="unknown")
    discarded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    player_image_paths: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    owner_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

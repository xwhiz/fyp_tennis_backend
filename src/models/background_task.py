from sqlalchemy import BigInteger, Boolean, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base, TimestampMixin


class BackgroundTask(Base, TimestampMixin):
    """Model for background tasks."""
    
    __tablename__ = "background_tasks"
    
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(default="created")
    name: Mapped[str] = mapped_column(default="")
    video_path: Mapped[str] = mapped_column(default="")
    description: Mapped[str] = mapped_column(default="")
    total_upload_size: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    uploaded_size: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    is_uploaded_fully: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    owner_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

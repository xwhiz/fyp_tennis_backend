from sqlalchemy import BigInteger, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base, TimestampMixin


class BackgroundTask(Base, TimestampMixin):
    """Model for background tasks."""
    
    __tablename__ = "background_tasks"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(default="created")
    name: Mapped[str] = mapped_column(default="")
    video_path: Mapped[str] = mapped_column(default="")
    description: Mapped[str] = mapped_column(default="")

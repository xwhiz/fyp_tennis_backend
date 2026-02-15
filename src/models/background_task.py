from sqlalchemy import BigInteger, String
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base, TimestampMixin


class BackgroundTask(Base, TimestampMixin):
    """Model for background tasks."""
    
    __tablename__ = "background_tasks"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    progress: Mapped[int] = mapped_column(default=0)
    total_steps: Mapped[int] = mapped_column(default=10)
    status: Mapped[str] = mapped_column(default="created")
    name: Mapped[str] = mapped_column(default="")
    video_path: Mapped[str] = mapped_column(default="")
    description: Mapped[str] = mapped_column(default="")

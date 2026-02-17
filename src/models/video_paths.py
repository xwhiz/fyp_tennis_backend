from sqlalchemy import BigInteger, String
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base, TimestampMixin


class VideoPaths(Base, TimestampMixin):
    """Model for storing video paths data."""
    
    __tablename__ = "video_paths"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    output_path: Mapped[str] = mapped_column(String(255), nullable=False)
    minimap_path: Mapped[str] = mapped_column(String(255), nullable=False)

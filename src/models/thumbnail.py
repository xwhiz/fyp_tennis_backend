from sqlalchemy import BigInteger, String
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base, TimestampMixin


class Thumbnail(Base, TimestampMixin):
    """Model for storing thumbnail data."""
    
    __tablename__ = "thumbnails"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    thumbnail_path: Mapped[str] = mapped_column(String(255), nullable=False)

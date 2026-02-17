from sqlalchemy import JSON, BigInteger
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base, TimestampMixin


class BallTrack(Base, TimestampMixin):
    """Model for storing ball track data."""
    
    __tablename__ = "ball_tracks"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column()
    ball_track: Mapped[dict] = mapped_column(JSON)

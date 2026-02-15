from sqlalchemy import JSON, BigInteger
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base, TimestampMixin


class Speed(Base, TimestampMixin):
    """Model for storing speed data."""
    
    __tablename__ = "speed"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    speed: Mapped[dict] = mapped_column(JSON, nullable=False)

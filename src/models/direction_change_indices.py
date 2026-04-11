from sqlalchemy import JSON, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base, TimestampMixin


class DirectionChangeIndices(Base, TimestampMixin):
    """Model for storing direction change indices."""
    
    __tablename__ = "direction_change_indices"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column()
    direction_change_indices: Mapped[dict] = mapped_column(JSON)
    owner_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

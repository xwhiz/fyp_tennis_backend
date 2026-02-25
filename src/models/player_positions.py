from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base, TimestampMixin


class PlayerPositions(Base, TimestampMixin):
    """Model for storing per-frame player bounding box positions."""

    __tablename__ = "player_positions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column()
    positions: Mapped[dict] = mapped_column(JSON)

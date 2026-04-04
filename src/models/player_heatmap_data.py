from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base, TimestampMixin


class PlayerHeatmapData(Base, TimestampMixin):
    """Model for storing court-space points used to generate player heatmaps."""

    __tablename__ = "player_heatmap_data"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(unique=True)
    top_points: Mapped[dict] = mapped_column(JSON)  # list of [x, y]
    bottom_points: Mapped[dict] = mapped_column(JSON)  # list of [x, y]

from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base, TimestampMixin


class HomographyMatrices(Base, TimestampMixin):
    """Per-frame homography (image -> court) for a task. Used to recompute heatmaps from player positions."""

    __tablename__ = "homography_matrices"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(unique=True)
    matrices: Mapped[dict] = mapped_column(JSON)  # list of 3x3 or null per frame

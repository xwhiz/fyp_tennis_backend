from sqlalchemy import JSON, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base, TimestampMixin


class RallyStats(Base, TimestampMixin):
    """Per-task rally list: one entry per valid scene (see build_rally_stats)."""

    __tablename__ = "rally_stats"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(unique=True)
    rallies: Mapped[list] = mapped_column(JSON)
    owner_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

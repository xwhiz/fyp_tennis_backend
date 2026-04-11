from sqlalchemy import BigInteger, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base, TimestampMixin


class ModelMetrics(Base, TimestampMixin):
    __tablename__ = "model_metrics"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    training_status: Mapped[str] = mapped_column(String, nullable=False, default="not_trained")
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision_: Mapped[float | None] = mapped_column("precision", Float, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_samples: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    owner_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

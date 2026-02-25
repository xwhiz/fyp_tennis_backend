from sqlalchemy import Float, String, BigInteger
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base, TimestampMixin


class ModelMetrics(Base, TimestampMixin):
    """Model for storing shot classifier training metrics."""

    __tablename__ = "model_metrics"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    training_status: Mapped[str] = mapped_column(String, default="not_trained", nullable=False)
    accuracy: Mapped[float] = mapped_column(Float, nullable=True)
    precision: Mapped[float] = mapped_column(Float, nullable=True)
    recall: Mapped[float] = mapped_column(Float, nullable=True)
    f1_score: Mapped[float] = mapped_column(Float, nullable=True)
    total_samples: Mapped[int] = mapped_column(BigInteger, nullable=True)




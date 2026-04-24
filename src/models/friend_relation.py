from sqlalchemy import BigInteger, Boolean, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base, TimestampMixin


class FriendRelation(Base, TimestampMixin):
    """Friend request: user_id requests friendship with friend_id."""

    __tablename__ = "friend_relations"
    __table_args__ = (UniqueConstraint("user_id", "friend_id", name="uq_friend_relations_user_friend"),)

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    friend_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    is_accepted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

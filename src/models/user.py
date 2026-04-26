from __future__ import annotations

from datetime import datetime
import uuid
from enum import Enum

import bcrypt
from sqlalchemy import Boolean, DateTime, Enum as SqlEnum, Float, String, func
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base


class UserRole(str, Enum):
    ADMIN = "admin"
    ANNOTATOR = "annotator"
    USER = "user"


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    first_name: Mapped[str] = mapped_column("firstName", String(255), nullable=False)
    last_name: Mapped[str] = mapped_column("lastName", String(255), nullable=False)
    player_height: Mapped[float | None] = mapped_column("playerHeight", Float, nullable=True)
    dominant_hand: Mapped[str] = mapped_column("dominantHand", String(16), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    at_tag: Mapped[str] = mapped_column("atTag", String(64), nullable=False, unique=True, index=True)
    profile_image_path: Mapped[str | None] = mapped_column("profileImagePath", String(255), nullable=True)
    password_hash: Mapped[str] = mapped_column("passwordHash", String(255), nullable=False)
    consent: Mapped[bool] = mapped_column(Boolean, nullable=False)
    role: Mapped[UserRole] = mapped_column(
        SqlEnum(UserRole, name="user_role", native_enum=False),
        default=UserRole.USER,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        "createdAt",
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        "updatedAt",
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    def set_password(self, password: str) -> None:
        self.password_hash = bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt(),
        ).decode("utf-8")

    def verify_password(self, password: str) -> bool:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            self.password_hash.encode("utf-8"),
        )

    def to_profile_dict(self) -> dict[str, str | float | None]:
        from src.utils.at_tag import display_at_tag
        from src.utils.profile_image import profile_image_url

        return {
            "firstName": self.first_name,
            "lastName": self.last_name,
            "playerHeight": self.player_height,
            "dominantHand": self.dominant_hand,
            "email": self.email,
            "atTag": display_at_tag(self.at_tag),
            "profileImageUrl": profile_image_url(self.profile_image_path),
        }

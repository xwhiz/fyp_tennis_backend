from sqlmodel import Session, select

from src.config import settings
from src.db.engine import Engine
from src.models.user import User, UserRole
from src.utils.at_tag import allocate_unique_at_tag


def seed_admin_user() -> None:
    if not settings.admin_email or not settings.admin_password:
        return

    normalized_email = settings.admin_email.strip().lower()

    with Session(Engine.instance()) as session:
        existing_admin = session.exec(
            select(User).where(User.email == normalized_email),
        ).first()
        if existing_admin:
            return

        admin_user = User(
            first_name=settings.admin_first_name,
            last_name=settings.admin_last_name,
            player_height=None,
            dominant_hand="right",
            email=normalized_email,
            consent=True,
            role=UserRole.ADMIN,
        )
        admin_user.set_password(settings.admin_password)
        admin_user.at_tag = allocate_unique_at_tag(session, admin_user.email)
        session.add(admin_user)
        session.commit()

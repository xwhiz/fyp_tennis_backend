"""Database session management."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.config import settings

# Create engine; SQLite does not support pool_size/max_overflow/pool_pre_ping
_is_sqlite = settings.database_url.strip().lower().startswith("sqlite")
if _is_sqlite:
    sqlite_memory = "memory" in settings.database_url
    engine = create_engine(
        settings.database_url,
        echo=False,
        connect_args={"check_same_thread": False} if sqlite_memory else {},
        poolclass=StaticPool if sqlite_memory else None,
    )
else:
    engine = create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        echo=False,
    )

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Get database session.

    Yields:
        Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

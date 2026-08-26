import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config.settings import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


engine = create_engine(
    url=settings.DATABASE_URL,
    pool_pre_ping=True,   # drops stale connections automatically
    pool_size=5,
    max_overflow=10,
    echo=False,           # set True to log every SQL statement
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


def get_db():
    """FastAPI dependency that provides a scoped DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables if they don't exist yet."""
    from app.database import models  # noqa: F401 — registers models with Base
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialised.")
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
    expire_on_commit=False,
)


from contextlib import contextmanager

def get_db():
    """FastAPI dependency that provides a scoped DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session():
    """Context manager for obtaining a transactional database session."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db():
    """Create all tables if they don't exist yet."""
    from app.database import models as app_models  # noqa: F401 — registers task/diary models
    from app.memory import models as memory_models  # noqa: F401 — registers memory models
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialised.")
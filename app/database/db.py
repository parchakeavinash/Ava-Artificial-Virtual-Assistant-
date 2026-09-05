import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config.settings import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


def _create_resilient_engine():
    """
    Creates SQLAlchemy engine with automatic fallback to SQLite
    if PostgreSQL is unreachable or connection fails.
    """
    url = settings.DATABASE_URL
    if url.startswith("postgresql"):
        try:
            eng = create_engine(
                url=url,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
                echo=False,
                connect_args={"connect_timeout": 3},
            )
            with eng.connect() as conn:
                pass
            logger.info("Connected successfully to PostgreSQL database.")
            return eng
        except Exception as e:
            logger.warning(
                f"PostgreSQL connection failed: {e}. "
                "Falling back to durable local SQLite database (sqlite:///ava_storage.db)."
            )

    fallback_url = "sqlite:///ava_storage.db"
    return create_engine(fallback_url, connect_args={"check_same_thread": False})


engine = _create_resilient_engine()

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


from sqlalchemy import text


def init_db():
    """Create all tables if they don't exist yet, and run safe schema migrations."""
    from app.database import models as app_models  # noqa: F401 — registers task/diary models
    from app.memory import models as memory_models  # noqa: F401 — registers memory models

    Base.metadata.create_all(bind=engine)

    # Safe schema migrations for PostgreSQL
    if engine.dialect.name == "postgresql":
        try:
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS user_id VARCHAR(128) DEFAULT 'default_user';"))
                conn.execute(text("ALTER TABLE diary_entries ADD COLUMN IF NOT EXISTS user_id VARCHAR(128) DEFAULT 'default_user';"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_tasks_user ON tasks(user_id);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_diary_user ON diary_entries(user_id);"))
                # Resync primary key sequence to max(id) to avoid duplicate key errors
                conn.execute(text("SELECT setval(pg_get_serial_sequence('tasks', 'id'), COALESCE(MAX(id), 1)) FROM tasks;"))
                conn.execute(text("SELECT setval(pg_get_serial_sequence('diary_entries', 'id'), COALESCE(MAX(id), 1)) FROM diary_entries;"))
                conn.commit()
        except Exception as e:
            logger.warning(f"Schema migration notice: {e}")
    elif engine.dialect.name == "sqlite":
        # SQLite column safety
        try:
            with engine.connect() as conn:
                for table in ["tasks", "diary_entries"]:
                    cols = [row[1] for row in conn.execute(text(f"PRAGMA table_info({table});")).fetchall()]
                    if "user_id" not in cols:
                        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN user_id VARCHAR(128) DEFAULT 'default_user';"))
                conn.commit()
        except Exception as e:
            logger.warning(f"SQLite migration notice: {e}")

    logger.info("Database tables initialised.")
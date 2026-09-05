import contextvars
from app.config.settings import settings

current_user_context: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_user_context", default=settings.MEMORY_USER_ID
)


def get_current_user() -> str:
    """Returns the active user_id for the current execution thread/context."""
    return current_user_context.get()


def set_current_user(user_id: str) -> None:
    """Sets the active user_id for the current execution thread/context."""
    current_user_context.set(user_id)

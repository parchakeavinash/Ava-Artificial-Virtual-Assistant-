"""
SQLAlchemy ORM models for Ava's persistent storage.

Tables
------
tasks         — Personal task manager (no priority / due_date per user request)
diary_entries — Personal diary / idea keeper (no tag per user request)
"""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database.db import Base


class Task(Base):
    """A single task in Ava's task manager."""

    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(Text, nullable=False, default="default_user", index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")   # pending | completed | cancelled
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, default=None)

    def __repr__(self) -> str:
        return f"<Task id={self.id} user_id={self.user_id!r} title={self.title!r} status={self.status}>"


class DiaryEntry(Base):
    """A personal diary / idea entry."""

    __tablename__ = "diary_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(Text, nullable=False, default="default_user", index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False, default="")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self) -> str:
        return f"<DiaryEntry id={self.id} user_id={self.user_id!r} title={self.title!r}>"

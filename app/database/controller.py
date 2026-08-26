"""
Database service layer for Tasks and Diary entries.

All functions receive a SQLAlchemy Session and return plain Python objects
(or raise HTTPException for invalid inputs) so they can be used from both
FastAPI route handlers AND the Ava voice-tool layer.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database.models import DiaryEntry, Task
from app.database.schemas import DiaryCreate, DiaryUpdate, TaskCreate, TaskUpdate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===========================================================================
# TASK SERVICES
# ===========================================================================

def create_task(body: TaskCreate, db: Session) -> Task:
    """Create and persist a new task."""
    try:
        task = Task(title=body.title.strip(), description=body.description.strip())
        db.add(task)
        db.commit()
        db.refresh(task)
        logger.info("Task created with ID %s.", task.id)
        return task
    except SQLAlchemyError as e:
        db.rollback()
        logger.error("DB error creating task: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create task.",
        )


def get_all_tasks(db: Session) -> list[Task]:
    """Return all tasks ordered newest first."""
    try:
        tasks = db.query(Task).order_by(Task.created_at.desc()).all()
        logger.info("Retrieved %d tasks.", len(tasks))
        return tasks
    except SQLAlchemyError as e:
        logger.error("DB error fetching tasks: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch tasks.",
        )


def get_pending_tasks(db: Session, limit: int = 10) -> list[Task]:
    """Return pending tasks only, newest first."""
    try:
        tasks = (
            db.query(Task)
            .filter(Task.status == "pending")
            .order_by(Task.created_at.desc())
            .limit(limit)
            .all()
        )
        logger.info("Retrieved %d pending tasks.", len(tasks))
        return tasks
    except SQLAlchemyError as e:
        logger.error("DB error fetching pending tasks: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch pending tasks.",
        )


def get_single_task(task_id: int, db: Session) -> Task:
    """Return a single task by ID or raise 404."""
    if task_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Task ID must be a positive integer.",
        )
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if task is None:
            logger.warning("Task %s not found.", task_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task with ID {task_id} was not found.",
            )
        return task
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.error("DB error fetching task %s: %s", task_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch task.",
        )


def update_task(task_id: int, update_data: TaskUpdate, db: Session) -> Task:
    """Update title, description, or status of an existing task."""
    task = get_single_task(task_id, db)
    try:
        for key, value in update_data.model_dump(exclude_unset=True).items():
            setattr(task, key, value)
        db.commit()
        db.refresh(task)
        logger.info("Task %s updated.", task_id)
        return task
    except SQLAlchemyError as e:
        db.rollback()
        logger.error("DB error updating task %s: %s", task_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update task.",
        )


def complete_task(task_id: int, db: Session) -> Task:
    """Mark a task as completed and stamp completed_at."""
    task = get_single_task(task_id, db)
    try:
        task.status = "completed"
        task.completed_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(task)
        logger.info("Task %s marked as completed.", task_id)
        return task
    except SQLAlchemyError as e:
        db.rollback()
        logger.error("DB error completing task %s: %s", task_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to complete task.",
        )


def delete_task(task_id: int, db: Session) -> None:
    """Permanently delete a task by ID."""
    task = get_single_task(task_id, db)
    try:
        db.delete(task)
        db.commit()
        logger.info("Task %s deleted.", task_id)
    except SQLAlchemyError as e:
        db.rollback()
        logger.error("DB error deleting task %s: %s", task_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete task.",
        )


# ===========================================================================
# DIARY SERVICES
# ===========================================================================

def create_diary_entry(body: DiaryCreate, db: Session) -> DiaryEntry:
    """Persist a new diary / idea entry."""
    try:
        entry = DiaryEntry(title=body.title.strip(), content=body.content.strip())
        db.add(entry)
        db.commit()
        db.refresh(entry)
        logger.info("Diary entry created with ID %s.", entry.id)
        return entry
    except SQLAlchemyError as e:
        db.rollback()
        logger.error("DB error creating diary entry: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save diary entry.",
        )


def get_recent_diary(db: Session, limit: int = 5) -> list[DiaryEntry]:
    """Return the most recent diary entries."""
    try:
        entries = (
            db.query(DiaryEntry)
            .order_by(DiaryEntry.created_at.desc())
            .limit(limit)
            .all()
        )
        logger.info("Retrieved %d diary entries.", len(entries))
        return entries
    except SQLAlchemyError as e:
        logger.error("DB error fetching diary entries: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch diary entries.",
        )


def search_diary(query: str, db: Session, limit: int = 10) -> list[DiaryEntry]:
    """Full-text search over title + content (case-insensitive ILIKE)."""
    try:
        pattern = f"%{query.strip()}%"
        entries = (
            db.query(DiaryEntry)
            .filter(
                DiaryEntry.title.ilike(pattern) | DiaryEntry.content.ilike(pattern)
            )
            .order_by(DiaryEntry.created_at.desc())
            .limit(limit)
            .all()
        )
        logger.info("Found %d diary entries for query %r.", len(entries), query)
        return entries
    except SQLAlchemyError as e:
        logger.error("DB error searching diary: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search diary.",
        )


def get_single_diary_entry(entry_id: int, db: Session) -> DiaryEntry:
    """Return a diary entry by ID or raise 404."""
    if entry_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Entry ID must be a positive integer.",
        )
    try:
        entry = db.query(DiaryEntry).filter(DiaryEntry.id == entry_id).first()
        if entry is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Diary entry with ID {entry_id} was not found.",
            )
        return entry
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.error("DB error fetching diary entry %s: %s", entry_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch diary entry.",
        )


def update_diary_entry(entry_id: int, update_data: DiaryUpdate, db: Session) -> DiaryEntry:
    """Update the title or content of a diary entry."""
    entry = get_single_diary_entry(entry_id, db)
    try:
        for key, value in update_data.model_dump(exclude_unset=True).items():
            setattr(entry, key, value)
        db.commit()
        db.refresh(entry)
        logger.info("Diary entry %s updated.", entry_id)
        return entry
    except SQLAlchemyError as e:
        db.rollback()
        logger.error("DB error updating diary entry %s: %s", entry_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update diary entry.",
        )


def delete_diary_entry(entry_id: int, db: Session) -> None:
    """Permanently delete a diary entry by ID."""
    entry = get_single_diary_entry(entry_id, db)
    try:
        db.delete(entry)
        db.commit()
        logger.info("Diary entry %s deleted.", entry_id)
    except SQLAlchemyError as e:
        db.rollback()
        logger.error("DB error deleting diary entry %s: %s", entry_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete diary entry.",
        )

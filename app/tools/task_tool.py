"""
Task Manager voice tool — wraps database controller into simple string-returning
functions that the LLM can call, scoped to the active user.
"""

from typing import Optional

from app.database.controller import (
    complete_task as _complete_task,
    create_task as _create_task,
    delete_task as _delete_task,
    get_pending_tasks as _get_pending_tasks,
    get_single_task as _get_single_task,
    update_task as _update_task,
)
from app.database.db import SessionLocal
from app.database.schemas import TaskCreate, TaskUpdate
from app.tools.user_context import get_current_user


def _fmt_task(t) -> str:
    completed = f" | Completed: {t.completed_at.strftime('%b %d, %Y %H:%M')}" if t.completed_at else ""
    return f"[#{t.id}] {t.title} — {t.status}{completed}"


def create_task(title: str, description: str = "", user_id: Optional[str] = None) -> str:
    """
    Create a new task with a title and optional description.
    Use when the user says 'create a task', 'add a to-do', or 'remind me to...'.
    """
    uid = user_id or get_current_user()
    db = SessionLocal()
    try:
        task = _create_task(TaskCreate(title=title, description=description, user_id=uid), db, user_id=uid)
        return f"Done! Task #{task.id} '{task.title}' has been created."
    except Exception as e:
        return f"Error creating task: {e}"
    finally:
        db.close()


def update_task(task_id: int, title: str = "", description: str = "", user_id: Optional[str] = None) -> str:
    """
    Update the title or description of an existing task by its numeric ID.
    Use when the user says 'update task 3' or 'change task 5 description'.
    """
    uid = user_id or get_current_user()
    db = SessionLocal()
    try:
        data = TaskUpdate(
            title=title or None,
            description=description or None,
        )
        task = _update_task(task_id, data, db, user_id=uid)
        return f"Task #{task.id} '{task.title}' updated successfully."
    except Exception as e:
        return f"Error updating task: {e}"
    finally:
        db.close()


def get_task(task_id: int, user_id: Optional[str] = None) -> str:
    """
    Get details of a specific task by its numeric ID.
    Use when the user says 'what is task 2' or 'show me task 4'.
    """
    uid = user_id or get_current_user()
    db = SessionLocal()
    try:
        task = _get_single_task(task_id, db, user_id=uid)
        desc = f"\n   Description: {task.description}" if task.description else ""
        created = task.created_at.strftime("%b %d, %Y %H:%M")
        return (
            f"Task #{task.id}: {task.title}\n"
            f"   Status: {task.status}\n"
            f"   Created: {created}{desc}"
        )
    except Exception as e:
        return f"Error fetching task: {e}"
    finally:
        db.close()


def list_pending_tasks(limit: int = 10, user_id: Optional[str] = None) -> str:
    """
    List all pending (not yet completed) tasks.
    Use when the user says 'what are my tasks', 'show pending tasks', or 'what do I have to do'.
    """
    uid = user_id or get_current_user()
    db = SessionLocal()
    try:
        tasks = _get_pending_tasks(db, user_id=uid, limit=limit)
        if not tasks:
            return "You have no pending tasks right now. All clear!"
        lines = "\n".join(_fmt_task(t) for t in tasks)
        return f"You have {len(tasks)} pending task(s):\n{lines}"
    except Exception as e:
        return f"Error fetching tasks: {e}"
    finally:
        db.close()


def complete_task(task_id: int, user_id: Optional[str] = None) -> str:
    """
    Mark a task as completed by its numeric ID.
    Use when the user says 'mark task 1 as done', 'complete task 3', or 'finish task 5'.
    """
    uid = user_id or get_current_user()
    db = SessionLocal()
    try:
        task = _complete_task(task_id, db, user_id=uid)
        return f"Task #{task.id} '{task.title}' has been marked as completed."
    except Exception as e:
        return f"Error completing task: {e}"
    finally:
        db.close()


def delete_task(task_id: int, user_id: Optional[str] = None) -> str:
    """
    Permanently delete a task by its numeric ID.
    ONLY call this after the user explicitly confirms deletion.
    """
    uid = user_id or get_current_user()
    db = SessionLocal()
    try:
        _delete_task(task_id, db, user_id=uid)
        return f"Task #{task_id} has been permanently deleted."
    except Exception as e:
        return f"Error deleting task: {e}"
    finally:
        db.close()

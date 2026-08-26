"""
Diary / Idea Keeper voice tool — wraps database controller into simple
string-returning functions the LLM can call.

The diary is a personal brain-dump: ideas, thoughts, reflections, learnings.
"""

from app.database.controller import (
    create_diary_entry as _create_diary_entry,
    delete_diary_entry as _delete_diary_entry,
    get_recent_diary as _get_recent_diary,
    search_diary as _search_diary,
)
from app.database.db import SessionLocal
from app.database.schemas import DiaryCreate


def _fmt_entry(e) -> str:
    date = e.created_at.strftime("%b %d, %Y")
    title = f" — {e.title}" if e.title else ""
    snippet = e.content[:120] + "..." if len(e.content) > 120 else e.content
    return f"[#{e.id}] {date}{title}: {snippet}"


def add_diary_entry(content: str, title: str = "") -> str:
    """
    Save a new idea, note, thought, or personal reflection to the diary.
    Use when the user says 'note this down', 'add an idea', 'diary entry', or 'save this thought'.
    """
    db = SessionLocal()
    try:
        entry = _create_diary_entry(DiaryCreate(title=title, content=content), db)
        label = f" titled '{entry.title}'" if entry.title else ""
        return f"Got it! Your entry{label} has been saved to your diary as #{entry.id}."
    except Exception as e:
        return f"Error saving diary entry: {e}"
    finally:
        db.close()


def read_recent_diary(limit: int = 5) -> str:
    """
    Read the most recent diary/idea entries.
    Use when the user says 'read my diary', 'show my recent ideas', or 'what did I write recently'.
    """
    db = SessionLocal()
    try:
        entries = _get_recent_diary(db, limit=limit)
        if not entries:
            return "Your diary is empty. Start adding your ideas and thoughts!"
        lines = "\n".join(_fmt_entry(e) for e in entries)
        return f"Here are your {len(entries)} most recent diary entries:\n{lines}"
    except Exception as e:
        return f"Error reading diary: {e}"
    finally:
        db.close()


def search_diary(query: str) -> str:
    """
    Search diary entries by keyword across titles and content.
    Use when the user says 'find my notes about X', 'search diary for Y', or 'what did I write about Z'.
    """
    db = SessionLocal()
    try:
        entries = _search_diary(query, db)
        if not entries:
            return f"No diary entries found matching '{query}'."
        lines = "\n".join(_fmt_entry(e) for e in entries)
        return f"Found {len(entries)} diary entry/entries for '{query}':\n{lines}"
    except Exception as e:
        return f"Error searching diary: {e}"
    finally:
        db.close()


def delete_diary_entry(entry_id: int) -> str:
    """
    Permanently delete a diary entry by its numeric ID.
    ONLY call this after the user explicitly confirms deletion.
    """
    db = SessionLocal()
    try:
        _delete_diary_entry(entry_id, db)
        return f"Diary entry #{entry_id} has been permanently deleted."
    except Exception as e:
        return f"Error deleting diary entry: {e}"
    finally:
        db.close()

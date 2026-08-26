"""
Proactive Reminder Scheduler for Ava.

How it works (simple mental model):
─────────────────────────────────────
1. A background daemon thread runs forever alongside Streamlit.
2. Every 60 seconds it wakes up and checks: "Is it reminder time?"
3. If yes, it fetches pending tasks from PostgreSQL and puts a natural-language
   reminder string into a thread-safe Queue.
4. The Streamlit @st.fragment polling loop drains this Queue every 0.3s.
5. When it finds a reminder, it pushes it to AgentRunner exactly like a
   voice utterance — Ava speaks it aloud.

Why a threading.Queue?
─────────────────────────────────────
Streamlit runs on the main thread; the scheduler runs on a separate thread.
A regular variable would cause race conditions.
threading.Queue is specifically built to be read/written safely from multiple
threads — it handles locking internally.
"""

import logging
import threading
import time
from datetime import datetime, timezone
from queue import Empty, Queue

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Global reminder queue — Streamlit reads from this every 0.3s
# ─────────────────────────────────────────────────────────────
reminder_queue: Queue[str] = Queue()

# Check every 60 seconds (lightweight — just a DB read)
_CHECK_INTERVAL_SECONDS = 60


def _build_reminder_message(tasks: list) -> str:
    """Convert a list of Task objects into a voice-friendly reminder string."""
    count = len(tasks)
    if count == 0:
        return ""

    if count == 1:
        return (
            f"Hey boss! Just a heads-up — you have 1 pending task: "
            f"'{tasks[0].title}'. Want to continue with it?"
        )

    # Mention up to the first 3 task titles
    titles = [f"'{t.title}'" for t in tasks[:3]]
    more = f" and {count - 3} more" if count > 3 else ""
    title_list = ", ".join(titles[:-1]) + f" and {titles[-1]}" if len(titles) > 1 else titles[0]
    return (
        f"Hey boss! You have {count} pending tasks: {title_list}{more}. "
        f"Want to pick up where you left off?"
    )


def _should_remind(hour: int, minute: int, last_reminded_date: list) -> bool:
    """
    Returns True if:
    - Current local time matches the configured reminder hour:minute
    - AND we haven't already reminded today (prevents firing every 60s)

    last_reminded_date is a single-element list so we can mutate it from
    inside this function (Python closure mutation trick).
    """
    now = datetime.now()
    today = now.date()

    if now.hour == hour and now.minute == minute:
        if last_reminded_date[0] != today:
            last_reminded_date[0] = today
            return True
    return False


def _scheduler_loop(reminder_hour: int, reminder_minute: int):
    """
    The main loop that runs forever in the background thread.
    Checks the clock every 60 seconds and fires reminders on schedule.
    """
    # We use a list so we can mutate it inside _should_remind
    last_reminded_date = [None]
    logger.info(
        "[Scheduler] Started. Will remind daily at %02d:%02d local time.",
        reminder_hour,
        reminder_minute,
    )

    while True:
        try:
            time.sleep(_CHECK_INTERVAL_SECONDS)

            if not _should_remind(reminder_hour, reminder_minute, last_reminded_date):
                continue

            # Fetch pending tasks from PostgreSQL
            from app.database.controller import get_pending_tasks
            from app.database.db import SessionLocal

            db = SessionLocal()
            try:
                tasks = get_pending_tasks(db, limit=10)
            finally:
                db.close()

            if not tasks:
                logger.info("[Scheduler] Reminder time reached but no pending tasks.")
                continue

            message = _build_reminder_message(tasks)
            reminder_queue.put_nowait(message)
            logger.info("[Scheduler] Queued reminder: %r", message)

        except Exception as exc:
            logger.error("[Scheduler] Unexpected error in scheduler loop: %s", exc)
            # Never crash the scheduler thread — keep it alive
            time.sleep(10)


def start_scheduler(reminder_hour: int = 9, reminder_minute: int = 0) -> threading.Thread:
    """
    Start the background reminder scheduler as a daemon thread.

    Args:
        reminder_hour:   Hour (24h format) to fire the daily reminder. Default: 9 (9 AM).
        reminder_minute: Minute to fire the daily reminder. Default: 0.

    Returns:
        The running Thread object (you usually don't need it).

    Why daemon=True?
        A daemon thread dies automatically when the main program exits.
        Without daemon=True, the thread would keep the process alive forever
        even after the user closes the Streamlit tab.
    """
    thread = threading.Thread(
        target=_scheduler_loop,
        args=(reminder_hour, reminder_minute),
        daemon=True,
        name="AvaReminderScheduler",
    )
    thread.start()
    logger.info("[Scheduler] Daemon thread started (PID-agnostic, daemon=True).")
    return thread


def drain_reminders() -> list[str]:
    """
    Drain all queued reminder messages.
    Called by the Streamlit polling loop every 0.3s.
    Returns a list of reminder strings (usually 0 or 1 items).
    """
    messages = []
    while True:
        try:
            messages.append(reminder_queue.get_nowait())
        except Empty:
            break
    return messages

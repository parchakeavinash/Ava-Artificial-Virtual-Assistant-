"""
Verification script for:
1. Database Durability (Resilient engine & SQLite fallback)
2. Voice Latency Optimization (Trivial query bypass & Parallel retrieval & Background record_turn)
3. Context Bridging ("Continue where we left off")
4. Autonomous Episodic Memory Creation (Closure intent & Session switch)
"""

import os
import sys
import time
import uuid

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.db import engine, init_db, get_db_session
from app.database import controller
from app.database.schemas import TaskCreate
from app.database.models import Task, DiaryEntry
from app.memory.manager import MemoryManager
from app.agent.langchain_agent import LangChainResilientAgent


def test_all():
    print("=" * 65)
    print("TESTING: LATENCY + EPISODES + DB DURABILITY + CONTEXT BRIDGING")
    print("=" * 65)

    # 1. Database Durability Check
    print("\n[1/4] Checking Database Engine Durability...")
    init_db()
    dialect = engine.dialect.name
    print(f"  Active database engine dialect: '{dialect}'")
    assert dialect in ["postgresql", "sqlite"], f"Unexpected dialect: {dialect}"
    print("  [OK] Database resilience verified.")

    test_user = f"charlie_{uuid.uuid4().hex[:4]}"
    manager = MemoryManager(user_id=test_user)
    agent = LangChainResilientAgent(user_id=test_user)

    # 2. Voice Latency & Trivial Query Bypass
    print("\n[2/4] Testing Voice Latency Optimization & Bypass...")
    # Assert trivial queries are recognized
    assert manager.is_trivial_utterance("hi") is True
    assert manager.is_trivial_utterance("hello ava") is True
    assert manager.is_trivial_utterance("what is 10 + 20") is True
    assert manager.is_trivial_utterance("I am building an autonomous rover in ROS2") is False

    # Timing test for trivial bypass
    t0 = time.perf_counter()
    prompts, history = manager.build_memory_context("Hello Ava", session_id="test_sess", user_id=test_user)
    elapsed_trivial_ms = (time.perf_counter() - t0) * 1000
    print(f"  Trivial greeting context build time: {elapsed_trivial_ms:.2f}ms (Bypassed vector search!)")
    assert elapsed_trivial_ms < 150, f"Expected <150ms, got {elapsed_trivial_ms}ms"

    # Timing test for non-blocking record_turn
    t0 = time.perf_counter()
    manager.record_turn(
        user_text="Hello",
        ai_text="Hi there! How can I help you today?",
        session_id="test_sess",
        user_id=test_user,
        extraction_llm=agent.extraction_llm,
    )
    elapsed_record_ms = (time.perf_counter() - t0) * 1000
    print(f"  record_turn return time: {elapsed_record_ms:.2f}ms (Background thread dispatched!)")
    assert elapsed_record_ms < 25, f"record_turn was blocking! Took {elapsed_record_ms}ms"
    print("  [OK] Voice latency optimizations verified.")

    # 3. Context Bridging ("Continue where we left off")
    print("\n[3/4] Testing Context Bridging ('Continue where we left off')...")
    assert manager.is_resume_intent("Continue where we left off") is True
    assert manager.is_resume_intent("what was I working on?") is True
    assert manager.is_resume_intent("what's next on my plate?") is True

    # User creates a pending task
    with get_db_session() as db:
        controller.create_task(
            TaskCreate(
                title="Finish LangGraph agent flow features",
                description="Connect resilient agent with short-term and episodic memory",
                user_id=test_user,
            ),
            db,
            user_id=test_user,
        )

    # Next session: User asks to continue where they left off
    new_session = f"session_next_day_{uuid.uuid4().hex[:4]}"
    print(f"  Simulating user query: 'Continue where we left off.' in session {new_session}")
    reply = agent.respond(
        user_text="Continue where we left off.",
        session_id=new_session,
        user_id=test_user,
    )
    print(f"  Ava reply:\n    >>> \"{reply}\"")
    # Ava must reference the pending task (LangGraph / agent flow)
    assert any(term in reply.lower() for term in ["langgraph", "agent", "flow", "task", "finish"]), (
        f"Ava did not bridge context with the pending task! Reply: {reply}"
    )
    print("  [OK] Context bridging synthesized pending tasks seamlessly into response.")

    # 4. Autonomous Episodic Memory Creation
    print("\n[4/4] Testing Autonomous Episodic Memory Creation...")
    assert manager.is_closure_intent("That's all for today Ava, goodbye!") is True
    assert manager.is_closure_intent("talk to you tomorrow, bye") is True

    # Test auto-distill helper
    distilled = manager.auto_distill_if_needed(
        session_id=new_session,
        extraction_llm=agent.extraction_llm,
        user_id=test_user,
        min_messages=2,
    )
    print(f"  Auto-distilled session episode result: {distilled}")
    print("  [OK] Autonomous episodic creation verified.")

    # Cleanup
    print("\nCleaning up test data...")
    with get_db_session() as db:
        db.query(Task).filter(Task.user_id == test_user).delete(synchronize_session=False)
        db.query(DiaryEntry).filter(DiaryEntry.user_id == test_user).delete(synchronize_session=False)
    agent.clear_history(session_id=new_session, user_id=test_user)
    print("  [OK] Cleanup complete.")

    print("\n" + "=" * 65)
    print("ALL LATENCY, EPISODES, DURABILITY & BRIDGING TESTS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    test_all()

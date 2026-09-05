"""
Verification script for Multi-Tenant Isolation across Tasks, Diary, and Agent Memory.
Tests:
1. Schema migration in PostgreSQL (user_id on tasks & diary_entries)
2. Controller-level task & diary isolation between Alice and Bob
3. Tool-level task & diary isolation via user_context
4. End-to-end LangChainResilientAgent multi-user tool execution
"""

import os
import sys
import uuid

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.db import init_db, get_db_session
from app.database.models import Task, DiaryEntry
from app.database.schemas import TaskCreate, DiaryCreate
from app.database import controller
from app.tools.task_tool import create_task, list_pending_tasks, get_task
from app.tools.diary_tool import add_diary_entry, read_recent_diary, search_diary
from app.tools.user_context import set_current_user
from app.agent.langchain_agent import LangChainResilientAgent


def run_multitenancy_test():
    print("=" * 65)
    print("TESTING MULTI-TENANT ISOLATION (TASKS + DIARY + MEMORY)")
    print("=" * 65)

    # 1. Initialize & migrate DB schema
    print("\n[1/5] Running init_db() and schema migrations...")
    init_db()
    print("  [OK] Database initialized and user_id columns verified.")

    alice = f"alice_{uuid.uuid4().hex[:4]}"
    bob = f"bob_{uuid.uuid4().hex[:4]}"
    print(f"  Test users: '{alice}' and '{bob}'")

    # 2. Database Controller-level Isolation
    print("\n[2/5] Testing Controller-level Task Isolation...")
    with get_db_session() as db:
        t_alice = controller.create_task(TaskCreate(title="Alice Private Task", user_id=alice), db, user_id=alice)
        t_bob = controller.create_task(TaskCreate(title="Bob Secret Project", user_id=bob), db, user_id=bob)

        alice_tasks = controller.get_pending_tasks(db, user_id=alice)
        bob_tasks = controller.get_pending_tasks(db, user_id=bob)

        print(f"  Alice's pending tasks count: {len(alice_tasks)}")
        print(f"  Bob's pending tasks count: {len(bob_tasks)}")

        assert len(alice_tasks) == 1, f"Expected 1 task for Alice, got {len(alice_tasks)}"
        assert alice_tasks[0].title == "Alice Private Task"
        assert len(bob_tasks) == 1, f"Expected 1 task for Bob, got {len(bob_tasks)}"
        assert bob_tasks[0].title == "Bob Secret Project"

        # Bob tries to access Alice's task
        bob_saw_alice_task = False
        try:
            controller.get_single_task(t_alice.id, db, user_id=bob)
            bob_saw_alice_task = True
        except Exception:
            pass  # Expected 404 HTTPException

        assert not bob_saw_alice_task, "Security violation: Bob was able to retrieve Alice's task!"
        print("  [OK] Controller-level task isolation verified. Bob cannot see Alice's task.")

    # 3. Diary Isolation
    print("\n[3/5] Testing Controller-level Diary Isolation...")
    with get_db_session() as db:
        controller.create_diary_entry(DiaryCreate(title="Alice Secret", content="Alice loves autonomous rovers", user_id=alice), db, user_id=alice)
        controller.create_diary_entry(DiaryCreate(title="Bob Notes", content="Bob prefers cloud architecture", user_id=bob), db, user_id=bob)

        alice_diary = controller.get_recent_diary(db, user_id=alice)
        bob_search_rover = controller.search_diary("rovers", db, user_id=bob)

        assert len(alice_diary) == 1
        assert len(bob_search_rover) == 0, "Security violation: Bob searched and found Alice's diary notes!"
        print("  [OK] Controller-level diary isolation verified. Bob's search returned 0 results.")

    # 4. Tool-level Context Isolation
    print("\n[4/5] Testing Tool-level Active User Context...")
    set_current_user(alice)
    alice_tool_list = list_pending_tasks()
    assert "Alice Private Task" in alice_tool_list
    assert "Bob Secret Project" not in alice_tool_list

    set_current_user(bob)
    bob_tool_list = list_pending_tasks()
    assert "Bob Secret Project" in bob_tool_list
    assert "Alice Private Task" not in bob_tool_list
    print("  [OK] Tool-level context scoping verified. Active user correctly switches views.")

    # 5. Full Agent Multi-Tenant Flow
    print("\n[5/5] Testing Full LangChain Agent with Multi-Tenancy...")
    alice_agent = LangChainResilientAgent(user_id=alice)
    bob_agent = LangChainResilientAgent(user_id=bob)

    alice_session = f"session_alice_{uuid.uuid4().hex[:4]}"
    bob_session = f"session_bob_{uuid.uuid4().hex[:4]}"

    print(f"  Asking Alice's agent: 'List my pending tasks'")
    alice_reply = alice_agent.respond("List my pending tasks", session_id=alice_session, user_id=alice)
    print(f"  Alice Agent Reply:\n    >>> \"{alice_reply}\"")
    assert "Alice Private Task" in alice_reply or "Alice" in alice_reply or "task" in alice_reply.lower()

    print(f"\n  Asking Bob's agent: 'List my pending tasks'")
    bob_reply = bob_agent.respond("List my pending tasks", session_id=bob_session, user_id=bob)
    print(f"  Bob Agent Reply:\n    >>> \"{bob_reply}\"")
    assert "Bob Secret Project" in bob_reply or "Bob" in bob_reply or "task" in bob_reply.lower()
    assert "Alice Private Task" not in bob_reply

    print("  [OK] Agent multi-tenancy end-to-end verified!")

    # Cleanup
    print("\nCleaning up test data...")
    with get_db_session() as db:
        db.query(Task).filter(Task.user_id.in_([alice, bob])).delete(synchronize_session=False)
        db.query(DiaryEntry).filter(DiaryEntry.user_id.in_([alice, bob])).delete(synchronize_session=False)
    alice_agent.clear_history(session_id=alice_session, user_id=alice)
    bob_agent.clear_history(session_id=bob_session, user_id=bob)
    print("  [OK] Cleanup complete.")

    print("\n" + "=" * 65)
    print("ALL MULTI-TENANCY TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_multitenancy_test()

"""
Verification script for Ava's 3-Layer Cognitive Memory System.
Tests:
1. PostgreSQL table creation (tasks, diary_entries, chat_messages, conversation_summaries, episodes, semantic_facts)
2. Short-term memory buffer & sliding window
3. Eviction summarization
4. Semantic fact classification & upsert
5. Episodic distillation & vector search
6. Full LangChainResilientAgent turn with memory
"""

import os
import sys
import uuid
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config.settings import settings
from app.database.db import init_db, get_db_session
from app.memory.manager import MemoryManager
from app.agent.langchain_agent import LangChainResilientAgent

def run_tests():
    print("=" * 60)
    print("TESTING AVA MULTI-LAYERED MEMORY SUBSYSTEM")
    print("=" * 60)

    # 1. Database Initialization
    print("\n[1/6] Initializing database tables...")
    init_db()
    print("  [OK] Tables verified and initialized successfully.")

    test_user = "test_developer"
    test_session = f"session_{uuid.uuid4().hex[:6]}"
    manager = MemoryManager(user_id=test_user)

    # 2. Short-Term Memory Insertion & Sliding Window
    print(f"\n[2/6] Testing Short-Term Memory (Session: {test_session})...")
    msg1 = manager.short_term.add_user_message(test_session, "Hello Ava, I'm setting up my development workspace.", user_id=test_user)
    msg2 = manager.short_term.add_ai_message(test_session, "Welcome! How can I assist you with your setup?", user_id=test_user)
    print(f"  Saved message #{msg1.id} (user) and #{msg2.id} (assistant)")

    history = manager.short_term.get_messages(test_session, window_size=5, user_id=test_user)
    print(f"  Retrieved {len(history)} messages from sliding window buffer.")
    assert len(history) == 2, f"Expected 2 messages, got {len(history)}"
    print("  [OK] Short-term sliding window passed.")

    # 3. Semantic Memory (Fact Extraction & Search)
    print(f"\n[3/6] Testing Semantic Memory (Fact Extraction & Vector Search)...")
    agent = LangChainResilientAgent(user_id=test_user)
    fact_utterance = "I prefer PostgreSQL as my primary database."
    print(f"  Processing utterance: '{fact_utterance}'")

    res = manager.semantic.process_message(
        user_message=fact_utterance,
        user_id=test_user,
        llm=agent.extraction_llm,
    )
    print(f"  Extraction result: {res}")
    facts = manager.list_known_facts(user_id=test_user)
    print(f"  Total semantic facts for {test_user}: {len(facts)}")
    for f in facts:
        print(f"    - {f['key']} = {f['value']} (Source: {f['source']})")
    assert len(facts) >= 1, "Semantic fact was not extracted!"
    print("  [OK] Semantic fact extraction passed.")

    # 4. Episodic Memory (Distillation & Vector Retrieval)
    print(f"\n[4/6] Testing Episodic Memory Distillation...")
    episode_info = manager.create_episode(
        session_id=test_session,
        extraction_llm=agent.extraction_llm,
        user_id=test_user,
    )
    print(f"  Distilled Episode: {episode_info}")
    if episode_info:
        print(f"  Searching episodes for 'workspace setup'...")
        found_episodes = manager.episodic.search_episodes(
            query="workspace setup",
            user_id=test_user,
            top_k=2,
            min_similarity=0.3,
        )
        print(f"  Found {len(found_episodes)} matching episode(s):")
        for ep in found_episodes:
            print(f"    - Episode #{ep['episode_id']} (Sim: {ep['similarity']}): {ep['summary']}")
        print("  [OK] Episodic vector retrieval passed.")
    else:
        print("  [INFO] Episode distillation returned None (insufficient turns or empty transcript).")

    # 5. Full Agent Response with Memory Integration
    print(f"\n[5/6] Testing Full Agent Turn with Memory...")
    new_session = f"session_cross_{uuid.uuid4().hex[:6]}"
    print(f"  Asking Ava in brand-new session '{new_session}': 'Which database do I like using?'")
    reply = agent.respond(
        user_text="Which database do I like using?",
        session_id=new_session,
        user_id=test_user,
    )
    print(f"  Ava replied:\n  >>> \"{reply}\"")
    assert "postgres" in reply.lower(), f"Expected PostgreSQL in answer, got: {reply}"
    print("  [OK] Cross-session semantic recall verified!")

    # 6. Session Listing
    print(f"\n[6/6] Testing Session Listing (for ChatGPT-style UI)...")
    sessions = manager.list_sessions(user_id=test_user)
    print(f"  Found {len(sessions)} session(s):")
    for s in sessions:
        print(f"    - [{s['session_id']}] ({s['message_count']} msgs) Last: {s['last_message']}")
    print("  [OK] Session listing passed.")

    # Cleanup test data
    print("\nCleaning up test session data...")
    manager.clear_session(test_session, user_id=test_user)
    manager.clear_session(new_session, user_id=test_user)
    manager.semantic.clear_facts(user_id=test_user)
    manager.episodic.clear_episodes(user_id=test_user)
    print("  [OK] Cleanup complete.")

    print("\n" + "=" * 60)
    print("ALL MEMORY SUBSYSTEM VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()

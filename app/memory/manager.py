import logging
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.messages import BaseMessage, SystemMessage

from app.config.settings import settings
from app.database.db import init_db
from app.memory.embeddings import GeminiEmbeddingProvider
from app.memory.episodic import EpisodicMemoryManager
from app.memory.semantic import SemanticMemoryManager
from app.memory.short_term import ShortTermMemory

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Unified coordinator for Ava's 3-layer cognitive memory architecture:
    1. Short-Term Memory Buffer (Sliding Window + Running Eviction Summary)
    2. Episodic Memory (Experience distillation + Vector search via Gemini)
    3. Semantic Memory (Persistent user facts & preferences across sessions)
    """

    def __init__(self, user_id: Optional[str] = None):
        # Ensure database tables exist
        try:
            init_db()
        except Exception as e:
            logger.warning(f"Database initialization warning: {e}")

        self.user_id = user_id or settings.MEMORY_USER_ID
        self.embedding_provider = GeminiEmbeddingProvider()

        self.short_term = ShortTermMemory(default_window_size=settings.MEMORY_WINDOW)
        self.episodic = EpisodicMemoryManager(embedding_provider=self.embedding_provider)
        self.semantic = SemanticMemoryManager(embedding_provider=self.embedding_provider)

    def build_memory_context(
        self,
        user_text: str,
        session_id: str,
        user_id: Optional[str] = None,
    ) -> Tuple[List[SystemMessage], List[BaseMessage]]:
        """
        Retrieves relevant memory layers and constructs system prompts + history messages.
        Returns (injected_system_messages, sliding_window_history).
        """
        uid = user_id or self.user_id
        injected_prompts: List[SystemMessage] = []

        # 1. Semantic Facts (What Ava knows about the user)
        try:
            facts = self.semantic.search_facts(
                query=user_text,
                user_id=uid,
                top_k=settings.SEMANTIC_TOP_K,
                min_similarity=settings.SEMANTIC_MIN_SIMILARITY,
            )
            if facts:
                fact_bullets = "\n".join(
                    f"- {f['key'].replace('_', ' ').title()}: {f['value']}"
                    for f in facts
                )
                injected_prompts.append(
                    SystemMessage(content=f"Context from memory about the user:\n{fact_bullets}")
                )
        except Exception as e:
            logger.warning(f"Semantic facts retrieval skipped: {e}")

        # 2. Running Conversation Summary (Evicted earlier turns in this session)
        try:
            summary = self.short_term.get_summary(session_id=session_id, user_id=uid)
            if summary:
                injected_prompts.append(
                    SystemMessage(content=f"Summary of earlier discussion in this conversation:\n{summary}")
                )
        except Exception as e:
            logger.warning(f"Summary retrieval skipped: {e}")

        # 3. Episodic Memory (Cross-session past experiences/topics matching the query)
        try:
            episodes = self.episodic.search_episodes(
                query=user_text,
                user_id=uid,
                session_id=None,  # cross-session
                top_k=settings.EPISODIC_TOP_K,
                min_similarity=settings.EPISODIC_MIN_SIMILARITY,
            )
            if episodes:
                ep_blocks = []
                for ep in episodes:
                    date_str = ep["timestamp"][:10] if ep.get("timestamp") else "past"
                    events = "\n".join(f"  • {ev}" for ev in ep.get("events", []))
                    ep_blocks.append(f"Past Episode ({date_str}):\n{ep['summary']}\nEvents:\n{events}")
                injected_prompts.append(
                    SystemMessage(
                        content="Relevant memory from past conversations (use only if relevant):\n"
                        + "\n\n".join(ep_blocks)
                    )
                )
        except Exception as e:
            logger.warning(f"Episodic retrieval skipped: {e}")

        # 4. Sliding Window History
        try:
            history = self.short_term.get_messages(
                session_id=session_id,
                window_size=settings.MEMORY_WINDOW,
                user_id=uid,
            )
        except Exception as e:
            logger.warning(f"History retrieval skipped: {e}")
            history = []

        return injected_prompts, history

    def record_turn(
        self,
        user_text: str,
        ai_text: str,
        session_id: str,
        user_id: Optional[str] = None,
        extraction_llm: Optional[Any] = None,
    ) -> None:
        """
        Saves user and assistant messages to DB, and performs background
        semantic fact extraction & incremental running summary updates.
        """
        uid = user_id or self.user_id

        # 1. Save messages to short-term DB
        try:
            self.short_term.add_user_message(session_id=session_id, content=user_text, user_id=uid)
            self.short_term.add_ai_message(session_id=session_id, content=ai_text, user_id=uid)
        except Exception as e:
            logger.error(f"Failed to record turn messages in DB: {e}")

        # 2. Semantic Fact Extraction (Non-blocking / guarded)
        if extraction_llm:
            try:
                res = self.semantic.process_message(
                    user_message=user_text,
                    user_id=uid,
                    llm=extraction_llm,
                )
                if res:
                    action, key, val = res
                    logger.info(f"Learned user fact [{action}]: {key} = {val}")
            except Exception as e:
                logger.warning(f"Semantic fact processing failed: {e}")

            # 3. Incremental Summary of Evicted Turns
            try:
                self.short_term.update_summary_if_needed(
                    session_id=session_id,
                    llm=extraction_llm,
                    window_size=settings.MEMORY_WINDOW,
                    user_id=uid,
                )
            except Exception as e:
                logger.warning(f"Summary update failed: {e}")

    def create_episode(
        self,
        session_id: str,
        extraction_llm: Any,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Distill the active conversation session into a persistent episodic memory."""
        uid = user_id or self.user_id
        messages = self.short_term.get_raw_messages(session_id=session_id, user_id=uid, limit=50)
        if not messages:
            return None

        extracted = self.episodic.extract_episode_from_conversation(
            messages=messages,
            llm=extraction_llm,
            session_id=session_id,
            user_id=uid,
        )
        if not extracted:
            return None

        episode = self.episodic.store_episode(
            user_id=uid,
            session_id=session_id,
            summary=extracted.get("summary", ""),
            events=extracted.get("events", []),
            topics=extracted.get("topics", []),
            start_message_id=extracted.get("start_message_id"),
            end_message_id=extracted.get("end_message_id"),
            timestamp=extracted.get("timestamp"),
        )
        return {
            "episode_id": episode.episode_id,
            "summary": episode.summary,
            "events": episode.events,
            "topics": episode.topics,
        }

    def list_known_facts(self, user_id: Optional[str] = None) -> List[dict]:
        """Returns list of semantic facts known about the user."""
        uid = user_id or self.user_id
        facts = self.semantic.list_facts(user_id=uid)
        return [
            {
                "key": f.key,
                "value": f.value,
                "source": f.source,
                "updated_at": f.updated_at.strftime("%Y-%m-%d %H:%M") if f.updated_at else "",
            }
            for f in facts
        ]

    def list_sessions(self, user_id: Optional[str] = None) -> List[dict]:
        """Returns list of conversation sessions for this user."""
        uid = user_id or self.user_id
        return self.short_term.list_sessions(user_id=uid)

    def clear_session(self, session_id: str, user_id: Optional[str] = None) -> int:
        """Clears messages and summary for a session."""
        uid = user_id or self.user_id
        return self.short_term.clear_memory(session_id=session_id, user_id=uid)

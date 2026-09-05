import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.messages import BaseMessage, SystemMessage

from app.config.settings import settings
from app.database.db import get_db_session, init_db
from app.database.controller import get_pending_tasks
from app.memory.embeddings import GeminiEmbeddingProvider
from app.memory.episodic import EpisodicMemoryManager
from app.memory.semantic import SemanticMemoryManager
from app.memory.short_term import ShortTermMemory

logger = logging.getLogger(__name__)

# Patterns for skipping vector searches on trivial turns (Voice Latency Optimization)
GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|yo|greetings|good\s+(morning|afternoon|evening)|thanks|thank\s+you|ok|okay|bye|goodbye|cool|yes|no|clear|test)\b",
    re.IGNORECASE,
)
MATH_PATTERNS = re.compile(
    r"\b(calculate|math)\b|[\+\-\*\/\%]|(\d+\s*[\+\-\*\/]\s*\d+)",
    re.IGNORECASE,
)

# Patterns for Context Bridging ("Continue where we left off")
RESUME_PATTERNS = re.compile(
    r"\b(continue\s+(where\s+we\s+left\s+off|our\s+work)|what\s+(was\s+i|were\s+we)\s+(doing|working\s+on)|"
    r"what(\'s|\s+is)\s+next(\s+on\s+my\s+plate)?|pick\s+up\s+where\s+we\s+left|catch\s+me\s+up|"
    r"what\s+did\s+we\s+do\s+last|where\s+did\s+we\s+leave\s+off|where\s+were\s+we)\b",
    re.IGNORECASE,
)

# Patterns for Conversation Closure (Autonomous Episodic Distillation)
CLOSURE_PATTERNS = re.compile(
    r"\b(bye|goodbye|that\'s\s+all(\s+for\s+today|\s+for\s+now)?|see\s+you(\s+later|\s+tomorrow)?|"
    r"talk\s+to\s+you\s+later|done\s+for\s+the\s+day|we\s+are\s+done)\b",
    re.IGNORECASE,
)


class MemoryManager:
    """
    Unified coordinator for Ava's 3-layer cognitive memory architecture:
    1. Short-Term Memory Buffer (Sliding Window + Running Eviction Summary)
    2. Episodic Memory (Experience distillation + Vector search via Gemini)
    3. Semantic Memory (Persistent user facts & preferences across sessions)
    
    Includes:
    - Low-latency parallel retrieval & trivial query bypass
    - Context bridging for 'continue where we left off'
    - Autonomous episodic creation on session switches and goodbye intents
    - Background non-blocking post-turn distillation
    """

    def __init__(self, user_id: Optional[str] = None):
        try:
            init_db()
        except Exception as e:
            logger.warning(f"Database initialization warning: {e}")

        self.user_id = user_id or settings.MEMORY_USER_ID
        self.embedding_provider = GeminiEmbeddingProvider()

        self.short_term = ShortTermMemory(default_window_size=settings.MEMORY_WINDOW)
        self.episodic = EpisodicMemoryManager(embedding_provider=self.embedding_provider)
        self.semantic = SemanticMemoryManager(embedding_provider=self.embedding_provider)
        self._executor = ThreadPoolExecutor(max_workers=4)

    def is_trivial_utterance(self, text: str) -> bool:
        """Determines if the utterance is a simple greeting/math command where vector search is skipped."""
        stripped = text.strip()
        words = stripped.split()
        if len(words) <= 4 and GREETING_PATTERNS.search(stripped):
            return True
        if len(words) <= 10 and MATH_PATTERNS.search(stripped):
            return True
        return False

    def is_resume_intent(self, text: str) -> bool:
        """Checks if the user is asking to resume where they left off."""
        return bool(RESUME_PATTERNS.search(text.strip()))

    def is_closure_intent(self, text: str) -> bool:
        """Checks if the user is saying goodbye or closing the conversation."""
        return bool(CLOSURE_PATTERNS.search(text.strip()))

    def build_context_bridge(self, user_id: Optional[str] = None) -> Optional[SystemMessage]:
        """
        Synthesizes active pending tasks + most recent episodic memory into a
        prioritized continuity prompt when the user asks 'continue where we left off'.
        """
        uid = user_id or self.user_id
        items = []

        # 1. Fetch pending tasks for this user
        try:
            with get_db_session() as db:
                pending_tasks = get_pending_tasks(db, user_id=uid, limit=3)
                if pending_tasks:
                    task_lines = "\n".join(f"  • [Task #{t.id}] {t.title}" for t in pending_tasks)
                    items.append(f"Active Pending Tasks for User:\n{task_lines}")
        except Exception as e:
            logger.warning(f"Context bridge task retrieval warning: {e}")

        # 2. Fetch the most recent episodic memory
        try:
            recent_episodes = self.episodic.list_episodes(user_id=uid, limit=1)
            if recent_episodes:
                ep = recent_episodes[0]
                date_str = ep.timestamp.strftime("%b %d, %Y") if ep.timestamp else "recently"
                events = ", ".join(ep.events[:3]) if ep.events else ep.summary
                items.append(f"Last Major Work Session ({date_str}):\n  • Summary: {ep.summary}\n  • Key Events: {events}")
        except Exception as e:
            logger.warning(f"Context bridge episode retrieval warning: {e}")

        if not items:
            return None

        content = (
            "CONTINUITY CONTEXT (User asked to continue where they left off):\n"
            + "\n\n".join(items)
            + "\n\nProactively synthesize this context to guide the user back into their work naturally and conversationally!"
        )
        return SystemMessage(content=content)

    def build_memory_context(
        self,
        user_text: str,
        session_id: str,
        user_id: Optional[str] = None,
    ) -> Tuple[List[SystemMessage], List[BaseMessage]]:
        """
        Retrieves relevant memory layers and constructs system prompts + history messages.
        Features:
        - Trivial query bypass for minimal latency (<10ms)
        - Parallel semantic and episodic retrieval via ThreadPoolExecutor (~180ms)
        - Context bridging for 'continue where we left off'
        """
        uid = user_id or self.user_id
        injected_prompts: List[SystemMessage] = []

        # Check for Context Bridging ("Continue where we left off")
        if self.is_resume_intent(user_text):
            bridge_prompt = self.build_context_bridge(user_id=uid)
            if bridge_prompt:
                injected_prompts.append(bridge_prompt)

        # Check for Trivial Query Bypass
        skip_vector_search = self.is_trivial_utterance(user_text)

        if not skip_vector_search:
            # Run semantic facts search and episodic search in parallel
            future_facts = self._executor.submit(
                self.semantic.search_facts,
                query=user_text,
                user_id=uid,
                top_k=settings.SEMANTIC_TOP_K,
                min_similarity=settings.SEMANTIC_MIN_SIMILARITY,
            )
            future_episodes = self._executor.submit(
                self.episodic.search_episodes,
                query=user_text,
                user_id=uid,
                session_id=None,
                top_k=settings.EPISODIC_TOP_K,
                min_similarity=settings.EPISODIC_MIN_SIMILARITY,
            )

            # 1. Semantic Facts (What Ava knows about the user)
            try:
                facts = future_facts.result(timeout=4.0)
                if facts:
                    fact_bullets = "\n".join(
                        f"- {f['key'].replace('_', ' ').title()}: {f['value']}"
                        for f in facts
                    )
                    injected_prompts.append(
                        SystemMessage(content=f"Context from memory about the user:\n{fact_bullets}")
                    )
            except Exception as e:
                logger.warning(f"Semantic facts retrieval skipped/timed out: {e}")

            # 2. Episodic Memory (Past cross-session experiences)
            try:
                episodes = future_episodes.result(timeout=4.0)
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
                logger.warning(f"Episodic retrieval skipped/timed out: {e}")

        # 3. Running Conversation Summary (Evicted earlier turns in this session)
        try:
            summary = self.short_term.get_summary(session_id=session_id, user_id=uid)
            if summary:
                injected_prompts.append(
                    SystemMessage(content=f"Summary of earlier discussion in this conversation:\n{summary}")
                )
        except Exception as e:
            logger.warning(f"Summary retrieval skipped: {e}")

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
        Saves user and assistant messages to DB instantly, then dispatches background
        semantic fact extraction, incremental summary, and autonomous episode creation.
        Returns immediately (<2ms) so voice TTS is not delayed!
        """
        uid = user_id or self.user_id

        # 1. Save messages to short-term DB immediately
        try:
            self.short_term.add_user_message(session_id=session_id, content=user_text, user_id=uid)
            self.short_term.add_ai_message(session_id=session_id, content=ai_text, user_id=uid)
        except Exception as e:
            logger.error(f"Failed to record turn messages in DB: {e}")

        # 2. Asynchronous background distillation worker
        if extraction_llm:
            def _background_worker():
                # A. Semantic Fact Extraction
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
                    logger.warning(f"Background semantic fact processing failed: {e}")

                # B. Incremental Summary of Evicted Turns
                try:
                    self.short_term.update_summary_if_needed(
                        session_id=session_id,
                        llm=extraction_llm,
                        window_size=settings.MEMORY_WINDOW,
                        user_id=uid,
                    )
                except Exception as e:
                    logger.warning(f"Background summary update failed: {e}")

                # C. Autonomous Episode Creation on Closure Intent
                if self.is_closure_intent(user_text):
                    try:
                        ep = self.create_episode(
                            session_id=session_id,
                            extraction_llm=extraction_llm,
                            user_id=uid,
                        )
                        if ep:
                            logger.info(f"Auto-distilled episode on conversation closure: {ep['summary']}")
                    except Exception as e:
                        logger.warning(f"Auto-distillation on closure failed: {e}")

            threading.Thread(target=_background_worker, daemon=True).start()

    def auto_distill_if_needed(
        self,
        session_id: str,
        extraction_llm: Any,
        user_id: Optional[str] = None,
        min_messages: int = 4,
    ) -> Optional[Dict[str, Any]]:
        """
        Called when a session closes or switches.
        Automatically distills the session into an episode if it has enough messages.
        """
        uid = user_id or self.user_id
        try:
            messages = self.short_term.get_raw_messages(session_id=session_id, user_id=uid, limit=50)
            if len(messages) >= min_messages:
                return self.create_episode(session_id=session_id, extraction_llm=extraction_llm, user_id=uid)
        except Exception as e:
            logger.warning(f"Auto-distill check failed: {e}")
        return None

    def create_episode(
        self,
        session_id: str,
        extraction_llm: Any,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Distill the active conversation session into a persistent episodic memory."""
        uid = user_id or self.user_id
        messages = self.short_term.get_raw_messages(session_id=session_id, user_id=uid, limit=50)
        if not messages or len(messages) < 2:
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

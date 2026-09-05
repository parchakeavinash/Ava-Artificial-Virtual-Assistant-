import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage

from app.config.settings import settings
from app.database.db import get_db_session
from app.memory.embeddings import EmbeddingProvider, GeminiEmbeddingProvider
from app.memory.models import SemanticFact

logger = logging.getLogger(__name__)


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class SemanticMemoryManager:
    """
    Manages persistent user facts (semantic memory).
    Facts are user-scoped and survive across sessions:
      - Preferences  ("I prefer Python")
      - Identity     ("My name is Avi")
      - Projects     ("I'm building an AI voice agent")
      - Tools/Stack  ("I use PostgreSQL")
    """

    KEY_SIMILARITY_THRESHOLD = 0.92

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ):
        self.embedding_provider = embedding_provider or GeminiEmbeddingProvider()
        self.top_k = top_k or settings.SEMANTIC_TOP_K
        self.min_similarity = min_similarity or settings.SEMANTIC_MIN_SIMILARITY

    def classify_message(self, user_message: str, llm) -> bool:
        """
        Fast binary classification: does this message reveal a personal fact?
        Returns True if yes, False if it's a question/general statement.
        """
        prompt = [
            SystemMessage(
                content=(
                    "You are a classifier for Ava, an AI assistant. Determine whether the user's message contains "
                    "a personal fact about the user themselves — such as a preference, "
                    "personal detail, project they are working on, tool they use, or goal they have.\n\n"
                    "Reply with ONLY the word YES or NO.\n\n"
                    "Examples:\n"
                    "  'My favorite language is Python.' → YES\n"
                    "  'I am building an AI voice agent.' → YES\n"
                    "  'My name is Avi.' → YES\n"
                    "  'I prefer PostgreSQL over MySQL.' → YES\n"
                    "  'What is the speed of light?' → NO\n"
                    "  'Hello' → NO\n"
                    "  'Create a task to buy groceries.' → NO\n"
                    "  'Explain Python lists.' → NO"
                )
            ),
            HumanMessage(content=f"User message: {user_message}"),
        ]
        try:
            response = llm.invoke(prompt)
            answer = str(response.content).strip().upper()
            return answer.startswith("YES")
        except Exception as e:
            logger.warning(f"Semantic classifier failed: {e}")
            return False

    def extract_fact(self, user_message: str, llm) -> Optional[Dict[str, str]]:
        """
        Extracts a single key/value fact from a user message.
        Returns {"key": "...", "value": "..."} or None.
        """
        prompt = [
            SystemMessage(
                content=(
                    "You are a fact extraction system for Ava. Extract a single key-value fact "
                    "from the user's message.\n\n"
                    "RULES:\n"
                    "1. key must be a short snake_case label (e.g. favorite_language, current_project, name)\n"
                    "2. value must be the specific fact the user stated\n"
                    "3. Extract ONLY what is explicitly stated — do not infer or expand\n"
                    "4. If there are multiple facts, pick the most important one\n\n"
                    "Output MUST be valid JSON with exactly this schema:\n"
                    '{"key": "snake_case_label", "value": "the fact"}\n\n'
                    "Examples:\n"
                    "  'My favorite language is Python.'         → {\"key\": \"favorite_language\", \"value\": \"Python\"}\n"
                    "  'I am building an AI voice agent.'        → {\"key\": \"current_project\", \"value\": \"AI voice agent\"}\n"
                    "  'My name is Avi.'                         → {\"key\": \"name\", \"value\": \"Avi\"}\n"
                    "  'I prefer PostgreSQL for my database.'    → {\"key\": \"preferred_database\", \"value\": \"PostgreSQL\"}\n\n"
                    "Return ONLY raw JSON. No markdown, no explanation."
                )
            ),
            HumanMessage(content=f"User message: {user_message}"),
        ]
        try:
            response = llm.invoke(prompt)
            raw = str(response.content).strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0]
            data = json.loads(raw.strip())
            key = data.get("key", "").strip()
            value = data.get("value", "").strip()
            if key and value:
                return {"key": key, "value": value}
            return None
        except Exception as e:
            logger.warning(f"Semantic fact extraction failed: {e}")
            return None

    def upsert_fact(
        self,
        user_id: str,
        key: str,
        value: str,
        source: Optional[str] = None,
    ) -> Tuple[str, Optional[SemanticFact]]:
        """
        Stores or updates a semantic fact.
        Returns ("created"|"updated"|"ignored", fact_record).
        """
        embed_text = f"{key}: {value}"
        vector = self._embed(embed_text)

        with get_db_session() as db:
            existing = (
                db.query(SemanticFact)
                .filter(SemanticFact.user_id == user_id, SemanticFact.key == key)
                .first()
            )

            if existing:
                if existing.value.strip().lower() == value.strip().lower():
                    logger.debug(f"Semantic fact unchanged: [{user_id}] {key}={value}")
                    return ("ignored", existing)
                else:
                    logger.info(f"Updating semantic fact: [{user_id}] {key}: '{existing.value}' → '{value}'")
                    existing.value = value
                    existing.source = source or existing.source
                    existing.embedding = vector
                    existing.updated_at = datetime.now(timezone.utc)
                    db.flush()
                    db.refresh(existing)
                    return ("updated", existing)
            else:
                logger.info(f"Creating semantic fact: [{user_id}] {key}={value}")
                fact = SemanticFact(
                    user_id=user_id,
                    key=key,
                    value=value,
                    source=source,
                    embedding=vector,
                )
                db.add(fact)
                db.flush()
                db.refresh(fact)
                return ("created", fact)

    def search_facts(
        self,
        query: str,
        user_id: str,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves relevant semantic facts for the user by cosine similarity."""
        k = top_k if top_k is not None else self.top_k
        threshold = min_similarity if min_similarity is not None else self.min_similarity

        query_vector = self._embed(query)
        if not query_vector:
            return []

        with get_db_session() as db:
            facts = db.query(SemanticFact).filter(SemanticFact.user_id == user_id).all()

        results = []
        for fact in facts:
            if not fact.embedding:
                continue
            sim = _cosine_similarity(query_vector, fact.embedding)
            if sim >= threshold:
                results.append({
                    "key": fact.key,
                    "value": fact.value,
                    "source": fact.source,
                    "similarity": round(sim, 4),
                    "updated_at": fact.updated_at.isoformat() if fact.updated_at else None,
                })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:k]

    def process_message(
        self,
        user_message: str,
        user_id: str,
        llm,
    ) -> Optional[Tuple[str, str, str]]:
        """
        Full pipeline: classify → extract → upsert.
        Returns (action, key, value) if a fact was processed, else None.
        """
        if len(user_message.strip()) < 8:
            return None

        is_personal = self.classify_message(user_message, llm)
        if not is_personal:
            return None

        fact = self.extract_fact(user_message, llm)
        if not fact:
            return None

        action, _ = self.upsert_fact(
            user_id=user_id,
            key=fact["key"],
            value=fact["value"],
            source=user_message,
        )
        return (action, fact["key"], fact["value"])

    def list_facts(self, user_id: str) -> List[SemanticFact]:
        with get_db_session() as db:
            return (
                db.query(SemanticFact)
                .filter(SemanticFact.user_id == user_id)
                .order_by(SemanticFact.updated_at.desc())
                .all()
            )

    def delete_fact(self, user_id: str, key: str) -> bool:
        with get_db_session() as db:
            deleted = (
                db.query(SemanticFact)
                .filter(SemanticFact.user_id == user_id, SemanticFact.key == key)
                .delete()
            )
            return deleted > 0

    def clear_facts(self, user_id: str) -> int:
        with get_db_session() as db:
            return db.query(SemanticFact).filter(SemanticFact.user_id == user_id).delete()

    def _embed(self, text: str) -> Optional[List[float]]:
        try:
            return self.embedding_provider.embed(text)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None

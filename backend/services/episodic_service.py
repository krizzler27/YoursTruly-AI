"""Episodic write plus read over older chat turns, merged into memory.

Write freezes the last 2 turns verbatim and summarizes the older head
via summarize_text before any generation acquire. Read ranks stored
summaries by token overlap, recency breaking ties.
"""

from typing import Dict, List, Optional
import uuid

from sqlalchemy.orm import Session

from config import config
from core.context_budget import count_tokens
from core.logging import get_logger
from repository.episodic_repository import EpisodicMemoryRepository
from repository.semantic_repository import (
    MEMORY_VECTOR_MIN_SCORE,
    cosine_sim,
    memory_tokens,
)
import services.summarize_service as summarize_mod

logger = get_logger(__name__)

ROLLUP_EVERY_N_TURNS = 10
ROLLUP_MAX_TOKENS = 128


class EpisodicService:
    """Rollup older turns into per-chat summaries, recall top few."""

    def __init__(self, db: Session):
        self.db = db
        self.repo = EpisodicMemoryRepository(db)

    def rollup_if_needed(
        self,
        conversation_id: uuid.UUID,
        turns: Optional[List[Dict[str, str]]],
        usable_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """Summarize the dropped prefix when triggered, else None.

        Triggers on history overflow (used over SUMMARY_TRIGGER of usable)
        or every 10 turns. Never summarizes the kept last-2 suffix.
        Busy engine, timeout, or model failure returns None so the caller
        falls back to truncate-oldest.
        """
        items = list(turns or [])
        if len(items) <= 2:
            return None
        if not self._triggered(items, usable_tokens):
            return None
        head = items[:-2]
        head_text = "\n".join(
            f"{'User' if (t or {}).get('role') == 'user' else 'Assistant'}: {(t or {}).get('content', '')}"
            for t in head
        ).strip()
        if not head_text:
            return None
        try:
            from services.llama_engine import LlamaEngine

            if LlamaEngine.get_instance("chat").is_generating():
                logger.warning("summary_skipped_busy slot=chat episodic")
                return None
        except Exception as e:
            logger.debug("episodic busy check skipped: %s", e)
        try:
            summary = summarize_mod.summarize_text(head_text, max_tokens=ROLLUP_MAX_TOKENS)
        except Exception as e:
            logger.warning("summary_skipped_busy episodic err=%s", e)
            return None
        if not (summary or "").strip():
            return None
        clean = summary.strip()
        existing = self.repo.list_by_conversation(conversation_id, limit=1000)
        start = max([r.turn_end for r in existing] + [0])
        self.repo.create(conversation_id, clean, start, start + len(head))
        logger.info("episodic rollup stored conv=%s turns=%s", conversation_id, len(head))
        return clean

    def _triggered(self, turns: List[Dict[str, str]], usable_tokens: Optional[int]) -> bool:
        """True on every-10-turns cadence or usable-budget overflow."""
        if len(turns) >= ROLLUP_EVERY_N_TURNS and len(turns) % ROLLUP_EVERY_N_TURNS == 0:
            return True
        if usable_tokens:
            used = sum(count_tokens((t or {}).get("content") or "") for t in turns)
            if used > float(config.SUMMARY_TRIGGER) * int(usable_tokens):
                return True
        return False

    def recall(
        self,
        conversation_id: uuid.UUID,
        query: str,
        limit: int = 3,
        query_vector: Optional[List[float]] = None,
        candidate_vectors: Optional[Dict[str, List[float]]] = None,
    ) -> List[str]:
        """Top summaries by token overlap, vector-only hits filling the rest.

        No vector keeps today's overlap-then-recency order. With a
        query_vector plus candidate_vectors (row id string to vector over
        the summary text), cosine rescues paraphrases into slots past the
        overlap hits, recency breaking ties.
        """
        try:
            rows = self.repo.list_recent_for_query(conversation_id, limit=10)
        except Exception as e:
            logger.debug("episodic recall skipped: %s", e)
            return []
        toks = memory_tokens(query)
        scored = []
        for idx, row in enumerate(rows or []):
            text = (getattr(row, "summary", "") or "").strip()
            if not text:
                continue
            overlap = len(toks & memory_tokens(text)) if toks else 0
            scored.append((overlap, -idx, text, row))
        if not (query_vector and candidate_vectors):
            scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
            return [text for _, _, text, _ in scored[: max(0, limit)]]
        ranked = sorted(
            [(ov, neg, text) for ov, neg, text, _ in scored if ov > 0],
            key=lambda s: (s[0], s[1]),
            reverse=True,
        )
        vector_only = []
        for overlap, neg, text, row in scored:
            if overlap > 0:
                continue
            vec = candidate_vectors.get(str(getattr(row, "id", "")))
            sim = cosine_sim(query_vector, vec) if vec else 0.0
            if sim >= MEMORY_VECTOR_MIN_SCORE:
                vector_only.append((sim, neg, text))
        vector_only.sort(key=lambda s: (s[0], s[1]), reverse=True)
        ranked.extend(vector_only)
        return [text for _, _, text in ranked[: max(0, limit)]]

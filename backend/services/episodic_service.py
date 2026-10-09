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
EPISODIC_ROW_CAP = 20
EPISODIC_MERGE_PROTECT_NEWEST = 3
EPISODIC_FORGET_PROTECT_NEWEST = 10


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
        existing = self.repo.list_by_conversation(conversation_id, limit=1000)
        start = max([r.turn_end for r in existing] + [0])
        # Incremental fix: only summarize messages after `start`.
        # `items` is the last-100 window; map absolute `start` to a local
        # offset so re-rollups don't re-summarize rows 0..start.
        offset = 0
        try:
            from db.models import MessagesModel

            db_total = (
                self.db.query(MessagesModel)
                .filter(MessagesModel.conversation_id == conversation_id)
                .count()
            )
            if db_total >= len(items):
                offset = max(0, db_total - len(items))
        except Exception:
            pass
        effective_start = max(int(start), int(offset))
        local_start = effective_start - offset
        if local_start >= len(items) - 2:
            return None
        head = items[local_start:-2] if local_start > 0 else items[:-2]
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
        end = effective_start + len(head)
        try:
            from services.semantic_writer import _embed_texts

            vecs = _embed_texts([clean])
            vec = vecs[0] if vecs else None
        except Exception:
            vec = None
        self.repo.create(conversation_id, clean, effective_start, end, embedding=vec)
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
        """Top summaries by token overlap, stored plus override vectors after.

        No vector keeps today's overlap-then-recency order. With a
        query_vector, cosine over stored row vectors (candidate_vectors
        entries winning ties) rescues paraphrases into slots past the
        overlap hits, recency breaking ties. Returned overlap hits and
        over-threshold vector hits bump recall_count in one commit.
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
        if not query_vector:
            scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
            top = scored[: max(0, limit)]
            self._touch_hits([r for ov, _, _, r in top if ov > 0])
            return [text for _, _, text, _ in top]
        ranked = sorted(
            [(ov, neg, text, row) for ov, neg, text, row in scored if ov > 0],
            key=lambda s: (s[0], s[1]),
            reverse=True,
        )
        vector_only = []
        for overlap, neg, text, row in scored:
            if overlap > 0:
                continue
            vec = (candidate_vectors or {}).get(str(getattr(row, "id", "")))
            if vec is None:
                try:
                    from repository.semantic_repository import decode_embedding

                    vec = decode_embedding(getattr(row, "embedding", None))
                except Exception:
                    vec = None
            sim = cosine_sim(query_vector, vec) if vec else 0.0
            if sim >= MEMORY_VECTOR_MIN_SCORE:
                vector_only.append((sim, neg, text, row))
        vector_only.sort(key=lambda s: (s[0], s[1]), reverse=True)
        ranked.extend(vector_only)
        top = ranked[: max(0, limit)]
        self._touch_hits([r for _, _, _, r in top])
        return [text for _, _, text, _ in top]

    def _touch_hits(self, rows) -> None:
        """Bump recall_count plus stamp on hit rows, fail-open."""
        try:
            ids = [getattr(r, "id", None) for r in (rows or [])]
            ids = [i for i in ids if i is not None]
            if ids:
                self.repo.mark_recalled(ids)
        except Exception as e:
            logger.debug("episodic recall touch skipped: %s", e)

    def compact_if_needed(self, conversation_id: uuid.UUID) -> Dict[str, object]:
        """Merge one oldest pair plus forget zero-hit rows, fail-open.

        Runs after rollup_if_needed in the rollup worker. When rows exceed
        EPISODIC_ROW_CAP, merges exactly one oldest adjacent pair outside the
        newest-3 recall backing, then forgets zero-hit rows oldest-first
        outside the newest-10 recall pool. Merged rows carry min/max span and
        copy the oldest parent created_at so created_at ordering stays
        chronological. One pair per call keeps growth O(1) amortized.
        """
        result: Dict[str, object] = {"compacted": 0, "merged_turns": "", "forgot": 0}
        try:
            rows = self.repo.list_ordered_for_compaction(conversation_id, limit=1000)
        except Exception as e:
            logger.debug("episodic compact skipped: %s", e)
            return result
        if len(rows) <= EPISODIC_ROW_CAP:
            return result
        # Merge exactly one oldest adjacent pair, never touching newest-3.
        merged_id = None
        if len(rows) > EPISODIC_MERGE_PROTECT_NEWEST:
            protect = EPISODIC_MERGE_PROTECT_NEWEST
            if 1 < len(rows) - protect:
                first, second = rows[0], rows[1]
                merged_span, merged_id = self._merge_oldest_pair(
                    conversation_id, first, second
                )
                if merged_span:
                    result["compacted"] = 1
                    result["merged_turns"] = merged_span
                else:
                    merged_id = None
                try:
                    rows = self.repo.list_ordered_for_compaction(
                        conversation_id, limit=1000
                    )
                except Exception as e:
                    logger.debug("episodic compact relist skipped: %s", e)
        # Forget zero-hit rows oldest-first when still over cap.
        try:
            if len(rows) > EPISODIC_ROW_CAP:
                forgot = self._forget_zero_hits(rows, exclude_ids={merged_id} if merged_id else None)
                result["forgot"] = forgot
        except Exception as e:
            logger.debug("episodic forget skipped: %s", e)
        logger.info(
            "episodic compact conv=%s compacted=%s merged_turns=%s forgot=%s rows=%s",
            conversation_id,
            result["compacted"],
            result["merged_turns"],
            result["forgot"],
            len(rows),
        )
        return result

    def _merge_oldest_pair(self, conversation_id: uuid.UUID, first, second):
        """Merge two adjacent rows via worker-slot summary, else empty span.

        Returns (span, new_id): span is "" plus None on abort so the caller
        keeps both parents. The new row id lets the same-pass forget step
        spare the just-merged row instead of deleting it as oldest zero-hit.
        """
        combined = (
            ((getattr(first, "summary", "") or "").strip()
            + "\n"
            + (getattr(second, "summary", "") or "").strip()).strip()
        )
        if not combined:
            return "", None
        try:
            merged = summarize_mod.summarize_text(
                combined, max_tokens=ROLLUP_MAX_TOKENS, role="worker"
            )
        except Exception as e:
            logger.debug("episodic compact merge skipped: %s", e)
            return "", None
        if not (merged or "").strip():
            return "", None
        clean = merged.strip()
        try:
            parent_tokens = count_tokens(
                getattr(first, "summary", "") or ""
            ) + count_tokens(getattr(second, "summary", "") or "")
            if count_tokens(clean) > parent_tokens:
                logger.debug("episodic compact merge grew tokens, abort")
                return "", None
        except Exception as e:
            logger.debug("episodic compact token check skipped: %s", e)
            return "", None
        start = min(int(first.turn_start), int(second.turn_start))
        end = max(int(first.turn_end), int(second.turn_end))
        try:
            # Create merged row FIRST so a failure keeps both parents.
            new_row = self.repo.create(conversation_id, clean, start, end)
            try:
                stamps = [
                    c
                    for c in (
                        getattr(first, "created_at", None),
                        getattr(second, "created_at", None),
                    )
                    if c is not None
                ]
                if stamps:
                    new_row.created_at = min(stamps)
                hits = [
                    int(getattr(r, "recall_count", 0) or 0) for r in (first, second)
                ]
                new_row.recall_count = max(hits) if hits else 0
                times = [
                    getattr(r, "last_recalled_at", None) for r in (first, second)
                ]
                times = [t for t in times if t is not None]
                if times:
                    new_row.last_recalled_at = max(times)
                self.db.commit()
            except Exception as e:
                logger.debug("episodic compact carry skipped: %s", e)
            self.repo.delete_many([first.id, second.id])
        except Exception as e:
            logger.debug("episodic compact merge write skipped: %s", e)
            return "", None
        return f"{start}-{end}", new_row.id

    def _forget_zero_hits(self, rows, exclude_ids=None) -> int:
        """Delete zero-hit rows oldest-first outside newest-10 pool."""
        need = len(rows) - EPISODIC_ROW_CAP
        if need <= 0:
            return 0
        skipped = {str(i) for i in (exclude_ids or set()) if i is not None}
        pool_end = max(0, len(rows) - EPISODIC_FORGET_PROTECT_NEWEST)
        victims = [
            r
            for r in rows[:pool_end]
            if int(getattr(r, "recall_count", 0) or 0) == 0
            and str(getattr(r, "id", "")) not in skipped
        ][:need]
        if not victims:
            return 0
        try:
            return int(self.repo.delete_many([r.id for r in victims]))
        except Exception as e:
            logger.debug("episodic forget delete skipped: %s", e)
            return 0

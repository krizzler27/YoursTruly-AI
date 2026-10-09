from datetime import datetime, timezone
from typing import List
import uuid

from sqlalchemy.orm import Session

from db.models import EpisodicMemoryModel
from repository.base_repository import BaseRepository


class EpisodicMemoryRepository(BaseRepository[EpisodicMemoryModel]):
    """CRUD over per-chat episodic summaries - rollup write, recall read."""

    def __init__(self, db: Session):
        super().__init__(EpisodicMemoryModel, db)

    def create(
        self,
        conversation_id: uuid.UUID,
        summary: str,
        turn_start: int,
        turn_end: int,
        embedding=None,
    ) -> EpisodicMemoryModel:
        """Append one summary covering turns [turn_start, turn_end)."""
        from repository.semantic_repository import encode_embedding

        return super().create(
            conversation_id=conversation_id,
            summary=summary or "",
            turn_start=int(turn_start),
            turn_end=int(turn_end),
            embedding=encode_embedding(embedding),
        )

    def list_by_conversation(
        self, conversation_id: uuid.UUID, limit: int = 100
    ) -> List[EpisodicMemoryModel]:
        """One chat's summaries oldest first, capped at the latest limit."""
        rows = (
            self.db.query(self.model)
            .filter(self.model.conversation_id == conversation_id)
            .order_by(self.model.created_at.desc(), self.model.id.desc())
            .limit(limit)
            .all()
        )
        return list(reversed(rows))

    def list_ordered_for_compaction(
        self, conversation_id: uuid.UUID, limit: int = 1000
    ) -> List[EpisodicMemoryModel]:
        """One chat's summaries ordered by turn span for compaction.

        Ordered by turn_start then turn_end so merged rows (turn_start=min,
        turn_end=max) sort back into chronological position. Merged rows copy
        the oldest parent created_at so list_by_conversation (created_at order)
        stays consistent.
        """
        return (
            self.db.query(self.model)
            .filter(self.model.conversation_id == conversation_id)
            .order_by(
                self.model.turn_start.asc(),
                self.model.turn_end.asc(),
                self.model.created_at.asc(),
                self.model.id.asc(),
            )
            .limit(limit)
            .all()
        )

    def list_recent_for_query(
        self, conversation_id: uuid.UUID, limit: int = 10
    ) -> List[EpisodicMemoryModel]:
        """Newest summaries first - the recall candidate pool."""
        return (
            self.db.query(self.model)
            .filter(self.model.conversation_id == conversation_id)
            .order_by(self.model.created_at.desc(), self.model.id.desc())
            .limit(limit)
            .all()
        )

    def mark_recalled(self, row_ids: List[uuid.UUID]) -> int:
        """Bump recall_count and stamp last_recalled_at in one commit."""
        ids = list(dict.fromkeys(row_ids or []))
        if not ids:
            return 0
        now = datetime.now(timezone.utc)
        rows = (
            self.db.query(self.model).filter(self.model.id.in_(ids)).all()
        )
        for row in rows:
            try:
                row.recall_count = int(getattr(row, "recall_count", 0) or 0) + 1
            except Exception:
                row.recall_count = 1
            row.last_recalled_at = now
        self.db.commit()
        return len(rows)

    def delete_by_conversation(self, conversation_id: uuid.UUID) -> int:
        """Delete one chat's summaries, returning the row count."""
        count = (
            self.db.query(self.model)
            .filter(self.model.conversation_id == conversation_id)
            .delete(synchronize_session=False)
        )
        self.db.commit()
        return int(count)

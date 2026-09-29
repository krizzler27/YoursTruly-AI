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
    ) -> EpisodicMemoryModel:
        """Append one summary covering turns [turn_start, turn_end)."""
        return super().create(
            conversation_id=conversation_id,
            summary=summary or "",
            turn_start=int(turn_start),
            turn_end=int(turn_end),
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

    def delete_by_conversation(self, conversation_id: uuid.UUID) -> int:
        """Delete one chat's summaries, returning the row count."""
        count = (
            self.db.query(self.model)
            .filter(self.model.conversation_id == conversation_id)
            .delete(synchronize_session=False)
        )
        self.db.commit()
        return int(count)

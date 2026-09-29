from typing import List, Optional
import re
import uuid

from sqlalchemy.orm import Session

from db.models import SemanticMemoryModel
from repository.base_repository import BaseRepository


def memory_tokens(text: str) -> set:
    """Lowercase alnum tokens len 3 plus, the key-overlap unit."""
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) >= 3}


class SemanticMemoryRepository(BaseRepository[SemanticMemoryModel]):
    """CRUD over durable user facts - semantic recall across chats."""

    def __init__(self, db: Session):
        super().__init__(SemanticMemoryModel, db)

    def get_by_key(self, key: str) -> Optional[SemanticMemoryModel]:
        """One fact by normalized key, None when absent."""
        clean = (key or "").strip().lower()
        if not clean:
            return None
        return self.db.query(self.model).filter(self.model.key == clean).first()

    def upsert(self, key: str, value: str) -> SemanticMemoryModel:
        """Insert or replace one fact by normalized key."""
        clean = (key or "").strip().lower()
        if not clean:
            raise ValueError("key cannot be empty")
        row = self.get_by_key(clean)
        if row is None:
            row = self.model(key=clean, value=value or "")
            self.db.add(row)
        else:
            row.value = value or ""
        self.db.commit()
        self.db.refresh(row)
        return row

    def list_all(
        self, limit: int = 100, skip: int = 0
    ) -> List[SemanticMemoryModel]:
        """Newest facts first, capped for overlap scoring."""
        return (
            self.db.query(self.model)
            .order_by(self.model.updated_at.desc(), self.model.id.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    def find_relevant(self, query: str, limit: int = 5) -> List[SemanticMemoryModel]:
        """Top facts by key-token overlap with the query, best first."""
        toks = memory_tokens(query)
        if not toks:
            return []
        scored = []
        for row in self.list_all(limit=100):
            if len(toks & memory_tokens(row.key or "")):
                scored.append(row)
        scored.sort(
            key=lambda r: len(toks & memory_tokens(r.key or "")), reverse=True
        )
        return scored[: max(0, limit)]

    def delete_by_key(self, key: str) -> bool:
        """Delete one fact by normalized key, False when absent."""
        row = self.get_by_key(key)
        if row is None:
            return False
        self.db.delete(row)
        self.db.commit()
        return True

    def delete(self, id_or_key: uuid.UUID | str) -> bool:
        """Delete by UUID id or fact key, False when absent."""
        if isinstance(id_or_key, uuid.UUID):
            return super().delete(id_or_key)
        try:
            return super().delete(uuid.UUID(str(id_or_key)))
        except Exception:
            pass
        return self.delete_by_key(str(id_or_key))

from typing import Dict, List, Optional
import json
import re
import uuid

from sqlalchemy.orm import Session

from db.models import SemanticMemoryModel
from repository.base_repository import BaseRepository

MEMORY_EMBED_CHARS = 200
MEMORY_VECTOR_MIN_SCORE = 0.5


def memory_tokens(text: str) -> set:
    """Lowercase alnum tokens len 3 plus, the key-overlap unit."""
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) >= 3}


def memory_embed_text(text: str, max_chars: int = MEMORY_EMBED_CHARS) -> str:
    """Recall-time candidate text clipped for one batched embed call."""
    return ((text or "").strip())[:max_chars]


def cosine_sim(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    """Cosine similarity, 0.0 on empty, ragged, or dim mismatch."""
    if not isinstance(a, list) or not isinstance(b, list):
        return 0.0
    if not a or len(a) != len(b):
        return 0.0
    try:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
    except (TypeError, ArithmeticError):
        return 0.0
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def encode_embedding(vec: Optional[List[float]]) -> Optional[str]:
    """JSON text for one vector, None when empty or bad."""
    try:
        if not vec or not isinstance(vec, list):
            return None
        return json.dumps([float(x) for x in vec])
    except (TypeError, ValueError):
        return None


def decode_embedding(raw: Optional[str]) -> Optional[List[float]]:
    """Stored JSON text to vector, None on empty or bad data."""
    try:
        if not raw or not isinstance(raw, str):
            return None
        vals = json.loads(raw)
        if not isinstance(vals, list) or not vals:
            return None
        return [float(x) for x in vals]
    except (TypeError, ValueError):
        return None


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

    def upsert(
        self, key: str, value: str, embedding: Optional[List[float]] = None
    ) -> SemanticMemoryModel:
        """Insert or replace one fact by normalized key."""
        clean = (key or "").strip().lower()
        if not clean:
            raise ValueError("key cannot be empty")
        row = self.get_by_key(clean)
        encoded = encode_embedding(embedding)
        if row is None:
            row = self.model(key=clean, value=value or "", embedding=encoded)
            self.db.add(row)
        else:
            row.value = value or ""
            row.embedding = encoded
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

    def find_relevant(
        self,
        query: str,
        limit: int = 5,
        query_vector: Optional[List[float]] = None,
        candidate_vectors: Optional[Dict[str, List[float]]] = None,
    ) -> List[SemanticMemoryModel]:
        """Top facts by key-token overlap, vector-only hits filling the rest.

        Overlap ranks first so exact matches stay put; with a query_vector
        stored vectors (plus optional candidate_vectors override) cosine
        rescues paraphrases into remaining slots. No vector keeps
        today's overlap-only behavior.
        """
        toks = memory_tokens(query)
        if not toks and not query_vector:
            return []
        scored = []
        for row in self.list_all(limit=100):
            overlap = len(toks & memory_tokens(row.key or "")) if toks else 0
            scored.append((overlap, row))
        scored.sort(key=lambda p: p[0], reverse=True)
        ranked = [row for overlap, row in scored if overlap > 0]
        if query_vector:
            effective: Dict[str, List[float]] = {}
            for overlap, row in scored:
                if overlap > 0:
                    continue
                try:
                    stored = decode_embedding(getattr(row, "embedding", None))
                except Exception:
                    stored = None
                if stored:
                    effective[row.key or ""] = stored
            if candidate_vectors:
                for k, v in candidate_vectors.items():
                    if v:
                        effective[k] = v
            if effective:
                vector_only = []
                for overlap, row in scored:
                    if overlap > 0:
                        continue
                    vec = effective.get(row.key or "")
                    sim = cosine_sim(query_vector, vec) if vec else 0.0
                    if sim >= MEMORY_VECTOR_MIN_SCORE:
                        vector_only.append((sim, row))
                vector_only.sort(key=lambda p: p[0], reverse=True)
                ranked.extend(row for _, row in vector_only)
        return ranked[: max(0, limit)]

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

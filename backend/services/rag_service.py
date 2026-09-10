"""RAG read path + document lifecycle.

Ingest writes stay in IngestService (single-commit atomicity).
This service orchestrates: embed query → hybrid search, plus
document list/delete with store cleanup.
"""

import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from db.models import DocumentsModel
from repository.document_repository import DocumentRepository
from repository.lance_repository import LanceRepository
from services.ingest_service import IngestService
from services.llama_service import EmbeddingEngine


class RagService:
    """Search and document management over the hybrid chunk store."""

    def __init__(
        self,
        db: Session,
        engine: Optional[EmbeddingEngine] = None,
        lance: Optional[LanceRepository] = None,
        docs: Optional[DocumentRepository] = None,
    ):
        self.db = db
        self.engine = engine or EmbeddingEngine.get_instance()
        self.lance = lance or LanceRepository(db)
        self.docs = docs or DocumentRepository(db)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Embed the query, fuse vector + BM25 candidates, hydrate hits."""
        clean = (query or "").strip()

        if not clean:
            raise ValueError("query cannot be empty")

        vectors = self.engine.embed([clean])

        if not vectors or not vectors[0]:
            raise RuntimeError("embed returned no vector for query")

        return self.lance.hybrid_search(clean, vectors[0], top_k=top_k)

    def ingest_file(self, source_path: str, filename: str) -> DocumentsModel:
        """Index one uploaded file via the Phase 1 atomic ingest path."""
        svc = IngestService(db=self.db, engine=self.engine, lance=self.lance)

        return svc.ingest(source_path, filename)

    def list_documents(self, limit: int = 100) -> List[DocumentsModel]:
        """List ingested files newest first."""
        return self.docs.list_recent(limit=limit)

    def delete_document(self, document_id: uuid.UUID) -> uuid.UUID:
        """Remove a file row plus its vectors, FTS entries, and chunks."""
        existing = self.docs.get_by_id(document_id)

        if existing is None:
            raise ValueError(f"Document {document_id} not found")

        self.lance.delete_document(existing.id, commit=False)

        ok = self.docs.delete(existing.id)

        if not ok:
            raise ValueError(f"Document {document_id} not found")

        return existing.id

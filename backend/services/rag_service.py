"""RAG read path + document lifecycle (ingest writes live in IngestService)."""

import uuid
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from db.models import DocumentsModel
from core.logging import get_logger
from core.trace import traceable
from repository.document_repository import DocumentRepository
from repository.lance_repository import LanceRepository, sid
from schemas.rag_schemas import SearchHit
from services.llama_engine import EmbeddingEngine
from core.context_budget import (
    allocate,
    count_tokens,
    fit_history,
    fit_hits,
    rag_context_tokens as _derived_rag_tokens,
)
from services.llm_service import LLMService
from services.prompt_manager import PromptManager

logger = get_logger(__name__)


def rag_context_tokens(n_ctx: Optional[int] = None) -> int:
    """Retrieved-context budget derived from the chat window, never fixed."""
    return _derived_rag_tokens(n_ctx)


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
        self.engine = engine or EmbeddingEngine.get_instance("embed")
        self.lance = lance or LanceRepository(db)
        self.docs = docs or DocumentRepository(db)

    @traceable(name="rag.search")
    def search(
        self, query: str, top_k: int = 5, conversation_id: Optional[uuid.UUID] = None
    ) -> List[SearchHit]:
        """Embed the query, fuse vector + BM25 candidates, hydrate hits."""
        clean = (query or "").strip()

        if not clean:
            raise ValueError("query cannot be empty")

        logger.debug("search start q=%.100s top_k=%s", clean, top_k)
        vectors = self.engine.embed([clean])

        if not vectors or not vectors[0]:
            raise RuntimeError("embed returned no vector for query")

        doc_ids = None
        if conversation_id is not None:
            doc_ids = [
                sid(d.id)
                for d in self.docs.list_by_conversation(conversation_id, limit=1000)
            ]
            if not doc_ids:
                return []

        hits = self.lance.hybrid_search(
            clean, vectors[0], top_k=top_k, conversation_id=conversation_id, doc_ids=doc_ids
        )
        logger.info("search done hits=%s", len(hits))
        return hits

    def list_documents(
        self, limit: int = 100, conversation_id: Optional[uuid.UUID] = None
    ) -> List[DocumentsModel]:
        """List ingested files newest first, optionally one chat's."""
        if conversation_id is None:
            return self.docs.list_recent(limit=limit)
        return self.docs.list_by_conversation(conversation_id, limit=limit)

    def delete_document(self, document_id: uuid.UUID) -> uuid.UUID:
        """Remove a file row plus its vectors, FTS entries, and chunks."""
        from services.ingest_service import stored_upload_path

        existing = self.docs.get_by_id(document_id)

        if existing is None:
            raise ValueError(f"Document {document_id} not found")

        self.lance.delete_document(existing.id, commit=False)

        ok = self.docs.delete(existing.id)

        if not ok:
            raise ValueError(f"Document {document_id} not found")

        if existing.conversation_id is not None:
            try:
                stored_upload_path(existing.conversation_id, existing.filename).unlink(
                    missing_ok=True
                )
            except Exception:
                pass

        return existing.id

    def _filenames(self, document_ids) -> Dict[str, str]:
        """One query mapping dashed document ids to filenames."""
        ids = []
        for raw in document_ids:
            try:
                ids.append(uuid.UUID(str(raw)))
            except Exception:
                continue
        if not ids:
            return {}
        rows = (
            self.db.query(DocumentsModel)
            .filter(DocumentsModel.id.in_(ids))
            .all()
        )
        return {sid(r.id): r.filename for r in rows}

    def build_messages(
        self,
        query: str,
        hits: Optional[List[SearchHit]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_context_tokens: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """One builder: plain chat without hits, grounded prompt with hits."""
        clean = (query or "").strip()
        if not hits:
            budget = allocate(
                route="DIRECT", query_tokens=count_tokens(clean)
            )
            fitted_history, truncated, hist_used = fit_history(
                history or [], budget["history_cap"]
            )
            logger.info(
                "budget total=%s usable=%s history=%s rag=%s mem=%s route=%s overflow=%s",
                budget["total"],
                budget["usable"],
                hist_used,
                0,
                budget["mem_cap"],
                "DIRECT",
                truncated,
            )
            return LLMService.build_chat_messages(fitted_history, clean)

        logger.debug("build grounded hits=%s", len(hits))
        budget = allocate(route="RAG", query_tokens=count_tokens(clean))
        rag_cap = max_context_tokens or budget["rag_cap"]
        fitted, hits_truncated, rag_used = fit_hits(hits, rag_cap)
        fitted_history, hist_truncated, hist_used = fit_history(
            history or [], budget["history_cap"]
        )
        overflow = bool(hits_truncated or hist_truncated)
        logger.info(
            "budget total=%s usable=%s history=%s rag=%s mem=%s route=%s overflow=%s",
            budget["total"],
            budget["usable"],
            hist_used,
            rag_used,
            budget["mem_cap"],
            "RAG",
            overflow,
        )
        names = self._filenames(h.document_id for h in fitted)

        blocks = []
        for h in fitted:
            name = names.get(h.document_id, h.document_id)

            heading = (h.heading or "").strip()
            page = h.page
            if heading:
                label = f"{name}:{heading}"
            elif page is not None:
                label = f"{name}:p{page}"
            else:
                label = name
            text = (h.text or "").strip()
            blocks.append(f"[{label}]\n{text}")

        context_block = "\n\n".join(blocks)

        if not fitted_history:
            history_block = "No prior conversation."
        else:
            history_block = "\n".join(
                f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')}"
                for m in fitted_history
            )

        system_content = PromptManager.render(
            "rag_answer.j2",
            context_block=context_block,
            history_block=history_block,
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": clean},
        ]

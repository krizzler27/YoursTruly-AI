"""RAG read path + document lifecycle (ingest writes live in IngestService)."""

import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from db.models import DocumentsModel
from core.logging import get_logger
from repository.document_repository import DocumentRepository
from repository.lance_repository import LanceRepository, sid
from services.llama_engine import EmbeddingEngine, get_default_ctx
from services.llm_service import LLMService
from services.prompt_manager import PromptManager

logger = get_logger(__name__)

CHARS_PER_TOKEN = 4  # heuristic until engine-side token counting lands
ANSWER_RESERVE_TOKENS = 512
HISTORY_CHAR_CAP = 3000


def rag_context_tokens(n_ctx: Optional[int] = None) -> int:
    """Retrieved-context budget derived from the chat window, never fixed."""
    window = n_ctx or get_default_ctx()
    return max(256, window - ANSWER_RESERVE_TOKENS - HISTORY_CHAR_CAP // CHARS_PER_TOKEN)


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

    def search(
        self, query: str, top_k: int = 5, conversation_id: Optional[uuid.UUID] = None
    ) -> List[Dict[str, Any]]:
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
    ) -> List[Any]:
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
        hits: Optional[List[Dict[str, Any]]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_context_tokens: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """One builder: plain chat without hits, grounded prompt with hits."""
        clean = (query or "").strip()
        if not hits:
            return LLMService.build_chat_messages(history, clean)

        logger.debug("build grounded hits=%s", len(hits))
        budget = (max_context_tokens or rag_context_tokens()) * CHARS_PER_TOKEN
        blocks = []
        used = 0

        ordered = sorted(hits, key=lambda x: x.get("score", 0.0), reverse=True)
        names = self._filenames(h.get("document_id") for h in ordered)

        for h in ordered:
            name = names.get(str(h.get("document_id", "")), str(h.get("document_id", "")))

            heading = (h.get("heading") or "").strip()
            page = h.get("page")
            if heading:
                label = f"{name}:{heading}"
            elif page is not None:
                label = f"{name}:p{page}"
            else:
                label = name
            text = (h.get("text") or "").strip()
            block = f"[{label}]\n{text}"
            room = budget - used
            if room <= 0:
                break
            blocks.append(block[:room])
            used += len(blocks[-1])
            if len(block) > room:
                break  # straddler truncated, budget spent

        context_block = "\n\n".join(blocks)

        if not history:
            history_block = "No prior conversation."
        else:
            history_block = "\n".join(
                f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')}"
                for m in history
            )[:HISTORY_CHAR_CAP]

        system_content = PromptManager.render(
            "rag_answer.j2",
            context_block=context_block,
            history_block=history_block,
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": clean},
        ]

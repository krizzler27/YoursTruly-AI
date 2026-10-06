"""RAG read path + document lifecycle (ingest writes live in IngestService)."""

import uuid
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from config import config
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
    top_k_for_ctx,
    truncate_text,
    rag_context_tokens as _derived_rag_tokens,
)
from services.laya_service import HIT_GRADE_QUESTION, LayaService
from services.llm_service import LLMService
from services.prompt_manager import PromptManager

logger = get_logger(__name__)


def rag_context_tokens(n_ctx: Optional[int] = None) -> int:
    """Retrieved-context budget derived from the chat window, never fixed."""
    return _derived_rag_tokens(n_ctx)


def _fit_memory(mem: str, mem_cap: int) -> Tuple[str, int]:
    """Truncate the memory block to mem_cap, report used tokens."""
    if not mem or mem_cap <= 0:
        return ("", 0)
    cut = truncate_text(mem, mem_cap)
    return (cut, count_tokens(cut))


def _parse_hit_level(answers: Dict) -> Tuple[int, float]:
    """Argmax relevance level plus top-probability confidence.

    Shape mirrors the smoke-test check in
    backend/notebooks/laya_dataset/test_finetuned.py: answers holds
    hit_grade probabilities keyed by level ("0"/"1"/"2"). Raises on any
    malformed shape so the caller fails open to fused order.
    """
    ans = (answers or {}).get("hit_grade") or {}
    probs = ans.get("probabilities")
    if isinstance(probs, list):
        pairs = list(enumerate(float(p) for p in probs))
    elif isinstance(probs, dict):
        pairs = [(int(k), float(v)) for k, v in probs.items()]
    else:
        raise ValueError(f"hit grade probs shape {type(probs).__name__}")
    if {lv for lv, _ in pairs} != {0, 1, 2}:
        raise ValueError(f"hit grade levels {[lv for lv, _ in pairs]}")
    level, conf = max(pairs, key=lambda kv: kv[1])
    return int(level), float(conf)


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
        if not hits:
            return hits
        try:
            return self._grade_hits(clean, hits)
        except Exception as e:
            logger.debug("hit grading skipped, fused order kept: %s", e)
            return hits

    def _grade_hits(self, query: str, hits: List[SearchHit]) -> List[SearchHit]:
        """Reorder fused hits by Laya relevance, drop confident irrelevants.

        exact(2) first, then partial(1), then irrelevant(0), stable by
        fused score within a level. Level-0 drops only when a level>=1
        exists, so grading alone never empties the list. Below
        LAYA_HIT_THRESHOLD the head is advisory: fused order kept, grades
        logged. Truncates to top_k_for_ctx. Raises on any Laya failure so
        search() fails open to the fused list.
        """
        names = self._filenames(h.document_id for h in hits)
        states = []
        for h in hits:
            filename = names.get(h.document_id, h.document_id)
            states.append(
                {
                    "query": query,
                    "chunk": f"[{filename}] {(h.text or '')}",
                    "filename": filename,
                }
            )
        svc = LayaService.get_instance()
        questions = {"hit_grade": HIT_GRADE_QUESTION}
        batch = getattr(svc, "predict_many", None)
        if callable(batch):
            results = batch(states, questions)
        else:
            results = [svc.predict(s, questions) for s in states]
        if len(results) != len(hits):
            raise ValueError(
                f"hit grading count {len(results)} != hits {len(hits)}"
            )
        graded = []
        for h, res in zip(hits, results):
            answers = (res or {}).get("answers") or {}
            level, conf = _parse_hit_level(answers)
            graded.append((level, conf, h))
        best = max((c for _, c, _ in graded), default=0.0)
        if best < float(config.LAYA_HIT_THRESHOLD):
            logger.info("hit grading advisory best_conf=%.2f hits=%s", best, len(hits))
            return list(hits)
        ordered = sorted(graded, key=lambda g: g[0], reverse=True)
        if any(lv >= 1 for lv, _, _ in ordered):
            ordered = [g for g in ordered if g[0] >= 1]
        out = [h for _, _, h in ordered[: top_k_for_ctx()]]
        logger.info(
            "hit grading done hits=%s kept=%s best_conf=%.2f",
            len(hits),
            len(out),
            best,
        )
        return out

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
        memory_text: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """One builder: plain chat without hits, grounded prompt with hits."""
        clean = (query or "").strip()
        mem = (memory_text or "").strip()
        if not hits:
            budget = allocate(
                route="DIRECT",
                needs_memory=bool(mem),
                query_tokens=count_tokens(clean),
            )
            fitted_history, truncated, hist_used = fit_history(
                history or [], budget["history_cap"]
            )
            mem_block, mem_used = _fit_memory(mem, budget["mem_cap"])
            logger.info(
                "budget total=%s usable=%s history=%s rag=%s mem=%s mem_used=%s route=%s overflow=%s",
                budget["total"],
                budget["usable"],
                hist_used,
                0,
                budget["mem_cap"],
                mem_used,
                "DIRECT",
                truncated,
            )
            return LLMService.build_chat_messages(
                fitted_history, clean, memory_text=mem_block or None
            )

        logger.debug("build grounded hits=%s", len(hits))
        budget = allocate(
            route="RAG", needs_memory=bool(mem), query_tokens=count_tokens(clean)
        )
        rag_cap = max_context_tokens or budget["rag_cap"]
        fitted, hits_truncated, rag_used = fit_hits(hits, rag_cap)
        fitted_history, hist_truncated, hist_used = fit_history(
            history or [], budget["history_cap"]
        )
        mem_block, mem_used = _fit_memory(mem, budget["mem_cap"])
        overflow = bool(hits_truncated or hist_truncated)
        logger.info(
            "budget total=%s usable=%s history=%s rag=%s mem=%s mem_used=%s route=%s overflow=%s",
            budget["total"],
            budget["usable"],
            hist_used,
            rag_used,
            budget["mem_cap"],
            mem_used,
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
            memory_block=mem_block,
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": clean},
        ]

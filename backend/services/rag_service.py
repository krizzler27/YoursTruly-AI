"""RAG read path + document lifecycle.

Ingest writes stay in IngestService (single-commit atomicity).
This service orchestrates: embed query → hybrid search, plus
document list/delete with store cleanup.
"""

import gc
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from db.db_engine import SessionLocal  # worker threads need own sessions
from db.models import DocumentChunksModel, DocumentsModel
from repository.chat_repository import ChatRepository
from repository.document_repository import DocumentRepository
from repository.lance_repository import LanceRepository
from services.ingest_service import IngestService, stored_upload_path
from services.llama_engine import EmbeddingEngine, LlamaEngine
from services.llm_service import LLMService
from services.prompt_manager import PromptManager


_INGEST_LOCK = threading.Lock()  # one background ingest at a time
SUMMARY_MAX_CHARS = 200
SUMMARY_SOURCE_CHARS = 3000
CHARS_PER_TOKEN = 4  # heuristic until engine-side token counting lands
RAG_CONTEXT_TOKENS = 5000  # configured ceiling; UI selector later


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

        vectors = self.engine.embed([clean])

        if not vectors or not vectors[0]:
            raise RuntimeError("embed returned no vector for query")

        doc_ids = None
        if conversation_id is not None:
            doc_ids = [str(d.id) for d in self.docs.list_by_conversation(conversation_id, limit=1000)]
            if not doc_ids:
                return []

        return self.lance.hybrid_search(
            clean, vectors[0], top_k=top_k, conversation_id=conversation_id, doc_ids=doc_ids
        )

    def ingest_file(self, source_path: str, filename: str) -> DocumentsModel:
        """Index one uploaded file via the Phase 1 atomic ingest path."""
        svc = IngestService(db=self.db, engine=self.engine, lance=self.lance)

        return svc.ingest(source_path, filename)

    def list_documents(
        self, limit: int = 100, conversation_id: Optional[uuid.UUID] = None
    ) -> List[DocumentsModel]:
        """List ingested files newest first, optionally one chat's."""
        if conversation_id is None:
            return self.docs.list_recent(limit=limit)
        return self.docs.list_by_conversation(conversation_id, limit=limit)

    def delete_document(self, document_id: uuid.UUID) -> uuid.UUID:
        """Remove a file row plus its vectors, FTS entries, and chunks."""
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

    def build_messages(
        self,
        query: str,
        hits: Optional[List[Dict[str, Any]]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_context_tokens: int = RAG_CONTEXT_TOKENS,
    ) -> List[Dict[str, str]]:
        """One builder: plain chat without hits, grounded prompt with hits."""
        clean = (query or "").strip()
        if not hits:
            return LLMService.build_chat_messages(history, clean)

        budget = max_context_tokens * CHARS_PER_TOKEN
        blocks = []
        used = 0

        for h in sorted(hits, key=lambda x: x.get("score", 0.0), reverse=True):
            try:
                name = self.docs.get_by_id(
                    uuid.UUID(str(h.get("document_id")))
                ).filename
            except Exception:
                name = str(h.get("document_id", ""))

            heading = (h.get("heading") or "").strip()
            text = (h.get("text") or "").strip()
            block = f"[{name}:{heading}]\n{text}"
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
            )

        system_content = PromptManager.render(
            "rag_answer.j2",
            context_block=context_block,
            history_block=history_block,
            current_query=clean,
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": clean},
        ]

    def delete_document_by_name(self, filename: str, conversation_id: uuid.UUID) -> None:
        """Drop the same-name doc in this chat first (re-ingest replaces)."""
        row = (
            self.db.query(DocumentsModel)
            .filter(
                DocumentsModel.filename == filename,
                DocumentsModel.conversation_id == conversation_id,
            )
            .first()
        )
        if row is None:
            raise ValueError(f"Document {filename} not found")
        self.delete_document(row.id)

    def submit_ingest(
        self, source_path: str, filename: str, conversation_id: uuid.UUID
    ) -> DocumentsModel:
        """Validate, insert a pending row, start the background worker (202 path)."""
        if not _INGEST_LOCK.acquire(blocking=False):
            raise RuntimeError("ingest already running, try again shortly")

        try:
            if ChatRepository(self.db).get_by_id(conversation_id) is None:
                raise ValueError(f"Conversation {conversation_id} not found")
            try:
                self.delete_document_by_name(filename, conversation_id)
            except ValueError:
                pass  # first upload of this name

            doc = DocumentsModel(
                filename=filename,
                chunk_count=0,
                status="pending",
                conversation_id=conversation_id,
                summary="",
            )
            self.db.add(doc)
            self.db.commit()
            self.db.refresh(doc)

        except Exception:
            _INGEST_LOCK.release()
            raise

        worker = threading.Thread(
            target=self._run_ingest_job,
            args=(str(doc.id), source_path, filename, str(conversation_id)),
            daemon=True,
        )

        worker.start()

        return doc

    def _run_ingest_job(
        self, doc_id: str, source_path: str, filename: str, conversation_id: str
    ) -> None:
        """Worker thread: chunk/embed/store, then SLM summary; lock released here."""
        tmp = Path(source_path)
        db2 = SessionLocal()
        ok = False

        try:
            if not self._wait_for_idle(timeout_s=60):
                print(f"[INGEST] chat busy too long, leaving {filename} pending")
                return

            target = uuid.UUID(doc_id)
            pending = db2.query(DocumentsModel).filter(DocumentsModel.id == target).first()

            if pending is None:
                return  # deleted while queued
            db2.delete(pending)
            db2.commit()  # ingest() below rebuilds the final row atomically

            engine2 = EmbeddingEngine()
            try:
                doc = IngestService(db=db2, engine=engine2, lance=LanceRepository(db2)).ingest(
                    source_path, filename, uuid.UUID(conversation_id)
                )
            finally:
                engine2.unload()

            self._write_summary(db2, doc)
            ok = True  # owned copy stays; failures below remove it

        except Exception as e:
            db2.rollback()
            print(f"[INGEST] {filename} failed: {e}")

        finally:
            if not ok:
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass
            db2.close()
            _INGEST_LOCK.release()

    @staticmethod
    def _wait_for_idle(timeout_s: int = 120) -> bool:
        """Hold the worker until chat stops generating (protects the 8GB box)."""
        waited = 0
        while LlamaEngine.get_instance().is_generating():
            if waited >= timeout_s:
                return False
            time.sleep(2)
            waited += 2
        return True

    @staticmethod
    def _write_summary(db: Session, doc: DocumentsModel) -> None:
        """One short SLM pass over the head of the doc; extractive fallback."""
        chunks = (
            db.query(DocumentChunksModel)
            .filter(DocumentChunksModel.document_id == doc.id)
            .order_by(DocumentChunksModel.index)
            .all()
        )
        head = "\n".join(c.text for c in chunks)[:SUMMARY_SOURCE_CHARS]
        summary = ""
        if head.strip():
            try:
                text = LLMService().invoke(
                    [
                        {
                            "role": "user",
                            "content": PromptManager.render("doc_summary.j2", doc_text=head),
                        }
                    ],
                    max_tokens=128,
                    temperature=0.2,
                )
                summary = (text or "").strip().replace("\n", " ")[:SUMMARY_MAX_CHARS]
            except Exception as e:
                print(f"[INGEST] summary SLM failed, extractive fallback: {e}")

        if not summary:
            heads = [c.heading for c in chunks if (c.heading or "").strip()]
            basis = " / ".join(dict.fromkeys(heads)) or head[:SUMMARY_MAX_CHARS]
            summary = f"{doc.filename} — {basis}"[:SUMMARY_MAX_CHARS]

        doc.summary = summary
        db.commit()

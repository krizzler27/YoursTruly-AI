"""Background ingest queue: submit fast, index serially, report status.

Flow: router writes upload to a temp path -> submit_ingest inserts one
`pending` row and enqueues -> single daemon worker takes jobs FIFO, waits
for chat idle (one model resident on 8GB), indexes via IngestService,
moves temp to the canonical copy, marks `indexed`/`failed`.
"""

import threading
import time
from pathlib import Path
from queue import Queue
from uuid import UUID

from sqlalchemy.orm import Session

from db.db_engine import SessionLocal  # worker thread boundary: own session
from db.models import DocumentChunksModel, DocumentsModel
from repository.chat_repository import ChatRepository
from repository.document_repository import DocumentRepository
from repository.lance_repository import LanceRepository
from services.llama_engine import EmbeddingEngine, LlamaEngine

SUMMARY_MAX_CHARS = 200
SUMMARY_SOURCE_CHARS = 3000
IDLE_TIMEOUT_S = 60

_jobs: Queue = Queue()
_worker_lock = threading.Lock()
_worker_started = False


def submit_ingest(
    db: Session, source_path: str, filename: str, conversation_id: UUID
) -> DocumentsModel:
    """Insert one `pending` row and enqueue; never blocks on the engine."""

    if ChatRepository(db).get_by_id(conversation_id) is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    _drop_same_name(db, filename, conversation_id)

    doc = DocumentsModel(
        filename=filename,
        chunk_count=0,
        status="pending",
        conversation_id=conversation_id,
        summary="",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    _jobs.put((str(doc.id), source_path, filename, str(conversation_id)))
    _ensure_worker()
    return doc


def _drop_same_name(db: Session, filename: str, conversation_id: UUID) -> None:
    """Remove a stale same-name row so the pending insert keeps uniqueness.

    Index refs go now; the old file copy stays until the new upload succeeds.
    """
    docs = DocumentRepository(db)
    row = (
        db.query(DocumentsModel)
        .filter(
            DocumentsModel.filename == filename,
            DocumentsModel.conversation_id == conversation_id,
        )
        .first()
    )
    if row is None:
        return
    LanceRepository(db).delete_document(row.id, commit=False)
    docs.delete(row.id)


def _ensure_worker() -> None:
    """Start the single FIFO worker once per process."""
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        threading.Thread(target=_worker_loop, daemon=True).start()
        _worker_started = True


def _worker_loop() -> None:
    while True:
        job = _jobs.get()
        try:
            _process(job)
        except Exception as e:
            print(f"[INGEST] worker failed: {e}")
        finally:
            _jobs.task_done()


def _process(job: tuple) -> None:
    doc_id, source_path, filename, conversation_id = job
    tmp = Path(source_path)
    db = SessionLocal()

    try:
        if not _wait_for_idle(timeout_s=IDLE_TIMEOUT_S):
            _mark_failed(db, UUID(doc_id), filename, UUID(conversation_id),
                         "chat busy too long, re-upload to retry")
            return

        target = UUID(doc_id)
        pending = db.query(DocumentsModel).filter(DocumentsModel.id == target).first()

        if pending is None:
            return  # superseded while queued
        conv_id = pending.conversation_id
        # Same row rides the whole job: pending -> indexing -> indexed/failed.
        # IngestService.ingest() below swaps it for the final row atomically.
        pending.status = "indexing"
        db.commit()

        from services.ingest_service import IngestService, stored_upload_path

        engine = EmbeddingEngine()
        try:
            doc = IngestService(
                db=db, engine=engine, lance=LanceRepository(db)
            ).ingest(source_path, filename, conv_id)
        finally:
            engine.unload()

        try:
            tmp.replace(stored_upload_path(conv_id, filename))
        except Exception:
            pass

        _write_summary(db, doc)

    except Exception as e:
        db.rollback()
        print(f"[INGEST] {filename} failed: {e}")
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        _mark_failed(db, UUID(doc_id), filename, UUID(conversation_id), str(e)[:200])

    finally:
        db.close()


def _mark_failed(
    db: Session, doc_id: UUID, filename: str, conversation_id: UUID, error: str
) -> None:
    try:
        db.rollback()
        row = (
            db.query(DocumentsModel)
            .filter(
                DocumentsModel.filename == filename,
                DocumentsModel.conversation_id == conversation_id,
            )
            .first()
        )
        if row is None:
            db.add(
                DocumentsModel(
                    filename=filename,
                    chunk_count=0,
                    status="failed",
                    conversation_id=conversation_id,
                    summary=error,
                )
            )
        else:
            row.status = "failed"
            row.summary = error
        db.commit()
    except Exception as e:
        print(f"[INGEST] failed marker lost: {e}")


def _wait_for_idle(timeout_s: int = IDLE_TIMEOUT_S) -> bool:
    """Hold the worker until chat stops generating (protects the 8GB box)."""
    waited = 0
    while LlamaEngine.get_instance().is_generating():
        if waited >= timeout_s:
            return False
        time.sleep(2)
        waited += 2
    return True


def _write_summary(db: Session, doc: DocumentsModel) -> None:
    """One short SLM pass over the head of the doc; extractive fallback."""
    from services.llm_service import LLMService
    from services.prompt_manager import PromptManager

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
                [{"role": "user",
                  "content": PromptManager.render("doc_summary.j2", doc_text=head)}],
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


def queue_depth() -> int:
    """Pending jobs; lets the router report position without new state."""
    return _jobs.qsize()

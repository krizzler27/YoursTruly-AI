"""Background ingest queue: submit fast, index serially, report status.

Flow: router writes upload to a temp path -> submit_ingest inserts one
`pending` row under a temp name and enqueues -> single daemon worker
takes jobs FIFO, waits for chat idle (one model resident on 8GB),
indexes via IngestService (which atomically swaps any live same-name
row for the new one), moves temp to the canonical copy, marks
`indexed`/`failed`. The previously indexed version stays live until
the replacement commits; a failed replacement leaves it untouched.
Temp `__pending__` rows are internal and hidden from listings.
"""

import threading
import time
from pathlib import Path
from queue import Queue
from uuid import UUID

from sqlalchemy.orm import Session

from db.db_engine import SessionLocal  # worker thread boundary: own session
from core.logging import get_logger, get_request_id, set_conversation_id, set_request_id
from db.models import DocumentChunksModel, DocumentsModel
from repository.chat_repository import ChatRepository
from repository.document_repository import DocumentRepository, PENDING_PREFIX
from repository.lance_repository import LanceRepository
from services.llama_engine import EmbeddingEngine, LlamaEngine

logger = get_logger(__name__)

SUMMARY_MAX_CHARS = 200
SUMMARY_SOURCE_CHARS = 3000
IDLE_TIMEOUT_S = 60

_jobs: Queue = Queue()
_worker_lock = threading.Lock()
_worker_started = False


def submit_ingest(
    db: Session, source_path: str, filename: str, conversation_id: UUID
) -> DocumentsModel:
    """Insert one `pending` row and enqueue; never blocks on the engine.

    The row uses a temp name so the live same-name version (if any) keeps
    serving until the replacement commits. Track the row by id, not name.
    """

    if ChatRepository(db).get_by_id(conversation_id) is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    _drop_stale(db, filename, conversation_id)

    import uuid6

    doc = DocumentsModel(
        filename=f"{PENDING_PREFIX}{uuid6.uuid7().hex}",
        chunk_count=0,
        status="pending",
        conversation_id=conversation_id,
        summary="",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    _jobs.put((str(doc.id), source_path, filename, str(conversation_id), get_request_id()))
    _ensure_worker()
    return doc


def _drop_stale(db: Session, filename: str, conversation_id: UUID) -> None:
    """Remove same-name rows that can never go live (pending/indexing/failed).

    A live `indexed` row is left alone: the worker's atomic swap replaces
    it only after the new index commits.
    """
    docs = DocumentRepository(db)
    stale = (
        db.query(DocumentsModel)
        .filter(
            DocumentsModel.filename == filename,
            DocumentsModel.conversation_id == conversation_id,
            DocumentsModel.status != "indexed",
        )
        .all()
    )
    for row in stale:
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
            logger.warning("worker failed: %s", e)
        finally:
            _jobs.task_done()


def _drop_tmp(tmp: Path) -> None:
    """Temp uploads die with their job; the canonical copy is separate."""
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass


def _process(job: tuple) -> None:
    doc_id, source_path, filename, conversation_id, request_id = job
    set_request_id(request_id)
    set_conversation_id(conversation_id)
    tmp = Path(source_path)
    db = SessionLocal()

    try:
        if not _wait_for_idle(timeout_s=IDLE_TIMEOUT_S):
            _mark_failed(db, UUID(doc_id), filename, UUID(conversation_id),
                         "chat busy too long, re-upload to retry")
            _drop_tmp(tmp)
            return

        target = UUID(doc_id)
        pending = db.query(DocumentsModel).filter(DocumentsModel.id == target).first()

        if pending is None:
            _drop_tmp(tmp)
            return  # superseded while queued; nothing references this file
        conv_id = pending.conversation_id
        # Temp row marks progress only. IngestService.ingest() below writes
        # the final row under the real filename and atomically deletes the
        # old live same-name row, if any. The temp row's name never matches,
        # so it survives the swap and is removed explicitly by id below.
        pending.status = "indexing"
        db.commit()

        from services.ingest_service import IngestService, stored_upload_path

        engine = EmbeddingEngine()
        try:
            doc = IngestService(
                db=db, engine=engine, lance=LanceRepository(db)
            ).ingest(source_path, filename, conv_id)
            db.query(DocumentsModel).filter(
                DocumentsModel.id == target).delete(synchronize_session=False)
            db.commit()
        finally:
            engine.unload()

        try:
            tmp.replace(stored_upload_path(conv_id, filename))
        except Exception:
            pass

        _write_summary(db, doc)

    except Exception as e:
        db.rollback()
        logger.warning("%s failed: %s", filename, e)
        _drop_tmp(tmp)
        _mark_failed(db, UUID(doc_id), filename, UUID(conversation_id), str(e)[:200])

    finally:
        db.close()


def _mark_failed(
    db: Session, doc_id: UUID, filename: str, conversation_id: UUID, error: str
) -> None:
    """Mark this job failed without touching a live same-name version.

    The job's temp row is found by id first. If a live `indexed` row with
    the real name exists (failed replacement), the temp row is dropped and
    the old version keeps serving. Otherwise the temp row takes the real
    name as a visible `failed` row.
    """
    try:
        db.rollback()
        row = db.query(DocumentsModel).filter(DocumentsModel.id == doc_id).first()
        if row is None:
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
                db.commit()
                return
        live = (
            db.query(DocumentsModel)
            .filter(
                DocumentsModel.filename == filename,
                DocumentsModel.conversation_id == conversation_id,
                DocumentsModel.status == "indexed",
                DocumentsModel.id != row.id,
            )
            .first()
        )
        if live is not None:
            db.delete(row)
        else:
            row.filename = filename
            row.status = "failed"
            row.summary = error
        db.commit()
    except Exception as e:
        logger.warning("failed marker lost: %s", e)


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
            logger.warning("summary SLM failed, extractive fallback: %s", e)

    if not summary:
        heads = [c.heading for c in chunks if (c.heading or "").strip()]
        basis = " / ".join(dict.fromkeys(heads)) or head[:SUMMARY_MAX_CHARS]
        summary = f"{doc.filename} — {basis}"[:SUMMARY_MAX_CHARS]

    doc.summary = summary
    db.commit()


def queue_depth() -> int:
    """Jobs ahead of the just-enqueued one; never negative."""
    return max(0, _jobs.qsize() - 1)

import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from db.db_engine import get_db
from core.logging import get_logger, get_request_id, set_conversation_id
from schemas.rag_schemas import DocumentResponse, SearchHit, SearchRequest
from services.ingest_job import ingest_job
from services.ingest_service import SUPPORTED_SUFFIXES, staged_upload_path
from services.llama_engine import EmbeddingEngine
from services.rag_service import RagService

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["RAG"])


@router.post("/search", response_model=List[SearchHit])
def search(req: SearchRequest, db: Session = Depends(get_db)):
    """Hybrid search - sync def runs in threadpool, embed is blocking."""
    engine = EmbeddingEngine.get_instance("embed")
    try:
        svc = RagService(db, engine=engine)
        return svc.search(req.query, top_k=req.top_k, conversation_id=req.conversation_id)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    except FileNotFoundError as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "type": type(e).__name__, "request_id": get_request_id(), "hint": "Retry, or quote request_id when reporting"},
        )
    finally:
        engine.unload()


@router.post("/ingest", response_model=DocumentResponse, status_code=status.HTTP_202_ACCEPTED)
def ingest(
    file: UploadFile = File(...),
    conversation_id: uuid.UUID = Form(...),
    db: Session = Depends(get_db),
):
    """Accept txt/md/pdf for one chat; queued, indexed in background FIFO."""
    set_conversation_id(conversation_id)
    raw_name = file.filename or ""

    filename = Path(raw_name).name.strip()

    if not filename:
        return JSONResponse(status_code=400, content={"detail": "filename missing"})

    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_SUFFIXES:
        return JSONResponse(
            status_code=400, content={"detail": f"unsupported file type: {suffix}"}
        )

    # Temp copy: the canonical file is replaced only after a good index.
    tmp_path = str(staged_upload_path(conversation_id, filename))

    try:
        with open(tmp_path, "wb") as out:
            shutil.copyfileobj(file.file, out)
    except Exception as e:
        Path(tmp_path).unlink(missing_ok=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "type": type(e).__name__, "request_id": get_request_id(), "hint": "Retry, or quote request_id when reporting"},
        )

    try:
        doc = ingest_job.submit(db, tmp_path, filename, conversation_id)
        logger.info("ingest accepted file=%s", filename)

        return JSONResponse(
            status_code=202,
            content=DocumentResponse.model_validate(doc).model_dump(mode="json"),
        )
    except ValueError as e:
        Path(tmp_path).unlink(missing_ok=True)
        msg = str(e)
        if "not found" in msg:
            return JSONResponse(status_code=404, content={"detail": msg})
        return JSONResponse(status_code=400, content={"detail": msg})
    except Exception as e:
        Path(tmp_path).unlink(missing_ok=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "type": type(e).__name__, "request_id": get_request_id(), "hint": "Retry, or quote request_id when reporting"},
        )


@router.get("/documents", response_model=List[DocumentResponse])
def list_documents(
    db: Session = Depends(get_db), conversation_id: Optional[uuid.UUID] = None
):
    """List ingested files newest first, optionally one chat's."""
    try:
        svc = RagService(db)

        return svc.list_documents(limit=100, conversation_id=conversation_id)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "type": type(e).__name__, "request_id": get_request_id(), "hint": "Retry, or quote request_id when reporting"},
        )


@router.delete("/documents/{document_id}", status_code=status.HTTP_200_OK)
def delete_document(document_id: uuid.UUID, db: Session = Depends(get_db)):
    """Delete a file row plus its vectors, FTS entries, and chunks."""
    try:
        svc = RagService(db)

        deleted = svc.delete_document(document_id)
        logger.info("document deleted id=%s", deleted)

        return JSONResponse(content={"status": "deleted", "id": str(deleted)})
    except ValueError as e:
        msg = str(e)

        if "not found" in msg:
            return JSONResponse(status_code=404, content={"detail": msg})

        return JSONResponse(status_code=400, content={"detail": msg})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "type": type(e).__name__, "request_id": get_request_id(), "hint": "Retry, or quote request_id when reporting"},
        )

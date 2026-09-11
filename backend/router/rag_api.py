import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from db.db_engine import get_db
from schemas.rag_schemas import DocumentResponse, SearchHit, SearchRequest
from services.ingest_service import SUPPORTED_SUFFIXES, stored_upload_path
from services.rag_service import RagService

router = APIRouter(prefix="/api", tags=["RAG"])


@router.post("/search", response_model=List[SearchHit])
def search(req: SearchRequest, db: Session = Depends(get_db)):
    """Hybrid search — sync def runs in threadpool, embed is blocking."""
    try:
        svc = RagService(db)

        return svc.search(req.query, top_k=req.top_k, conversation_id=req.conversation_id)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    except FileNotFoundError as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"Exception occured": str(e), "type": type(e).__name__},
        )


@router.post("/ingest", response_model=DocumentResponse, status_code=status.HTTP_202_ACCEPTED)
def ingest(
    file: UploadFile = File(...),
    conversation_id: uuid.UUID = Form(...),
    db: Session = Depends(get_db),
):
    """Accept txt/md/pdf for one chat; chunk/embed/summary run in background."""
    raw_name = file.filename or ""

    filename = Path(raw_name).name.strip()

    if not filename:
        return JSONResponse(status_code=400, content={"detail": "filename missing"})

    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_SUFFIXES:
        return JSONResponse(
            status_code=400, content={"detail": f"unsupported file type: {suffix}"}
        )

    tmp_path: str | None = None

    try:
        tmp_path = str(stored_upload_path(conversation_id, filename))

        with open(tmp_path, "wb") as out:
            shutil.copyfileobj(file.file, out)
    except Exception as e:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
        return JSONResponse(
            status_code=500,
            content={"Exception occured": str(e), "type": type(e).__name__},
        )

    try:
        svc = RagService(db)

        doc = svc.submit_ingest(tmp_path, filename, conversation_id)

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
    except RuntimeError as e:
        Path(tmp_path).unlink(missing_ok=True)
        msg = str(e)
        if "already running" in msg:
            return JSONResponse(status_code=409, content={"detail": msg})
        return JSONResponse(status_code=500, content={"detail": msg})
    except Exception as e:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
        return JSONResponse(
            status_code=500,
            content={"Exception occured": str(e), "type": type(e).__name__},
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
            content={"Exception occured": str(e), "type": type(e).__name__},
        )


@router.delete("/documents/{document_id}", status_code=status.HTTP_200_OK)
def delete_document(document_id: uuid.UUID, db: Session = Depends(get_db)):
    """Delete a file row plus its vectors, FTS entries, and chunks."""
    try:
        svc = RagService(db)

        deleted = svc.delete_document(document_id)

        return JSONResponse(content={"status": "deleted", "id": str(deleted)})
    except ValueError as e:
        msg = str(e)

        if "not found" in msg:
            return JSONResponse(status_code=404, content={"detail": msg})

        return JSONResponse(status_code=400, content={"detail": msg})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"Exception occured": str(e), "type": type(e).__name__},
        )

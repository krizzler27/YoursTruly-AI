from typing import Optional

from fastapi import APIRouter, Query, status
from fastapi.responses import JSONResponse

from pathlib import Path

from config import config
from core.logging import get_logger
from schemas.api_schemas import CatalogRequest, ModelDownloadRequest, QuantsRequest, RecommendRequest, SORT_ALIASES
from services.download_manager import DownloadManager
from services.llmfit_services import LLMFitServices

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["LLMFit"])


@router.get('/system', status_code=status.HTTP_200_OK)
def system_info():
    """System hardware info — sync def runs in threadpool, not blocking event loop."""
    try:
        svc = LLMFitServices()
        data = svc.system_info()
        return JSONResponse(content=data)
    except FileNotFoundError as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"Exception occured": str(e), "type": type(e).__name__})


@router.get('/catalog', status_code=status.HTTP_200_OK)
def catalog(
    limit: int = Query(20, ge=1, le=100),
    providers: Optional[str] = Query(None, description="comma-separated, e.g. qwen,meta; null=trusted, empty=trusted"),
    include_community: bool = Query(False),
    perfect_only: bool = Query(False),
    sort: str = Query("score", description="score|tps|mem (vram alias)"),
):
    """Catalog search with filters. Deprecated: use POST /api/catalog. Sync def → threadpool."""
    try:
        svc = LLMFitServices()
        # normalize sort via single source (schemas)
        sort = SORT_ALIASES.get(sort.strip().lower(), sort.strip().lower()) if sort else "score"
        prov_list: Optional[list[str]] = None
        if providers is not None:
            prov_list = [p.strip().lower() for p in providers.split(",") if p.strip()]
            if not prov_list:
                prov_list = None  # empty -> trusted
        data = svc.discover_and_classify(
            limit=limit,
            include_community=include_community,
            perfect_only=perfect_only,
            providers=prov_list,
            sort=sort,
        )
        return JSONResponse(content=data)
    except FileNotFoundError as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"Exception occured": str(e), "type": type(e).__name__})


@router.post('/catalog', status_code=status.HTTP_200_OK)
def catalog_post(req: CatalogRequest):
    """POST catalog - sync def runs in threadpool, catalog 15-30s won't block event loop."""
    try:
        svc = LLMFitServices()
        providers = req.providers
        if providers is not None and len(providers) == 0:
            providers = None
        data = svc.discover_and_classify(
            limit=req.limit,
            include_community=req.include_community,
            perfect_only=req.perfect_only,
            providers=providers,
            sort=req.sort,
        )
        return JSONResponse(content=data)
    except FileNotFoundError as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"Exception occured": str(e), "type": type(e).__name__})


@router.post('/recommend', status_code=status.HTTP_200_OK)
def recommend(req: RecommendRequest):
    """Recommend with use_case filter — sync def → threadpool."""
    try:
        svc = LLMFitServices()
        providers = req.providers
        if providers is not None and len(providers) == 0:
            providers = None
        data = svc.recommend(
            limit=req.limit,
            use_case=req.use_case,
            min_fit=req.min_fit,
            runtime=req.runtime,
            providers=providers,
            sort=req.sort,
            include_community=req.include_community,
            perfect_only=req.perfect_only,
        )
        return JSONResponse(content=data)
    except FileNotFoundError as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"Exception occured": str(e), "type": type(e).__name__})


@router.post('/models/quants', status_code=status.HTTP_200_OK)
def list_quants(req: QuantsRequest):
    """List available quants — sync def → threadpool."""
    try:
        svc = LLMFitServices()
        data = svc.list_remote_ggufs(req.model)
        return JSONResponse(content=data)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"detail": str(e)})
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"Exception occured": str(e), "type": type(e).__name__})


@router.get('/models', status_code=status.HTTP_200_OK)
def models():
    """List installed models — key=path, value=name for direct use without Path concat."""
    try:
        svc = LLMFitServices()
        infos = svc.list_local_info()  # each {name, path, size_gb, ...}
        if not infos:
            return JSONResponse(
                content={
                    "message": f"No installed models in {Path(config.LLAMA_MODEL_PATH)}. Download a model via Explore.",
                    "models": [],
                    "details": [],
                    "path": str(Path(config.LLAMA_MODEL_PATH)),
                }
            )
        # key-value: frontend uses path as key, name as value
        kv = [{"name": i["name"], "path": i["path"]} for i in infos]
        return JSONResponse(content={"models": kv, "details": infos, "path": str(Path(config.LLAMA_MODEL_PATH))})
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"Exception occured": str(e), "type": type(e).__name__}
        )


@router.post('/models/download', status_code=status.HTTP_200_OK)
async def download_model(req: ModelDownloadRequest):
    """Enqueue single download — 202 instantly, 409 if already busy (neutral)."""
    try:
        mgr = DownloadManager.get_instance()
        job = mgr.start(repo_id=req.repo_id, quant=req.quant)
        logger.info("download start repo=%s job=%s", req.repo_id, job["id"])
        return JSONResponse(status_code=202, content={"job_id": job["id"], "status": job["status"], "job": job})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"detail": str(e)})
    except RuntimeError as e:
        msg = str(e)
        if "already in progress" in msg or "already installed" in msg:
            return JSONResponse(status_code=409, content={"detail": msg})
        return JSONResponse(status_code=500, content={"detail": msg})
    except Exception as e:
        return JSONResponse(status_code=500, content={"Exception occured": str(e), "type": type(e).__name__})


@router.get('/downloads', status_code=status.HTTP_200_OK)
async def list_downloads():
    """List all download jobs (for dock polling)."""
    try:
        mgr = DownloadManager.get_instance()
        jobs = mgr.list_all()
        return JSONResponse(content={"jobs": jobs})
    except Exception as e:
        return JSONResponse(status_code=500, content={"Exception occured": str(e), "type": type(e).__name__})


@router.get('/downloads/{job_id}', status_code=status.HTTP_200_OK)
async def get_download(job_id: str):
    """Poll single download job progress."""
    try:
        mgr = DownloadManager.get_instance()
        job = mgr.get(job_id)
        if not job:
            return JSONResponse(status_code=404, content={"detail": f"Job not found: {job_id}"})
        return JSONResponse(content=job)
    except Exception as e:
        return JSONResponse(status_code=500, content={"Exception occured": str(e), "type": type(e).__name__})


@router.delete('/downloads/{job_id}', status_code=status.HTTP_200_OK)
async def cancel_download(job_id: str):
    """Cancel a queued/downloading job."""
    try:
        mgr = DownloadManager.get_instance()
        job = mgr.get(job_id)
        if not job:
            return JSONResponse(status_code=404, content={"detail": f"Job not found: {job_id}"})
        ok = mgr.cancel(job_id)
        if not ok:
            return JSONResponse(status_code=400, content={"detail": "Cannot cancel completed/failed job"})
        return JSONResponse(content={"status": "cancelled", "job_id": job_id})
    except Exception as e:
        return JSONResponse(status_code=500, content={"Exception occured": str(e), "type": type(e).__name__})


@router.delete('/models/{filename}', status_code=status.HTTP_200_OK)
async def delete_model(filename: str):
    """Delete local model."""
    try:
        svc = LLMFitServices()
        res = svc.delete_local(filename)
        return JSONResponse(content=res)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"detail": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"Exception occured": str(e), "type": type(e).__name__})
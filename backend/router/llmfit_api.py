from typing import Optional

from fastapi import APIRouter, Query, status
from fastapi.responses import JSONResponse

from pathlib import Path

from config import config
from schemas.api_schemas import CatalogRequest, ModelDownloadRequest, QuantsRequest, RecommendRequest, SORT_ALIASES
from services.llama_service import LlamaEngine
from services.llmfit_services import LLMFitServices

router = APIRouter(prefix="/api", tags=["LLMFit"])


@router.get('/system', status_code=status.HTTP_200_OK)
async def system_info():
    """System hardware info."""
    try:
        svc = LLMFitServices()
        data = svc.system_info()
        return JSONResponse(content=data)
    except FileNotFoundError as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"Exception occured": str(e), "type": type(e).__name__})


@router.get('/catalog', status_code=status.HTTP_200_OK)
async def catalog(
    limit: int = Query(20, ge=1, le=100),
    providers: Optional[str] = Query(None, description="comma-separated, e.g. qwen,meta; null=trusted, empty=trusted"),
    include_community: bool = Query(False),
    perfect_only: bool = Query(False),
    sort: str = Query("score", description="score|tps|mem (vram alias)"),
):
    """Catalog search with filters. Deprecated: use POST /api/catalog."""
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
async def catalog_post(req: CatalogRequest):
    """POST catalog - clean body, defaults: trusted-only, perfect_only=True, sort=score."""
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
async def recommend(req: RecommendRequest):
    """Recommend with use_case filter (general|coding|reasoning|chat|multimodal)."""
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
async def list_quants(req: QuantsRequest):
    """List available quants for a model via `llmfit download <model> --list`."""
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
async def models():
    """List installed models."""
    try:
        engine = LlamaEngine.get_instance()
        files = engine.list_local_models()
        if not files:
            return JSONResponse(
                content={
                    "message": f"No installed models in {Path(config.LLAMA_MODEL_PATH)}. Place qwen2.5-3b-Q4_K_M.gguf there or trigger download.",
                    "models": [],
                    "path": str(Path(config.LLAMA_MODEL_PATH)),
                }
            )
        return JSONResponse(content={"models": files, "path": str(Path(config.LLAMA_MODEL_PATH))})
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"Exception occured": str(e), "type": type(e).__name__}
        )


@router.post('/models/download', status_code=status.HTTP_200_OK)
async def download_model(req: ModelDownloadRequest):
    """Download model."""
    try:
        svc = LLMFitServices()
        res = svc.download(repo_id=req.repo_id, filename=req.filename or "", quant=req.quant)
        return JSONResponse(content=res)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"detail": str(e)})
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
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
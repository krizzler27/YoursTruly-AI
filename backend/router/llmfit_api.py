from typing import Optional

from fastapi import APIRouter, Query, status
from fastapi.responses import JSONResponse

from schemas.api_schemas import ModelDownloadRequest
from services.llama_service import LlamaEngine, get_models_dir
from services.llmfit_services import LLMFitServices

router = APIRouter(prefix="/api", tags=["LLMFit"])


@router.get('/system', status_code=status.HTTP_200_OK)
async def system_info():
    """Hardware probe via llmfit CLI — for explore UI header."""
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
    providers: Optional[str] = Query(None, description="comma-separated, e.g. qwen,meta; empty = trusted only"),
    include_community: bool = Query(False),
    perfect_only: bool = Query(False),
    sort: str = Query("score", description="score|tps|vram"),
):
    """Browse + recommend with filters — your built catalog."""
    try:
        svc = LLMFitServices()
        prov_list: Optional[list[str]] = None
        if providers is not None:
            prov_list = [p.strip().lower() for p in providers.split(",") if p.strip()]
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
    except Exception as e:
        return JSONResponse(status_code=500, content={"Exception occured": str(e), "type": type(e).__name__})


@router.get('/models', status_code=status.HTTP_200_OK)
async def models():
    """List local GGUFs from ~/.yourstrulyai/models — used by chat model picker."""
    try:
        engine = LlamaEngine.get_instance()
        files = engine.list_local_models()
        if not files:
            return JSONResponse(
                content={
                    "message": f"No installed models in {get_models_dir()}. Place qwen2.5-3b-Q4_K_M.gguf there or trigger download.",
                    "models": [],
                    "path": str(get_models_dir()),
                }
            )
        return JSONResponse(content={"models": files, "path": str(get_models_dir())})
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"Exception occured": str(e), "type": type(e).__name__}
        )


@router.post('/models/download', status_code=status.HTTP_200_OK)
async def download_model(req: ModelDownloadRequest):
    """Via llmfit download — hardware-aware quant selection, cached then copied to ~/.yourstrulyai/models."""
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
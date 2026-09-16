from contextlib import asynccontextmanager
from pathlib import Path

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import FastAPI, status
from fastapi.staticfiles import StaticFiles

from config import config
from core.logging import get_logger, setup_logging
from core.middleware import RequestIdMiddleware
from router import chat_api, llmfit_api, rag_api
from db.models import Base
from db.db_engine import engine
from services.llama_engine import LlamaEngine

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(config.LOG_LEVEL)
    logger.info("startup")
    Base.metadata.create_all(bind=engine)

    try:
        LlamaEngine.get_instance().load()
        logger.info("Chat Model loaded")
    except Exception as e:
        logger.warning("model load skipped: %s", e)

    yield

    try:
        LlamaEngine.get_instance().unload()
    except Exception:
        pass
    logger.info("shutdown")


app = FastAPI(title='YoursTruly AI', lifespan=lifespan)

origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:8001",
    "http://127.0.0.1:8001",
]

app.add_middleware(RequestIdMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Conversation-Id", "X-TTFT", "X-Route", "X-Request-Id"],
)

app.include_router(chat_api.router)
app.include_router(llmfit_api.router)
app.include_router(rag_api.router)

@app.get("/api/health", status_code=status.HTTP_200_OK)
async def health() -> JSONResponse:
    try:
        h = LlamaEngine.get_instance().health()
        status_str = h.get("status", "error")
        is_healthy = status_str == "ready"
        return JSONResponse(content={
            "status": status_str,
            "model": h.get("model"),
            "path": h.get("path"),
            "exists": h.get("exists"),
            "loaded": h.get("loaded"),
            "generating": h.get("generating"),
            "available": h.get("available", []),
            "message": "Server is Healthy" if is_healthy else "No model installed - download via Explore",
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})


frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    setup_logging(config.LOG_LEVEL)
    uvicorn.run(app=app, host="127.0.0.1", port=8000, log_config=None)

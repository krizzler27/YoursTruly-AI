from contextlib import asynccontextmanager
from pathlib import Path

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import FastAPI, status
from fastapi.staticfiles import StaticFiles

from router import chat_api, llmfit_api
from db.models import Base
from db.db_engine import engine
from services.llama_service import LlamaEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # create tables on startup (SQLite, local-first)
    Base.metadata.create_all(bind=engine)
    # warmup llama engine — mmap fault + <1s TTFT
    try:
        eng = LlamaEngine.get_instance()
        eng.load()
    except Exception:
        # model missing or llama-cpp-python not installed — health will report error
        pass
    yield
    try:
        LlamaEngine.get_instance().unload()
    except Exception:
        pass


app = FastAPI(title='YoursTruly AI', lifespan=lifespan)

origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:8001",
    "http://127.0.0.1:8001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Conversation-Id", "X-TTFT"],
)

app.include_router(chat_api.router)
app.include_router(llmfit_api.router)

@app.get("/api/health", status_code=status.HTTP_200_OK)
async def health() -> JSONResponse:
    """Lightweight model health — process + file check, no daemon."""
    try:
        h = LlamaEngine.get_instance().health()
        # map to MODEL READY / OFFLINE contract for frontend
        status_str = h.get("status", "error")
        return JSONResponse(content={
            "status": status_str,  # ready | loading | error
            "model": h.get("model"),
            "path": h.get("path"),
            "exists": h.get("exists"),
            "loaded": h.get("loaded"),
            "generating": h.get("generating"),
            "message": "Server is Healthy" if status_str == "ready" else "Model not loaded",
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})


# frontend — serve vanilla JS + Oat at http://127.0.0.1:8000/
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app=app, host="127.0.0.1", port="8000")

from pathlib import Path

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import FastAPI, status
from fastapi.staticfiles import StaticFiles

from router import chat_api, llmfit_api
from db.models import Base
from db.db_engine import engine
from services.llmfit_services import LLMFitServices
from config import config

# create tables on startup (SQLite, local-first)
Base.metadata.create_all(bind=engine)

app = FastAPI(title='YoursTruly AI')

origins = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
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
def health() -> JSONResponse:
    return JSONResponse(content={"message":"Server is Healthy"})

@app.get("/api/health/ollama", status_code=status.HTTP_200_OK)
async def ollama_health() -> JSONResponse:
    """Fast Ollama liveness check (0.5s) for initial UI status."""
    try:
        running = await LLMFitServices().is_ollama_running(timeout=0.5)
        return JSONResponse(content={"running": running, "host": config.OLLAMA_HOST})
    except Exception as e:
        return JSONResponse(status_code=500, content={"running": False, "error": str(e), "host": config.OLLAMA_HOST})

# frontend — serve vanilla JS + Oat at http://127.0.0.1:8000/
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app=app, host="127.0.0.1", port="8000")

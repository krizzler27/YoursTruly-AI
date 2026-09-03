from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_model_path() -> str:
    return str(Path.home() / ".yourstrulyai" / "models" / "qwen2.5-3b-instruct-Q4_K_M.gguf")


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    LLAMA_MODEL_PATH: str = _default_model_path()
    LLAMA_N_CTX: Optional[int] = None  # auto: 2048 (8GB) / 4096 (≥12GB)
    LLAMA_N_THREADS: Optional[int] = None  # auto: psutil physical cores
    LLAMA_N_GPU_LAYERS: Optional[int] = None  # auto: -1 Vulkan else 0
    DATABASE_URL: str = "sqlite+pysqlite:///yourstrulyai.db"

config = Config()
from pathlib import Path
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_models_dir() -> str:
    return str(Path.home() / ".yourstrulyai" / "models")


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    LLAMA_MODEL_PATH: str = _default_models_dir()
    LLAMA_N_CTX: Optional[int] = None  # auto: 2048 (8GB) / 4096 (≥12GB)
    LLAMA_N_THREADS: Optional[int] = None  # auto: psutil physical cores
    LLAMA_N_GPU_LAYERS: Optional[int] = None  # auto: -1 Vulkan else 0
    DATABASE_URL: str = "sqlite+pysqlite:///yourstrulyai.db"
    LOG_LEVEL: str = "INFO"  # dev console verbosity; silent in frozen exe

    @field_validator("LLAMA_MODEL_PATH", mode="after")
    @classmethod
    def _ensure_models_dir(cls, v: str) -> str:
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

config = Config()



from pathlib import Path
from functools import lru_cache
from typing import Optional
import os
import sys

import psutil
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_models_dir() -> str:
    return str(Path.home() / ".yourstrulyai" / "models")


@lru_cache(maxsize=1)
def total_ram_gb() -> float:
    """Total RAM in GB, decided once per process."""
    total = psutil.virtual_memory().total
    return total / (1024**3) if total else 8.0


@lru_cache(maxsize=1)
def physical_cores() -> int:
    """Physical cores, decided once per process."""
    cores = psutil.cpu_count(logical=False)
    return int(cores) if cores else max(1, (os.cpu_count() or 4) // 2)


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    LLAMA_MODEL_PATH: str = _default_models_dir()
    LLAMA_N_CTX: Optional[int] = None  # auto: 4096 (<16GB) / 8192 (>=16GB)
    LLAMA_N_THREADS: Optional[int] = None  # auto: psutil physical cores
    LLAMA_N_GPU_LAYERS: Optional[int] = None  # auto: -1 Vulkan else 0
    DATABASE_URL: str = "sqlite+pysqlite:///yourstrulyai.db"
    LOG_LEVEL: str = "INFO"  # dev console verbosity; silent in frozen exe

    @property
    def IS_DEV(self) -> bool:
        """True from source, False in shipped exe; property so env can't spoof it."""
        return not getattr(sys, "frozen", False)

    @property
    def EFFECTIVE_N_CTX(self) -> int:
        """Resolved ctx: explicit override else RAM tier bins."""
        if self.LLAMA_N_CTX is not None:
            return int(self.LLAMA_N_CTX)
        return 8192 if total_ram_gb() >= 16.0 else 4096

    @property
    def EFFECTIVE_N_THREADS(self) -> int:
        """Resolved threads: explicit override else probed cores."""
        if self.LLAMA_N_THREADS is not None:
            return int(self.LLAMA_N_THREADS)
        return physical_cores()

    @field_validator("LLAMA_MODEL_PATH", mode="after")
    @classmethod
    def _ensure_models_dir(cls, v: str) -> str:
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

config = Config()



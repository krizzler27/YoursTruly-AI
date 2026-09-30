from pathlib import Path
from functools import lru_cache
from typing import List, Literal, Optional
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
    LLAMA_CHAT_MODEL: Optional[str] = None  # explicit chat GGUF, else finder picks most-recent
    LLAMA_EMBED_MODEL: Optional[str] = None  # explicit embed GGUF, else finder picks nomic
    LLAMA_WORKER_MODEL: Optional[str] = None  # explicit worker GGUF, else finder picks small qwen
    LLAMA_N_CTX: Optional[int] = None  # auto: 4096 (<16GB) / 8192 (>=16GB)
    LLAMA_N_THREADS: Optional[int] = None  # auto: psutil physical cores
    LLAMA_N_GPU_LAYERS: Optional[int] = None  # auto: -1 Vulkan else 0
    DATABASE_URL: str = "sqlite+pysqlite:///yourstrulyai.db"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"  # dev console verbosity; silent in frozen exe
    SAFETY_MARGIN: int = 256  # tokens held back from every window
    ANSWER_RESERVE_4K: int = 512  # decode headroom on the 4096 bin
    ANSWER_RESERVE_8K: int = 768  # decode headroom on the 8192 bin
    RAG_MAX_SHARE_4K: float = 0.7  # RAG fraction of usable on 4096
    RAG_MAX_SHARE_8K: float = 0.6  # RAG fraction of usable on 8192
    HISTORY_MAX_SHARE_4K: float = 0.9  # history fraction of usable on 4096
    HISTORY_MAX_SHARE_8K: float = 0.8  # history fraction of usable on 8192
    MEMORY_MAX_TOKENS_4K: int = 300  # memory carve-out on 4096
    MEMORY_MAX_TOKENS_8K: int = 500  # memory carve-out on 8192
    SUMMARY_TRIGGER: float = 0.85  # usable fraction that triggers episodic write
    SUMMARY_TIMEOUT_S: int = 30  # summary call budget before truncate fallback
    SUMMARY_MODEL_ROLE: str = "chat"  # summary slot until worker lands
    MIN_WORKER_RAM_GB: float = 14.0  # worker auto-route floor - true-16GB boxes report 14-15.9 after hardware reserve; 12GB-class stays out

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

    @property
    def EFFECTIVE_ANSWER_RESERVE(self) -> int:
        """Decode headroom for the active window bin."""
        return self.ANSWER_RESERVE_8K if self.EFFECTIVE_N_CTX >= 8192 else self.ANSWER_RESERVE_4K

    @property
    def EFFECTIVE_RAG_SHARE(self) -> float:
        """RAG fraction of usable tokens for the active bin."""
        return self.RAG_MAX_SHARE_8K if self.EFFECTIVE_N_CTX >= 8192 else self.RAG_MAX_SHARE_4K

    @property
    def EFFECTIVE_HISTORY_SHARE(self) -> float:
        """History fraction of usable tokens for the active bin."""
        return self.HISTORY_MAX_SHARE_8K if self.EFFECTIVE_N_CTX >= 8192 else self.HISTORY_MAX_SHARE_4K

    @property
    def EFFECTIVE_MEMORY_TOKENS(self) -> int:
        """Memory carve-out for the active window bin."""
        return self.MEMORY_MAX_TOKENS_8K if self.EFFECTIVE_N_CTX >= 8192 else self.MEMORY_MAX_TOKENS_4K

    @property
    def EFFECTIVE_CHAT_MODEL(self) -> str:
        """Resolved chat GGUF path, else a download hint error."""
        hits = list_models("chat", name=self.LLAMA_CHAT_MODEL)
        if not hits:
            raise FileNotFoundError(
                f"No chat GGUF in {self.LLAMA_MODEL_PATH}. Download a model via Explore."
            )
        return str(hits[0])

    @property
    def EFFECTIVE_EMBED_MODEL(self) -> str:
        """Resolved embed GGUF path, else a download hint error."""
        hits = list_models("embed", name=self.LLAMA_EMBED_MODEL)
        if not hits:
            raise FileNotFoundError(
                f"No nomic-embed-text GGUF in {self.LLAMA_MODEL_PATH}. "
                "Download e.g. nomic-ai/nomic-embed-text-v1.5-GGUF Q4_K_M."
            )
        return str(hits[0])

    @field_validator("LLAMA_MODEL_PATH", mode="after")
    @classmethod
    def _ensure_models_dir(cls, v: str) -> str:
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

config = Config()


def list_models(kind: str = "all", name: Optional[str] = None) -> List[Path]:
    """Installed GGUFs, most-recent first. kind filters chat/embed, name picks one exact model."""
    if kind not in ("all", "chat", "embed"):
        raise ValueError(f"Unknown model kind: {kind}")
    if name and name.strip():
        p = Path(name.strip())
        cand = p if p.is_file() else Path(config.LLAMA_MODEL_PATH) / p.name
        if cand.is_file() and (
            kind == "all"
            or ("nomic" in cand.name.lower()) == (kind == "embed")
        ):
            return [cand]
        return []
    try:
        found = sorted(
            [p for p in Path(config.LLAMA_MODEL_PATH).glob("*.gguf") if p.is_file()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return []
    if kind == "chat":
        return [p for p in found if "nomic" not in p.name.lower()]
    if kind == "embed":
        return [p for p in found if "nomic" in p.name.lower()]
    return found



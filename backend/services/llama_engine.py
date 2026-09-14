from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional
import gc
import os

import psutil

try:
    import llama_cpp
    from llama_cpp import Llama
except ImportError as e:
    raise RuntimeError("llama-cpp-python not installed.") from e

from config import config
from core.logging import get_logger

logger = get_logger(__name__)


def get_physical_cores() -> int:
    """Physical cores via psutil, fallback to logical//2."""
    cores = psutil.cpu_count(logical=False)
    return int(cores) if cores else max(1, (os.cpu_count() or 4) // 2)


def get_total_ram_gb() -> float:
    """Total RAM in GB via psutil, fallback 8.0 if undetermined."""
    total = psutil.virtual_memory().total
    return total / (1024**3) if total else 8.0


def get_default_ctx() -> int:
    if config.LLAMA_N_CTX is not None:
        return int(config.LLAMA_N_CTX)
    return 4096 if get_total_ram_gb() >= 12.0 else 2048


class LlamaEngine:
    """In-process llama.cpp lifecycle — load, switch, health, tenure guards."""

    _instance: Optional["LlamaEngine"] = None
    _lock: Lock = Lock()

    def __init__(self, model_path: Optional[str] = None):
        self.model_path: Optional[str] = model_path
        self.llm = None
        self._generating = False
        self._gen_lock = Lock()

    @staticmethod
    def _discover_models() -> List[Path]:
        """List installed GGUFs sorted by most recent mtime first."""
        models_dir = Path(config.LLAMA_MODEL_PATH)
        try:
            return sorted(
                [p for p in models_dir.glob("*.gguf") if p.is_file()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except Exception:
            return []

    def switch_model(self, full_path: Optional[str]) -> None:
        """Switch to GGUF at full path."""
        if not full_path or not full_path.strip():
            return
        if self.is_generating():
            logger.warning("switch rejected, busy")
            raise RuntimeError("System Busy — model is generating. Try again.")
        p = Path(full_path.strip())
        if not p.exists() or not p.is_file() or p.suffix.lower() != ".gguf":
            raise FileNotFoundError(f"Model not found: {full_path}")
        if (
            self.model_path
            and Path(self.model_path).resolve() == p.resolve()
            and self.is_loaded()
        ):
            return
        self.unload()
        self.model_path = str(p)
        self.load()

    @classmethod
    def get_instance(cls) -> "LlamaEngine":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def is_loaded(self) -> bool:
        return self.llm is not None

    def is_generating(self) -> bool:
        with self._gen_lock:
            return self._generating

    def _set_generating(self, value: bool) -> None:
        with self._gen_lock:
            self._generating = value

    def acquire(self) -> None:
        """Claim the engine or raise Busy."""
        if self.is_generating():
            logger.warning("acquire rejected, busy")
            raise RuntimeError("System Busy — model is generating. Try again.")
        self._set_generating(True)

    def release(self) -> None:
        """Release a previous acquire."""
        self._set_generating(False)

    def ensure_loaded(self) -> None:
        """Lazy load with a normalised error."""
        if not self.is_loaded():
            try:
                self.load()
            except FileNotFoundError as e:
                raise RuntimeError(str(e)) from e

    def load(self) -> None:
        """Load most recent GGUF."""
        if self.is_loaded():
            return
        disc = self._discover_models()
        mp = (
            Path(self.model_path)
            if self.model_path and Path(self.model_path).exists()
            else (disc[0] if disc else None)
        )
        if mp is None or not mp.exists():
            raise FileNotFoundError(
                f"No GGUF in {Path(config.LLAMA_MODEL_PATH)}. Download a model via Explore."
            )
        self.model_path = str(mp)

        cores = config.LLAMA_N_THREADS or get_physical_cores()
        ctx = get_default_ctx()
        logger.info("load model=%s ctx=%s", mp.name, ctx)
        common_kwargs = dict(
            model_path=str(mp),
            n_ctx=ctx,
            n_threads=cores,
            n_threads_batch=cores,
            n_batch=512,
            n_ubatch=256,
            type_k=llama_cpp.GGML_TYPE_Q4_0,
            type_v=llama_cpp.GGML_TYPE_Q4_0,
            flash_attn=True,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )
        gpu_layers = config.LLAMA_N_GPU_LAYERS
        if gpu_layers is None:
            try:
                self.llm = Llama(n_gpu_layers=-1, **common_kwargs)
            except Exception:
                self.llm = Llama(n_gpu_layers=0, **common_kwargs)
        else:
            self.llm = Llama(n_gpu_layers=int(gpu_layers), **common_kwargs)

        try:
            list(
                self.llm.create_chat_completion(
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=1,
                    stream=False,
                )
            )
        except Exception:
            pass

    def unload(self) -> None:
        if self.llm is not None:
            try:
                del self.llm
            except Exception:
                pass
            self.llm = None
            gc.collect()
            logger.info("unload")

    def health(self) -> Dict[str, object]:
        disc = self._discover_models()
        if disc:
            mp = (
                Path(self.model_path)
                if self.model_path and Path(self.model_path).exists()
                else disc[0]
            )
            status = "ready"
            exists = True
        else:
            mp = Path(config.LLAMA_MODEL_PATH) / "no-model.gguf"
            status = "error"
            exists = False
        return {
            "status": status,
            "model": mp.name if mp.name != "no-model.gguf" else None,
            "path": str(mp),
            "exists": exists,
            "loaded": self.is_loaded(),
            "generating": self.is_generating(),
            "available": [p.name for p in disc],
        }


class EmbeddingEngine(LlamaEngine):
    """nomic-embed-text engine — embedding only, inherits engine lifecycle."""

    _instance: Optional["EmbeddingEngine"] = None
    _lock: Lock = Lock()

    def __init__(
        self,
        model_path: Optional[str] = None,
        embed_ctx: int = 2048,
        batch_size: int = 16,
    ):
        super().__init__(model_path=model_path)
        self.embed_ctx = embed_ctx
        self.batch_size = batch_size

    def _resolve_model(self) -> Path:
        """nomic GGUF path: explicit model_path, else the *nomic*.gguf match."""
        if self.model_path and Path(self.model_path).exists():
            return Path(self.model_path)
        hits = [
            p for p in Path(config.LLAMA_MODEL_PATH).glob("*nomic*.gguf") if p.is_file()
        ]
        if not hits:
            raise FileNotFoundError(
                f"No nomic-embed-text GGUF in {config.LLAMA_MODEL_PATH}. "
                "Download e.g. nomic-ai/nomic-embed-text-v1.5-GGUF Q4_K_M."
            )
        return sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    def load(self) -> None:
        """Load nomic GGUF with embedding=True, clipped ctx, no GPU/warm-up."""
        if self.is_loaded():
            return
        mp = self._resolve_model()

        cores = config.LLAMA_N_THREADS or get_physical_cores()
        self.llm = Llama(
            model_path=str(mp),
            embedding=True,
            n_ctx=min(get_default_ctx(), self.embed_ctx),
            n_threads=cores,
            n_threads_batch=cores,
            # Encoder packs up to n_batch tokens per native call, which asserts
            # they fit n_ubatch: keep both at ctx so real-size chunks can't abort.
            n_batch=2048,
            n_ubatch=2048,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )
        self.model_path = str(mp)
        logger.info("embed load model=%s", mp.name)

    def embed(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """Embed texts (normalized)."""
        logger.debug("embed start count=%s", len(texts))

        if not self.is_loaded():
            self.load()

        batch_size = batch_size or self.batch_size
        vectors: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            out = self.llm.embed(texts[i : i + batch_size], normalize=True)
            for vec in out:
                vectors.append(list(vec))
        return vectors

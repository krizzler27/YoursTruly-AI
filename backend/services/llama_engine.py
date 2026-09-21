from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional
import gc

try:
    import llama_cpp
    from llama_cpp import Llama
except ImportError as e:
    raise RuntimeError("llama-cpp-python not installed.") from e

from config import config, list_models
from core.logging import get_logger

logger = get_logger(__name__)


class LlamaEngine:
    """In-process llama.cpp lifecycle - load, switch, health, tenure guards."""

    _instances: Dict[str, "LlamaEngine"] = {}
    _lock: Lock = Lock()

    def __init__(self, model_path: Optional[str] = None, role: Optional[str] = None):
        self.model_path: Optional[str] = model_path
        self._role: Optional[str] = role
        self.llm = None
        self._generating = False
        self._gen_lock = Lock()

    def switch_model(self, full_path: Optional[str]) -> None:
        """Switch to GGUF at full path."""
        if not full_path or not full_path.strip():
            return
        if self.is_generating():
            logger.warning("Model switch rejected - busy")
            raise RuntimeError("System Busy - model is generating. Try again.")
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
    def get_instance(cls, role: str = "chat", model_path: Optional[str] = None) -> "LlamaEngine":
        """Singleton slot per role. Creates empty, never loads."""
        with cls._lock:
            if role not in cls._instances:
                cls._instances[role] = cls(model_path=model_path, role=role)
            return cls._instances[role]

    def is_loaded(self) -> bool:
        return self.llm is not None

    def is_generating(self) -> bool:
        with self._gen_lock:
            return self._generating

    def _set_generating(self, value: bool) -> None:
        with self._gen_lock:
            self._generating = value

    def acquire(self) -> None:
        """Claim the engine or raise Busy. Check-and-set is atomic."""
        with self._gen_lock:
            if self._generating:
                logger.warning("Model busy - acquire rejected")
                raise RuntimeError("System Busy - model is generating. Try again.")
            self._generating = True

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
        """Load the slot path, else the config chat finder pick."""
        if self.is_loaded():
            return
        if self.model_path and Path(self.model_path).exists():
            mp = Path(self.model_path)
        else:
            mp = Path(config.EFFECTIVE_CHAT_MODEL)
        self.model_path = str(mp)

        cores = config.EFFECTIVE_N_THREADS
        ctx = config.EFFECTIVE_N_CTX
        logger.info("Model loaded - %s (ctx=%s)", mp.name, ctx)
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
                logger.warning("GPU load failed, retrying on CPU", exc_info=True)
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
            logger.warning("Model warmup probe failed for %s", mp.name, exc_info=True)

    def unload(self) -> None:
        if self.llm is not None:
            try:
                del self.llm
            except Exception:
                logger.warning("Model unload cleanup failed", exc_info=True)
            finally:
                self.llm = None
            gc.collect()
            logger.info("Model unloaded")

    def health(self) -> Dict[str, object]:
        disc = list_models()
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
    """nomic-embed-text engine - embedding only, inherits engine lifecycle."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        role: Optional[str] = None,
        embed_ctx: int = 2048,
        batch_size: int = 16,
    ):
        super().__init__(model_path=model_path, role=role or "embed")
        self.embed_ctx = embed_ctx
        self.batch_size = batch_size

    def load(self) -> None:
        """Load nomic GGUF with embedding=True, clipped ctx, no GPU/warm-up."""
        if self.is_loaded():
            return
        if self.model_path and Path(self.model_path).exists():
            mp = Path(self.model_path)
        else:
            mp = Path(config.EFFECTIVE_EMBED_MODEL)

        cores = config.EFFECTIVE_N_THREADS
        self.llm = Llama(
            model_path=str(mp),
            embedding=True,
            n_ctx=min(config.EFFECTIVE_N_CTX, self.embed_ctx),
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
        logger.info("Embedding model loaded - %s", mp.name)

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

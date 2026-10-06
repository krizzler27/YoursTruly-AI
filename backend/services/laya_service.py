"""Laya decision-model service - transient System One classifier.

Mirrors EmbeddingEngine in services/llama_engine.py: lazy load on first
use, unload after each predict so only one model is resident on the 8GB
box. Never preloaded in main lifespan. Load failure raises once with a
clear message; callers treat that as fail-open.
"""

from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional
import gc

from config import config
from core.logging import get_logger

logger = get_logger(__name__)

# Frozen question templates, byte-identical to training and to the smoke
# test in backend/notebooks/laya_dataset/test_finetuned.py CASES. Wording
# is model input: never reword without retraining (docs/laya.md 5.3).
ROUTE_QUESTION = {
    "type": "choice",
    "instructions": "Which capability should serve this message?",
    "criteria": {
        "DIRECT": "greetings, smalltalk, general knowledge answerable without files",
        "RAG": "needs attached local files or ingested documents",
        "WEB": "needs fresh internet info, current events, latest versions",
    },
}

NEEDS_MEMORY_QUESTION = {
    "type": "noul",
    "instructions": "Does this need personal memory like name, preferences, or earlier conversation?",
}

HIT_GRADE_QUESTION = {
    "type": "score",
    "instructions": "How relevant is this chunk to the query?",
    "criteria": ["irrelevant", "partial", "exact"],
}


class LayaService:
    """Singleton Laya wrapper - load, predict, unload per call."""

    _instance: Optional["LayaService"] = None
    _lock: Lock = Lock()

    def __init__(self, model_dir: Optional[str] = None):
        self._model_dir = model_dir
        self._agent = None

    @classmethod
    def get_instance(cls, model_dir: Optional[str] = None) -> "LayaService":
        """Singleton slot. Creates empty, never loads."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(model_dir=model_dir)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Drop the singleton, test use only."""
        with cls._lock:
            inst = cls._instance
            cls._instance = None
        if inst is not None:
            inst.unload()

    def _model_path(self) -> Path:
        """Model dir from knob or override; relative resolves at repo root."""
        raw = (self._model_dir or config.LAYA_MODEL_DIR or "").strip()
        p = Path(raw)
        if not p.is_absolute():
            root = Path(__file__).resolve().parent.parent.parent
            p = root / p
        return p

    def is_loaded(self) -> bool:
        return self._agent is not None

    def load(self) -> None:
        """Lazy load the fine-tuned weights, once."""
        if self.is_loaded():
            return
        model_dir = self._model_path()
        if not model_dir.is_dir():
            raise RuntimeError(
                f"Laya model dir not found: {model_dir}. "
                "Train via backend/notebooks/laya_finetune.ipynb or set LAYA_MODEL_DIR."
            )
        try:
            import laya

            # CPU-only: the 8GB box keeps Vulkan for the chat slot, and the
            # checkpoint is verified on CPU (~700-930ms per case).
            self._agent = laya.Agent(str(model_dir), device="cpu")
        except Exception as e:
            self._agent = None
            raise RuntimeError(f"Laya load failed for {model_dir}: {e}") from e
        logger.info("Laya model loaded - %s", model_dir.name)

    def unload(self) -> None:
        """Release weights so peak stays bounded, same as embed."""
        if self._agent is not None:
            try:
                del self._agent
            except Exception:
                logger.warning("Laya unload cleanup failed", exc_info=True)
            finally:
                self._agent = None
            gc.collect()
            logger.info("Laya model unloaded")

    def predict(
        self, state: Dict[str, Any], questions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """One transient forward pass: load, predict, unload."""
        self.load()
        try:
            return self._agent.predict(state, questions)
        finally:
            self.unload()

    def predict_many(
        self, states: list, questions: Dict[str, Dict[str, Any]]
    ) -> list:
        """Batched forward passes under one load, unload in finally.

        Same 8GB contract as predict (never resident across requests):
        one load serves the whole hit list instead of one load per hit.
        """
        self.load()
        try:
            agent = self._agent
            return [agent.predict(s, questions) for s in states]
        finally:
            self.unload()

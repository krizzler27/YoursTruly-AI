"""Background-friendly summarization over the worker or chat slot.

Ordering rule: call summarize_text BEFORE acquiring the chat engine for
generation, never nested inside acquire. It takes its own short acquire
(chat path) or owns the worker slot outright (worker path), so calling it
while holding the chat acquire would deadlock on the single streamline.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from config import config
from core.logging import get_logger
from services.llama_engine import LlamaEngine, WorkerEngine
from services.llm_service import LLMService
from services.prompt_manager import PromptManager

logger = get_logger(__name__)

CHARS_PER_TOKEN = 4


def extractive_fallback(text: str, max_chars: int) -> str:
    """First non-empty lines joined, truncated - no model needed."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return " / ".join(lines)[:max_chars].strip()


def _run_chat(text: str, max_tokens: int) -> str:
    """One SLM pass on the chat slot; Busy propagates to the caller."""
    engine = LlamaEngine.get_instance("chat")
    if engine.is_generating():
        raise RuntimeError("System Busy - model is generating. Try again.")
    engine.ensure_loaded()
    svc = LLMService(engine=engine)
    messages = [
        {"role": "user", "content": PromptManager.render("doc_summary.j2", doc_text=text)}
    ]
    return (svc.invoke(messages, max_tokens=max_tokens, temperature=0.2) or "").strip()


def _run_worker(text: str, max_tokens: int) -> str:
    """One SLM pass on the worker slot; load on demand, unload after."""
    engine = WorkerEngine.get_instance("worker")
    engine.ensure_loaded()
    try:
        svc = LLMService(engine=engine)
        messages: List[Dict[str, str]] = [
            {"role": "user", "content": PromptManager.render("doc_summary.j2", doc_text=text)}
        ]
        return (svc.invoke(messages, max_tokens=max_tokens, temperature=0.2) or "").strip()
    finally:
        engine.unload()


def summarize_text(
    text: str,
    max_tokens: int = 128,
    timeout: Optional[float] = None,
    role: Optional[str] = None,
) -> str:
    """SLM summary with extractive fallback; never raises on model failure.

    Picks the engine by role (default SUMMARY_MODEL_ROLE): the worker slot
    loads on demand and unloads after so chat never blocks, while the chat
    slot busy-checks and skips. Enforces timeout, falls back to the first
    lines on any failure. Returns "" for blank input.
    """
    if not (text or "").strip():
        return ""
    slot = role or config.SUMMARY_MODEL_ROLE
    budget_s = timeout if timeout is not None else config.SUMMARY_TIMEOUT_S
    head = text.strip()
    fallback_chars = max_tokens * CHARS_PER_TOKEN

    if slot == "worker":
        runner = _run_worker
    else:
        if LlamaEngine.get_instance("chat").is_generating():
            logger.warning("summary_skipped_busy slot=chat")
            return extractive_fallback(head, fallback_chars)
        runner = _run_chat

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(runner, head, max_tokens)
        try:
            out = future.result(timeout=budget_s)
        except TimeoutError:
            logger.warning("summary_timeout slot=%s budget_s=%s", slot, budget_s)
            future.add_done_callback(lambda _f: _f.exception())
            return extractive_fallback(head, fallback_chars)
        if not out:
            return extractive_fallback(head, fallback_chars)
        return out
    except RuntimeError as e:
        if "Busy" in str(e):
            logger.warning("summary_skipped_busy slot=%s err=%s", slot, e)
        else:
            logger.warning("summary SLM failed, extractive fallback: %s", e)
        return extractive_fallback(head, fallback_chars)
    except Exception as e:
        logger.warning("summary SLM failed, extractive fallback: %s", e)
        return extractive_fallback(head, fallback_chars)
    finally:
        pool.shutdown(wait=False, cancel_futures=False)

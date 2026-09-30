"""Background-friendly summarization over the worker or chat slot.

Ordering rule: call summarize_text BEFORE acquiring the chat engine for
generation, never nested inside acquire. It takes its own short acquire
(chat path) or owns the worker slot outright (worker path), so calling it
while holding the chat acquire would deadlock on the single streamline.

Blocking semantics: the worker slot is a plain blocking call - the
background caller (rollup/ingest workers) already owns that slot, so no
timeout is enforced there. The foreground chat slot busy-check-skips while
generating and enforces timeout through one shared pool; every failure
falls back to the extractive first-lines summary.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from config import config, total_ram_gb
from core.logging import get_logger
from services.llama_engine import LlamaEngine, WorkerEngine, resolve_worker_model_path
from services.llm_service import LLMService
from services.prompt_manager import PromptManager

logger = get_logger(__name__)

CHARS_PER_TOKEN = 4
DEFAULT_SUMMARY_ROLE = "chat"
MIN_WORKER_RAM_GB = 16.0

# One shared pool for the foreground chat slot only: timeout enforcement
# without building a throwaway executor per call. Worker-slot calls never
# touch this pool; they block on their own owned slot.
_CHAT_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="summary-chat")


def extractive_fallback(text: str, max_chars: int) -> str:
    """First non-empty lines joined, truncated - no model needed."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return " / ".join(lines)[:max_chars].strip()


def resolve_slot(explicit_role: Optional[str] = None) -> str:
    """Pick the summary slot without loading weights.

    Explicit role always wins. A non-default SUMMARY_MODEL_ROLE is
    respected as-is. Otherwise auto-route: worker when RAM >= 16GB and
    a real worker GGUF resolves, else chat.
    """
    if explicit_role:
        logger.debug("summary slot explicit role=%s", explicit_role)
        return explicit_role
    configured = getattr(config, "SUMMARY_MODEL_ROLE", DEFAULT_SUMMARY_ROLE) or DEFAULT_SUMMARY_ROLE
    if configured != DEFAULT_SUMMARY_ROLE:
        logger.debug("summary slot configured role=%s", configured)
        return configured
    try:
        ram_gb = total_ram_gb()
    except Exception as e:
        logger.debug("summary slot auto=chat ram probe failed err=%s", e)
        return DEFAULT_SUMMARY_ROLE
    if ram_gb < MIN_WORKER_RAM_GB:
        logger.debug("summary slot auto=chat ram_gb=%.1f below %.1f", ram_gb, MIN_WORKER_RAM_GB)
        return DEFAULT_SUMMARY_ROLE
    try:
        worker_path, is_fallback = resolve_worker_model_path()
    except Exception as e:
        logger.debug("summary slot auto=chat worker resolve failed err=%s", e)
        return DEFAULT_SUMMARY_ROLE
    if is_fallback:
        logger.debug("summary slot auto=chat no worker GGUF ram_gb=%.1f", ram_gb)
        return DEFAULT_SUMMARY_ROLE
    logger.debug("summary slot auto=worker ram_gb=%.1f path=%s", ram_gb, worker_path)
    return "worker"


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

    Picks the engine by role (default SUMMARY_MODEL_ROLE). The worker slot
    is fully blocking: the background caller owns it, so the call runs
    inline with no timeout enforced. The chat slot is foreground-only: it
    skips while the engine is generating (summary_skipped_busy) and enforces
    timeout through the shared pool, logging summary_timeout on expiry.
    Blank input returns "" and any failure returns the first-lines fallback.
    Without an explicit role, 16GB+ boxes with a real worker GGUF auto-route
    to worker so background summaries skip the chat single-flight guard.
    """
    if not (text or "").strip():
        return ""
    slot = resolve_slot(role)
    head = text.strip()
    fallback_chars = max_tokens * CHARS_PER_TOKEN

    if slot == "worker":
        try:
            out = _run_worker(head, max_tokens)
        except RuntimeError as e:
            if "Busy" in str(e):
                logger.warning("summary_skipped_busy slot=%s err=%s", slot, e)
            else:
                logger.warning("summary SLM failed, extractive fallback: %s", e)
            return extractive_fallback(head, fallback_chars)
        except Exception as e:
            logger.warning("summary SLM failed, extractive fallback: %s", e)
            return extractive_fallback(head, fallback_chars)
        if not out:
            return extractive_fallback(head, fallback_chars)
        return out

    if LlamaEngine.get_instance("chat").is_generating():
        logger.warning("summary_skipped_busy slot=chat")
        return extractive_fallback(head, fallback_chars)
    budget_s = timeout if timeout is not None else config.SUMMARY_TIMEOUT_S
    future = _CHAT_POOL.submit(_run_chat, head, max_tokens)
    try:
        out = future.result(timeout=budget_s)
    except TimeoutError:
        logger.warning("summary_timeout slot=%s budget_s=%s", slot, budget_s)
        future.add_done_callback(lambda _f: _f.exception())
        return extractive_fallback(head, fallback_chars)
    except RuntimeError as e:
        if "Busy" in str(e):
            logger.warning("summary_skipped_busy slot=%s err=%s", slot, e)
        else:
            logger.warning("summary SLM failed, extractive fallback: %s", e)
        return extractive_fallback(head, fallback_chars)
    except Exception as e:
        logger.warning("summary SLM failed, extractive fallback: %s", e)
        return extractive_fallback(head, fallback_chars)
    if not out:
        return extractive_fallback(head, fallback_chars)
    return out

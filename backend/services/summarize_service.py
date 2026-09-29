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

from config import config
from core.logging import get_logger
from services.llama_engine import LlamaEngine, WorkerEngine
from services.llm_service import LLMService
from services.prompt_manager import PromptManager

logger = get_logger(__name__)

CHARS_PER_TOKEN = 4

# One shared pool for the foreground chat slot only: timeout enforcement
# without building a throwaway executor per call. Worker-slot calls never
# touch this pool; they block on their own owned slot.
_CHAT_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="summary-chat")


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

    Picks the engine by role (default SUMMARY_MODEL_ROLE). The worker slot
    is fully blocking: the background caller owns it, so the call runs
    inline with no timeout enforced. The chat slot is foreground-only: it
    skips while the engine is generating (summary_skipped_busy) and enforces
    timeout through the shared pool, logging summary_timeout on expiry.
    Blank input returns "" and any failure returns the first-lines fallback.
    """
    if not (text or "").strip():
        return ""
    slot = role or config.SUMMARY_MODEL_ROLE
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

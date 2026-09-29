"""Token budgets over one shared window: count, allocate, fit."""

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from config import config

if TYPE_CHECKING:
    from schemas.rag_schemas import SearchHit

_FALLBACK_CHARS_PER_TOKEN = 4


def count_tokens(text: Optional[str]) -> int:
    """Count via loaded chat engine, else ceil(len/4); never force-loads."""
    if not text:
        return 0
    try:
        from services.llama_engine import LlamaEngine

        engine = LlamaEngine.get_instance("chat")
        llm = getattr(engine, "llm", None)
        if llm is None:
            raise RuntimeError("chat engine not loaded")
        return max(0, len(list(llm.tokenize(text.encode("utf-8")))))
    except Exception:
        return max(1, math.ceil(len(text) / _FALLBACK_CHARS_PER_TOKEN))


def truncate_text(text: str, max_tokens: int) -> str:
    """Char-proportional cut measured by count_tokens; deterministic."""
    if not text or max_tokens <= 0:
        return ""
    if count_tokens(text) <= max_tokens:
        return text
    cut = text[: max(1, max_tokens * _FALLBACK_CHARS_PER_TOKEN)]
    while cut and count_tokens(cut) > max_tokens:
        cut = cut[: int(len(cut) * 0.9)]
    return cut


def top_k_for_ctx(
    n_ctx: Optional[int] = None, override: Optional[int] = None
) -> int:
    """Top-k by window bin: 3 at 4096, 5 at 8192 and above."""
    if override is not None:
        return int(override)
    window = n_ctx or config.EFFECTIVE_N_CTX
    return 5 if window >= 8192 else 3


def allocate(
    route: str = "DIRECT",
    needs_memory: bool = False,
    n_ctx: Optional[int] = None,
    system_tokens: int = 300,
    query_tokens: int = 0,
) -> Dict[str, int]:
    """Split one window into answer, memory, rag, and history caps."""
    window = n_ctx or config.EFFECTIVE_N_CTX
    safety = int(config.SAFETY_MARGIN)
    total = max(0, window - safety)
    answer = int(
        config.ANSWER_RESERVE_8K if window >= 8192 else config.ANSWER_RESERVE_4K
    )
    usable = max(0, total - answer - max(0, system_tokens) - max(0, query_tokens))
    if needs_memory:
        mem = int(
            config.MEMORY_MAX_TOKENS_8K
            if window >= 8192
            else config.MEMORY_MAX_TOKENS_4K
        )
    else:
        mem = 0
    mem = max(0, min(mem, usable))
    rest = usable - mem
    if (route or "DIRECT").upper() == "RAG":
        share = float(
            config.RAG_MAX_SHARE_8K if window >= 8192 else config.RAG_MAX_SHARE_4K
        )
        rag_cap = max(0, min(rest, int(rest * share)))
        history_cap = max(0, rest - rag_cap)
    else:
        share = float(
            config.HISTORY_MAX_SHARE_8K
            if window >= 8192
            else config.HISTORY_MAX_SHARE_4K
        )
        history_cap = max(0, min(rest, int(rest * share)))
        rag_cap = 0
    return {
        "total": total,
        "safety": safety,
        "answer_reserve": answer,
        "usable": usable,
        "history_cap": history_cap,
        "rag_cap": rag_cap,
        "mem_cap": mem,
        "top_k": top_k_for_ctx(window),
    }


def rag_context_tokens(n_ctx: Optional[int] = None) -> int:
    """RAG token budget derived from allocate, never a fixed constant."""
    return allocate(route="RAG", n_ctx=n_ctx)["rag_cap"]


def fit_history(
    turns: Optional[List[Dict[str, str]]], cap_tokens: int
) -> Tuple[List[Dict[str, str]], bool, int]:
    """Drop oldest turns first, keep last 2 verbatim; report truncation."""
    items = list(turns or [])

    def _used(ts: List[Dict[str, str]]) -> int:
        return sum(count_tokens((t or {}).get("content") or "") for t in ts)

    if not items:
        return ([], False, 0)
    used = _used(items)
    if used <= cap_tokens:
        return (items, False, used)
    fitted = list(items)
    while len(fitted) > 2 and _used(fitted) > cap_tokens:
        fitted.pop(0)
    return (fitted, True, _used(fitted))


def fit_hits(
    hits: Optional[List["SearchHit"]], cap_tokens: int
) -> Tuple[List["SearchHit"], bool, int]:
    """Greedy score-ordered fill, truncate straddler; report truncation."""
    items = list(hits or [])
    if not items:
        return ([], False, 0)
    if cap_tokens <= 0:
        return ([], True, 0)
    ordered = sorted(
        items, key=lambda h: getattr(h, "score", 0.0) or 0.0, reverse=True
    )
    fitted: List["SearchHit"] = []
    used = 0
    truncated = False
    for h in ordered:
        need = count_tokens(getattr(h, "text", "") or "")
        if used + need <= cap_tokens:
            fitted.append(h)
            used += need
            continue
        room = cap_tokens - used
        if room > 0:
            cut = truncate_text(getattr(h, "text", "") or "", room)
            if cut:
                fitted.append(h.model_copy(update={"text": cut}))
                used += count_tokens(cut)
        truncated = True
        break
    return (fitted, truncated, used)

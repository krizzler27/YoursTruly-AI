"""Dev-only LangSmith gate — identity when off, real @traceable when on.

Off by default (no env, no key, or frozen exe) — gate is cheap and no
network happens. On only when IS_DEV + LANGSMITH_TRACING=true + API key
are present; run against public/synthetic data only.
"""

import os
from typing import Any, Callable, Optional

from dotenv import load_dotenv

load_dotenv()


def _enabled() -> bool:
    try:
        from config import config

        if not config.IS_DEV:
            return False
    except Exception:
        return False
    if os.getenv("LANGSMITH_TRACING", "").lower() != "true" and os.getenv(
        "LANGCHAIN_TRACING_V2", ""
    ).lower() != "true":
        return False
    if not (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
        return False
    return True


def traceable(*, name: Optional[str] = None, **kw: Any) -> Callable:
    """Usage: @traceable(name="...") only — no-op when gate is off."""
    if not _enabled():
        return lambda fn: fn

    from langsmith import traceable as _real

    if name is not None:
        return _real(name=name, **kw)
    return _real(**kw) if kw else _real

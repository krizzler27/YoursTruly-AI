"""Explicit-remember write path - hardcoded patterns, no model use."""

import re
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from core.logging import get_logger
from db.models import SemanticMemoryModel
from repository.semantic_repository import SemanticMemoryRepository
from schemas.rag_schemas import SemanticFactList

logger = get_logger(__name__)

_VALUE_MAX_CHARS = 500
_KEY_MAX_CHARS = 128

_CALL_ME_RE = re.compile(r"\bcall\s+me\s+(?P<value>[a-zA-Z][a-zA-Z'.\- ]{0,60})", re.IGNORECASE)
_MY_NAME_IS_RE = re.compile(r"\bmy\s+name\s+is\s+(?P<value>[a-zA-Z][a-zA-Z'.\- ]{0,60})", re.IGNORECASE)
_REMEMBER_MY_ATTR_RE = re.compile(
    r"\bremember\s+(?:that\s+)?my\s+(?P<key>[a-zA-Z][a-zA-Z '\-]{0,40}?)\s+is\s+(?P<value>.+)",
    re.IGNORECASE,
)
_REMEMBER_PREF_RE = re.compile(
    r"\bremember\s+(?:that\s+)?i\s+(?:prefer|like|love|hate|dislike)\s+(?P<value>.+)",
    re.IGNORECASE,
)
_MY_ATTR_RE = re.compile(
    r"\bmy\s+(?P<key>[a-zA-Z][a-zA-Z '\-]{0,40}?)\s+is\s+(?P<value>.+)",
    re.IGNORECASE,
)
_PREF_RE = re.compile(
    r"\bi\s+(?:prefer|like|love|hate|dislike)\s+(?P<value>.+)",
    re.IGNORECASE,
)
_REMEMBER_RE = re.compile(r"\bremember\b", re.IGNORECASE)
_GENERIC_IS_RE = re.compile(
    r"^(?:the\s+)?(?P<key>[a-zA-Z][a-zA-Z '\-]{0,40}?)\s+is\s+(?P<value>.+)$",
    re.IGNORECASE,
)
_NAME_CLAUSE_RE = re.compile(r"\s+and\s+(?:i|my|we|you)\b", re.IGNORECASE)
_QUESTION_START_RE = re.compile(
    r"^(what|which|who|how|when|where|why|is|are|do|does|did|can|could|would|should)\b",
    re.IGNORECASE,
)


def _clean_value(raw: str) -> str:
    """Strip quotes and trailing punctuation, cap length."""
    clean = (raw or "").strip().strip("\"'").strip()
    clean = re.sub(r"[.,!?;:\-]+$", "", clean).strip()
    return clean[:_VALUE_MAX_CHARS].strip()


def _clean_key(raw: str) -> str:
    """Lowercase attribute name, single-spaced, capped."""
    clean = re.sub(r"\s+", " ", (raw or "").strip().lower()).strip(" '\":;-")
    return clean[:_KEY_MAX_CHARS].strip()


def _clean_name(raw: str) -> str:
    """One short name: drop trailing clauses, keep the first word."""
    first = _NAME_CLAUSE_RE.split((raw or "").strip(), maxsplit=1)[0]
    clean = _clean_value(first)
    return clean.split()[0] if clean.split() else ""


def parse_explicit_fact(text: str) -> Optional[Tuple[str, str]]:
    """Parse one (key, value) fact from an explicit command, else None."""
    raw = (text or "").strip()
    if not raw:
        return None
    is_question = raw.endswith("?") or bool(_QUESTION_START_RE.search(raw))
    has_command = bool(_REMEMBER_RE.search(raw) or _CALL_ME_RE.search(raw))
    if is_question and not has_command:
        return None
    if raw.endswith("?"):
        return None
    m = _CALL_ME_RE.search(raw)
    if m:
        value = _clean_name(m.group("value"))
        if value:
            return ("name", value)
    m = _REMEMBER_MY_ATTR_RE.search(raw)
    if m:
        key, value = _clean_key(m.group("key")), _clean_value(m.group("value"))
        if key and value:
            return (key, value)
    m = _MY_NAME_IS_RE.search(raw)
    if m:
        value = _clean_name(m.group("value"))
        if value:
            return ("name", value)
    m = _REMEMBER_PREF_RE.search(raw)
    if m:
        value = _clean_value(m.group("value"))
        if value:
            return ("preference", value)
    m = _MY_ATTR_RE.search(raw)
    if m and not _QUESTION_START_RE.search(raw):
        key, value = _clean_key(m.group("key")), _clean_value(m.group("value"))
        if key and value:
            return (key, value)
    m = _PREF_RE.search(raw)
    if m and not _QUESTION_START_RE.search(raw):
        value = _clean_value(m.group("value"))
        if value:
            return ("preference", value)
    if _REMEMBER_RE.search(raw):
        tail = _REMEMBER_RE.split(raw, maxsplit=1)[1]
        tail = re.sub(r"^(that|this|please|to|it|:|,|\-)+", "", tail.strip(), flags=re.IGNORECASE).strip()
        m = _GENERIC_IS_RE.match(tail)
        if m:
            key, value = _clean_key(m.group("key")), _clean_value(m.group("value"))
            if key and value:
                return (key, value)
        value = _clean_value(tail)
        if value:
            return (_clean_key(value) or "note", value)
    return None


def remember_explicit(db: Session, query: str) -> Optional[SemanticMemoryModel]:
    """Store one explicit fact, None when absent or on error."""
    try:
        parsed = parse_explicit_fact(query)
        if parsed is None:
            return None
        key, value = parsed
        return SemanticMemoryRepository(db).upsert(key, value)
    except Exception as e:
        logger.debug("explicit remember skipped: %s", e)
        return None


EXTRACTION_MAX_TOKENS = 256
EXTRACTION_BATCH_TURNS = 10


def extract_facts_slm(lines: List[str]) -> List[Tuple[str, str]]:
    """Extract (key, value) facts via worker-slot SLM, [] on any failure."""
    batch = [ln.strip() for ln in (lines or []) if (ln or "").strip()]
    if not batch:
        return []
    try:
        from services.llama_engine import WorkerEngine
        from services.llm_service import LLMService
        from services.prompt_manager import PromptManager

        prompt_text = PromptManager.render(
            "extract_facts.j2",
            user_lines="\n".join(f"- {ln}" for ln in batch),
        )
    except Exception as e:
        logger.warning("extract_skipped_prompt err=%s", e)
        return []
    try:
        engine = WorkerEngine.get_instance("worker")
        engine.ensure_loaded()
        try:
            svc = LLMService(engine=engine)
            result = svc.invoke(
                [{"role": "user", "content": prompt_text}],
                max_tokens=EXTRACTION_MAX_TOKENS,
                temperature=0.2,
                structured_output=SemanticFactList,
            )
        finally:
            engine.unload()
    except RuntimeError as e:
        if "Busy" in str(e):
            logger.warning("extract_skipped_busy err=%s", e)
        else:
            logger.warning("extract SLM failed, skipped: %s", e)
        return []
    except Exception as e:
        logger.warning("extract SLM failed, skipped: %s", e)
        return []
    try:
        assert isinstance(result, SemanticFactList)
        return [
            (f.key, f.value) for f in result.facts if (f.key or "").strip()
        ]
    except Exception as e:
        logger.debug("extract parse skipped: %s", e)
        return []


def extract_and_store(db: Session, messages, limit: int = EXTRACTION_BATCH_TURNS) -> int:
    """Extract background facts from recent user turns, count stored."""
    try:
        texts = [
            (m or {}).get("content", "")
            for m in list(messages or [])[-max(1, int(limit)):]
            if (m or {}).get("role") == "user"
        ]
        batch = [
            t.strip()
            for t in texts
            if (t or "").strip() and parse_explicit_fact(t) is None
        ]
        if not batch:
            return 0
        stored = 0
        for key, value in extract_facts_slm(batch):
            clean_key, clean_value = _clean_key(key), _clean_value(value)
            if not clean_key or not clean_value:
                continue
            try:
                SemanticMemoryRepository(db).upsert(clean_key, clean_value)
                stored += 1
            except Exception as e:
                logger.debug("extract store skipped: %s", e)
                continue
        return stored
    except Exception as e:
        logger.debug("extract_and_store skipped: %s", e)
        return 0

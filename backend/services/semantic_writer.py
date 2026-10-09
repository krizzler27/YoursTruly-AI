"""Explicit-remember fast path plus background semantic extraction."""

import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from config import config
from core.logging import get_logger
from db.models import SemanticMemoryModel
from repository.semantic_repository import SemanticMemoryRepository
from schemas.rag_schemas import SemanticFactList

logger = get_logger(__name__)

# Shared pool so a hung worker SLM cannot stall the serial background
# queue; mirrors summarize_service worker discipline with the same budget.
_EXTRACT_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="extract-worker")

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


def _embed_fact(key: str, value: str) -> Optional[List[float]]:
    """One write-time vector, None on engine failure.

    Loads the small mapped embed model on demand like recall does, then
    unloads to restore the one-resident-model discipline.
    """
    try:
        from repository.semantic_repository import memory_embed_text
        from services.llama_engine import EmbeddingEngine

        clipped = memory_embed_text(f"{key or ''}: {value or ''}")
        if not clipped.strip():
            return None
        engine = EmbeddingEngine.get_instance("embed")
        engine.ensure_loaded()
        try:
            vecs = engine.embed([clipped])
        finally:
            engine.unload()
        if not isinstance(vecs, list) or not vecs:
            return None
        vec = vecs[0]
        if not isinstance(vec, list) or not vec:
            return None
        return [float(x) for x in vec]
    except Exception as e:
        logger.debug("semantic embed skipped: %s", e)
        return None


def _embed_texts(texts: List[str]) -> List[Optional[List[float]]]:
    """Batch vectors with load plus unload around, Nones on failure."""
    clipped = [t for t in ((s or "").strip() for s in (texts or []))]
    if not any(clipped):
        return [None] * len(clipped)
    try:
        from services.llama_engine import EmbeddingEngine

        engine = EmbeddingEngine.get_instance("embed")
        engine.ensure_loaded()
        try:
            vecs = engine.embed(clipped)
        finally:
            engine.unload()
    except Exception as e:
        logger.debug("semantic batch embed skipped: %s", e)
        return [None] * len(clipped)
    out: List[Optional[List[float]]] = []
    for vec in vecs or []:
        if isinstance(vec, list) and vec:
            try:
                out.append([float(x) for x in vec])
                continue
            except (TypeError, ValueError):
                pass
        out.append(None)
    while len(out) < len(clipped):
        out.append(None)
    return out[: len(clipped)]


def _embed_many(pairs: List[Tuple[str, str]]) -> List[Optional[List[float]]]:
    """Batch write-time vectors for background use, load plus unload around.

    Off the request path one bounded load is safe; unload restores the
    one-resident-model discipline. All fail-open to None.
    """
    if not pairs:
        return []
    from repository.semantic_repository import memory_embed_text

    return _embed_texts([memory_embed_text(f"{k or ''}: {v or ''}") for k, v in pairs])


def remember_explicit(db: Session, query: str) -> Optional[SemanticMemoryModel]:
    """Store one explicit fact, None when absent or on error."""
    try:
        parsed = parse_explicit_fact(query)
        if parsed is None:
            return None
        key, value = parsed
        vec = _embed_fact(key, value)
        return SemanticMemoryRepository(db).upsert(key, value, embedding=vec)
    except Exception as e:
        logger.debug("explicit remember skipped: %s", e)
        return None


EXTRACTION_MAX_TOKENS = 256
EXTRACTION_BATCH_TURNS = 10


def _run_extract(prompt_text: str):
    """One SLM pass on the worker slot; load on demand, unload after."""
    from services.llama_engine import WorkerEngine
    from services.llm_service import LLMService

    engine = WorkerEngine.get_instance("worker")
    engine.ensure_loaded()
    try:
        svc = LLMService(engine=engine)
        return svc.invoke(
            [{"role": "user", "content": prompt_text}],
            max_tokens=EXTRACTION_MAX_TOKENS,
            temperature=0.2,
            structured_output=SemanticFactList,
        )
    finally:
        engine.unload()


def extract_facts_slm(
    lines: List[str], timeout: Optional[float] = None
) -> List[Tuple[str, str]]:
    """Extract (key, value) facts via worker-slot SLM, [] on any failure."""
    batch = [ln.strip() for ln in (lines or []) if (ln or "").strip()]
    if not batch:
        return []
    try:
        from services.prompt_manager import PromptManager

        prompt_text = PromptManager.render(
            "extract_facts.j2",
            user_lines="\n".join(f"- {ln}" for ln in batch),
        )
    except Exception as e:
        logger.warning("extract_skipped_prompt err=%s", e)
        return []
    budget_s = timeout if timeout is not None else config.SUMMARY_TIMEOUT_S
    future = _EXTRACT_POOL.submit(_run_extract, prompt_text)
    try:
        result = future.result(timeout=budget_s)
    except TimeoutError:
        logger.warning("extract_timeout budget_s=%s", budget_s)
        future.add_done_callback(lambda _f: _f.exception())
        return []
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
        pairs = []
        for key, value in extract_facts_slm(batch):
            clean_key, clean_value = _clean_key(key), _clean_value(value)
            if not clean_key or not clean_value:
                continue
            pairs.append((clean_key, clean_value))
        stored = 0
        for (clean_key, clean_value), vec in zip(pairs, _embed_many(pairs)):
            try:
                SemanticMemoryRepository(db).upsert(
                    clean_key, clean_value, embedding=vec
                )
                stored += 1
            except Exception as e:
                logger.debug("extract store skipped: %s", e)
                continue
        return stored
    except Exception as e:
        logger.debug("extract_and_store skipped: %s", e)
        return 0


BACKFILL_DEFAULT_LIMIT = 10


def backfill_missing_vectors(db: Session, limit: int = BACKFILL_DEFAULT_LIMIT) -> int:
    """Embed vector-less semantic and episodic rows, capped. Returns healed."""
    try:
        from db.models import EpisodicMemoryModel, SemanticMemoryModel
        from repository.episodic_repository import EpisodicMemoryRepository
        from repository.semantic_repository import memory_embed_text

        remaining = max(0, int(limit))
        if not remaining:
            return 0
        targets = []
        try:
            sem_rows = (
                db.query(SemanticMemoryModel)
                .filter(SemanticMemoryModel.embedding.is_(None))
                .order_by(SemanticMemoryModel.updated_at.desc())
                .limit(remaining)
                .all()
            )
        except Exception as e:
            logger.debug("backfill semantic list skipped: %s", e)
            sem_rows = []
        for row in sem_rows or []:
            if remaining <= 0:
                break
            text = memory_embed_text(f"{getattr(row, 'key', '') or ''}: {getattr(row, 'value', '') or ''}")
            if text.strip():
                targets.append((row, text))
                remaining -= 1
        try:
            epi_rows = (
                db.query(EpisodicMemoryModel)
                .filter(EpisodicMemoryModel.embedding.is_(None))
                .order_by(EpisodicMemoryModel.created_at.desc())
                .limit(max(0, remaining))
                .all()
            )
        except Exception as e:
            logger.debug("backfill episodic list skipped: %s", e)
            epi_rows = []
        for row in epi_rows or []:
            if remaining <= 0:
                break
            text = memory_embed_text(getattr(row, "summary", "") or "")
            if text.strip():
                targets.append((row, text))
                remaining -= 1
        if not targets:
            return 0
        healed = 0
        for row, vec in zip(
            [r for r, _ in targets],
            _embed_texts([t for _, t in targets]),
        ):
            if not isinstance(vec, list) or not vec:
                continue
            try:
                from repository.semantic_repository import encode_embedding

                row.embedding = encode_embedding(vec)
                db.commit()
                healed += 1
            except Exception as e:
                logger.debug("backfill store skipped: %s", e)
                try:
                    db.rollback()
                except Exception:
                    pass
                continue
        return healed
    except Exception as e:
        logger.debug("backfill skipped: %s", e)
        return 0

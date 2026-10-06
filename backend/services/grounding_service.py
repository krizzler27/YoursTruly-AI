"""L3 grounding gate - check the draft against cited context before emit.

Grades RAG-with-hits drafts only via Laya is_grounded (frozen template,
byte-identical to the smoke test in
backend/notebooks/laya_dataset/test_finetuned.py). DIRECT turns stream
exactly as today: zero behavior change, zero latency.

Three-state verdict from grade_draft:
- True means grounded, stream the buffered draft.
- False means ungrounded, substitute the refusal template.
- None means skip the gate (empty input, malformed Laya shape, or any
  error), stream the draft as today (fail-open).
"""

import asyncio
import uuid
from typing import AsyncIterator, Dict, List, Optional, Tuple, Union

from sqlalchemy.orm import Session

from config import config
from core.logging import get_logger
from db.models import DocumentsModel
from repository.lance_repository import sid
from services.laya_service import LayaService

logger = get_logger(__name__)

# Frozen question template, byte-identical to training and to the smoke
# test CASES ("grounding hallucinated"). Wording is model input: never
# reword without retraining (docs/laya.md 5.3).
IS_GROUNDED_QUESTION = {
    "type": "noul",
    "instructions": "Is this answer fully supported by the provided context, with no outside facts?",
}


def should_gate(route: str, hits) -> bool:
    """Gate RAG-with-hits turns only, DIRECT streams untouched."""
    return route == "RAG" and bool(hits)


def resolve_filenames(db: Session, hits) -> Dict[str, str]:
    """Map hit document ids to filenames, {} on any error (fail-open)."""
    try:
        ids = []
        for h in hits or []:
            raw = getattr(h, "document_id", None)
            if raw is None and isinstance(h, dict):
                raw = h.get("document_id")
            try:
                ids.append(uuid.UUID(str(raw)))
            except Exception:
                continue
        if not ids:
            return {}
        rows = (
            db.query(DocumentsModel)
            .filter(DocumentsModel.id.in_(ids))
            .all()
        )
        return {sid(r.id): r.filename for r in rows}
    except Exception as e:
        logger.debug("grounding filenames skipped: %s", e)
        return {}


def build_context_block(hits, filenames: Optional[Dict[str, str]] = None) -> str:
    """Rebuild the cited context block, same labels as rag build_messages."""
    names = filenames or {}
    blocks = []
    for h in hits or []:
        if isinstance(h, dict):
            doc_id = h.get("document_id", "")
            heading = (h.get("heading") or "").strip()
            page = h.get("page")
            text = (h.get("text") or "").strip()
        else:
            doc_id = getattr(h, "document_id", "")
            heading = (getattr(h, "heading", "") or "").strip()
            page = getattr(h, "page", None)
            text = (getattr(h, "text", "") or "").strip()
        name = names.get(doc_id, doc_id)
        if heading:
            label = f"{name}:{heading}"
        elif page is not None:
            label = f"{name}:p{page}"
        else:
            label = name
        if text:
            blocks.append(f"[{label}]\n{text}")
    return "\n\n".join(blocks)


def grade_draft(
    query: str, context: Union[str, list], draft: str
) -> Optional[bool]:
    """Grade draft-vs-context via one is_grounded predict, never raises."""
    try:
        draft_text = (draft or "").strip()
        if isinstance(context, list):
            context_text = build_context_block(context)
        else:
            context_text = (context or "").strip()
        clean_query = (query or "").strip()
        if not draft_text or not context_text or not clean_query:
            return None
        res = LayaService.get_instance().predict(
            {"query": clean_query, "context": context_text, "answer": draft_text},
            {"is_grounded": IS_GROUNDED_QUESTION},
        )
        answers = (res or {}).get("answers") or {}
        ans = answers.get("is_grounded") or {}
        raw = ans.get("noul")
        if raw is None:
            return None
        try:
            p = float(raw)
        except (TypeError, ValueError):
            return None
        return bool(p >= float(config.LAYA_GROUND_THRESHOLD))
    except Exception as e:
        logger.debug("grounding grade skipped: %s", e)
        return None


def refusal_text(filenames) -> str:
    """Grounded refusal naming the cited files, inviting a rephrase."""
    if isinstance(filenames, dict):
        filenames = list(filenames.values())
    names = sorted({(n or "").strip() for n in filenames or []} - {""})
    if names:
        return (
            f"I can't verify that answer from the cited files ({', '.join(names)}). "
            "Please rephrase your question or ask about something covered in those files."
        )
    return (
        "I can't verify that answer from the cited files. "
        "Please rephrase your question or ask about something covered in those files."
    )


async def gate_draft_stream(
    stream: AsyncIterator[str],
    *,
    query: str,
    hits,
    filenames: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], str, Optional[bool]]:
    """Buffer a RAG draft, grade it, decide what to send.

    Returns (send_parts, send_text, verdict): deltas to emit in order plus
    the text to persist (draft on pass or skip, refusal on fail). Stream
    errors propagate so the caller maps them like today's sse_wrap.
    """
    buf: List[str] = []
    async for delta in stream:
        if delta:
            buf.append(delta)
    draft = "".join(buf).strip()
    if not draft:
        return ([], "", None)
    context_block = build_context_block(hits, filenames)
    verdict = await asyncio.to_thread(grade_draft, query, context_block, draft)
    if verdict is False:
        refusal = refusal_text(filenames)
        return ([refusal], refusal, verdict)
    return (list(buf), draft, verdict)

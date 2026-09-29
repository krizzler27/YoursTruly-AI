"""Agentic entry decider - which capability serves this query.

Routes today: DIRECT (answer from the model) and RAG (local documents).
WEB and other capabilities slot in as new route values plus branches.
Graders (hits, grounding, answer) land here as knobs when needed.
"""

from typing import Dict, List, Optional
import uuid

from sqlalchemy.orm import Session

from repository.document_repository import DocumentRepository
from core.logging import get_logger
from schemas.rag_schemas import RouteDecision
from services.llm_service import LLMService
from services.prompt_manager import PromptManager

logger = get_logger(__name__)


class Decider:
    """Entry decider - mechanical guards plus one structured SLM call."""

    def __init__(
        self,
        db: Session,
        llm: Optional[LLMService] = None,
        max_tokens: int = 64,
        temperature: float = 0.2,
    ):
        self.db = db
        self.llm = llm or LLMService()
        self.docs = DocumentRepository(db)
        self.max_tokens = max_tokens
        self.temperature = temperature

    def decide(
        self,
        query: str,
        conversation_id: Optional[uuid.UUID] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> RouteDecision:
        """DIRECT when empty or the chat has no docs, else one structured SLM call."""
        clean = (query or "").strip()
        if not clean:
            return RouteDecision(route="DIRECT", reason="empty query")
        if conversation_id is not None:
            attached = self.docs.list_by_conversation(conversation_id, limit=8)
            if not attached:
                return RouteDecision(route="DIRECT", reason="no documents attached")
        else:
            if not self.docs.list_recent(limit=1):
                return RouteDecision(route="DIRECT", reason="no documents indexed")
            attached = self.docs.list_recent(limit=8)

        if _asks_about_files(clean):
            return RouteDecision(route="RAG", reason="files inventory question")

        recent = list(history or [])[-2:]
        if _follows_up_files(clean, recent, attached):
            return RouteDecision(route="RAG", reason="follow-up on attached file")

        mem_flag = needs_memory(clean, recent)
        # TODO(P1-schema): thread mem_flag into allocate(needs_memory=...)
        # in rag_service.build_messages and RagGraph.run; the
        # semantic_memory table lookup replaces this keyword stub.
        logger.debug("memory gate needs_memory=%s", mem_flag)

        inventory = "\n".join(
            f"- {d.filename}" + (f": {d.summary}" if (d.summary or "").strip() else "")
            for d in attached
        )
        history_block = _history_text(recent)

        messages = [
            {
                "role": "system",
                "content": PromptManager.render(
                    "rag_gate.j2",
                    attached_docs=inventory,
                    history_block=history_block,
                ),
            },
            {"role": "user", "content": clean},
        ]
        try:
            result = self.llm.invoke(
                messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                structured_output=RouteDecision,
            )
            assert isinstance(result, RouteDecision)
            logger.info("decide route=%s reason=%.100s", result.route, result.reason)
            return result
        except RuntimeError as e:
            # Grammar-constrained SLMs often emit the right word in the
            # wrong envelope; recover it from the raw output if present.
            recovered = _recover_route(str(e))
            if recovered is not None:
                logger.info("decide route=%s reason=recovered", recovered)
                return RouteDecision(route=recovered, reason="recovered")
            logger.warning("decider fallback: %s", e)
            return RouteDecision(route="DIRECT", reason="decider fallback")
        except Exception as e:
            logger.warning("decider fallback: %s", e)
            return RouteDecision(route="DIRECT", reason="decider fallback")


def needs_memory(query: str, history: Optional[List[Dict[str, str]]] = None) -> bool:
    """Stub gate for semantic memory - keyword check until P1-schema lands."""
    import re

    parts = [(query or "")]
    for t in list(history or [])[-2:]:
        parts.append((t or {}).get("content") or "")
    text = " ".join(parts).lower()
    patterns = [
        r"\bcall me\b",
        r"\bremember\b",
        r"\bprefer\b",
        r"\bmy name is\b",
        r"\bname is\b",
        r"\bmy name\b",
    ]
    return any(re.search(p, text) for p in patterns)


def _history_text(turns: Optional[List[Dict[str, str]]], cap_chars: int = 500) -> str:
    """Last turns as User/Assistant lines, capped near cap_chars."""
    items = list(turns or [])[-2:]
    if not items:
        return "No prior conversation."
    lines = []
    for m in items:
        role = "User" if (m or {}).get("role") == "user" else "Assistant"
        lines.append(f"{role}: {(m or {}).get('content', '')}")
    text = "\n".join(lines).strip()
    if len(text) > cap_chars:
        text = text[:cap_chars].rstrip()
    return text or "No prior conversation."


def _follows_up_files(
    query: str,
    history: Optional[List[Dict[str, str]]],
    attached=None,
) -> bool:
    """Short follow-up referring to a prior file question in history."""
    import re

    recent = list(history or [])[-2:]
    if not recent:
        return False
    q = (query or "").lower().strip()
    if not q:
        return False
    hist_text = " ".join(
        ((t or {}).get("content") or "") for t in recent
    ).lower()
    names = []
    for d in attached or []:
        fn = (getattr(d, "filename", "") or "").strip().lower()
        if fn:
            names.append(fn)
    prior_ref = (
        any(n and n in hist_text for n in names)
        or any(_asks_about_files(((t or {}).get("content") or "")) for t in recent)
        or any(tok in hist_text for tok in (".md", ".pdf", ".txt", "attached"))
    )
    if not prior_ref:
        return False
    patterns = [
        r"\bsummariz\w*\s+(it|that|this|above)\b",
        r"^summarize it\b",
        r"\bsection\s+\d+\b",
        r"\bchapter\s+\d+\b",
        r"\bpage\s+\d+\b",
        r"^(what about|and|what does|explain)\b.*\b(section|chapter|page|it|that|this|above)\b",
        r"\b(that|this|it)\s+(part|section|paragraph|page)\b",
    ]
    return any(re.search(p, q) for p in patterns)


def _asks_about_files(query: str) -> bool:
    """Inventory questions answerable from the attached file list."""
    import re

    text = query.lower()
    patterns = [
        r"\bdo you have\b.*\b(files?|documents?|context)\b",
        r"\bwhat\b.*\b(files?|documents?)\b",
        r"\blist\b.*\b(files?|documents?)\b",
        r"\bwhich\b.*\b(files?|documents?)\b",
        r"\bany\b.*\b(files?|documents?)\b",
        r"\battached\b",
        r"\bin your context\b",
        r"\bin (the )?context\b",
    ]
    return any(re.search(p, text) for p in patterns)


def _recover_route(error: str) -> Optional[str]:
    """Last DIRECT|RAG token in the raw model output, if any."""
    import re

    raw = error.split("raw:", 1)[-1]
    found = re.findall(r"\b(DIRECT|RAG)\b", raw, flags=re.IGNORECASE)
    if not found:
        return None
    return found[-1].upper()

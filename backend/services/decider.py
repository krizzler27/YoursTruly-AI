"""Agentic entry decider - which capability serves this query.

Routes today: DIRECT (answer from the model) and RAG (local documents).
WEB and other capabilities slot in as new route values plus branches.
Graders (hits, grounding, answer) land here as knobs when needed.
"""

from typing import Dict, List, NamedTuple, Optional
import uuid

from sqlalchemy.orm import Session

from config import config
from repository.document_repository import DocumentRepository
from repository.episodic_repository import EpisodicMemoryRepository
from repository.semantic_repository import SemanticMemoryRepository, memory_tokens
from core.logging import get_logger
from schemas.rag_schemas import RouteDecision
from services.laya_service import (
    LayaService,
    NEEDS_MEMORY_QUESTION,
    ROUTE_QUESTION,
)
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

        try:
            mem_hits = SemanticMemoryRepository(self.db).find_relevant(clean, limit=5)
        except Exception:
            mem_hits = []
        laya_decision = self._laya_decide(
            clean, recent, attached, mem_hits, conversation_id
        )
        if laya_decision is not None:
            return laya_decision
        sem_hit, epi_hit = _stub_evidence(
            clean, recent, mem_hits, self.db, conversation_id
        )
        mem_flag = bool(sem_hit or epi_hit)
        # mem_flag feeds allocate(needs_memory=...) in build_messages;
        # RagGraph._build passes the same rows as memory_text.
        logger.debug(
            "memory gate needs_memory=%s sem=%s epi=%s mem_hits=%s",
            mem_flag,
            sem_hit,
            epi_hit,
            len(mem_hits),
        )

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
            logger.info(
                "decide route=%s reason=%.100s mem=%s",
                result.route,
                result.reason,
                mem_flag,
            )
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

    def _laya_decide(
        self,
        clean: str,
        recent: List[Dict[str, str]],
        attached,
        mem_hits,
        conversation_id: Optional[uuid.UUID] = None,
    ) -> Optional[RouteDecision]:
        """One transient Laya call for route plus needs-memory.

        Returns a decision on success, None on any failure so decide()
        continues on today's SLM path exactly (fail-open).
        """
        try:
            inventory = "\n".join(
                f"- {d.filename}" + (f": {d.summary}" if (d.summary or "").strip() else "")
                for d in attached
            )
            state = {
                "query": clean,
                "history": _history_text(recent),
                "attached_docs": inventory,
            }
            res = LayaService.get_instance().predict(
                state,
                {"route": ROUTE_QUESTION, "needs_memory": NEEDS_MEMORY_QUESTION},
            )
            answers = (res or {}).get("answers") or {}
            label, conf = _parse_choice(answers, "route")
            p_mem = _parse_noul(answers, "needs_memory")
            if p_mem >= config.LAYA_MEMORY_THRESHOLD:
                mem_flag = True
            else:
                sem_hit, epi_hit = _stub_evidence(
                    clean, recent, mem_hits, self.db, conversation_id
                )
                mem_flag = bool(sem_hit or epi_hit)
                logger.debug(
                    "laya memory below threshold mem=%s sem=%s epi=%s",
                    mem_flag,
                    sem_hit,
                    epi_hit,
                )
            decision = _laya_route(
                label, conf, bool(attached), config.LAYA_ROUTE_THRESHOLD
            )
            logger.info(
                "laya decide route=%s conf=%.2f mem=%s",
                decision.route,
                conf,
                mem_flag,
            )
            return decision
        except Exception as e:
            logger.debug("laya decide skipped, SLM path: %s", e)
            return None


_MEMORY_PATTERNS = [
    r"\bcall me\b",
    r"\bremember\b",
    r"\bprefer\b",
    r"\bmy name is\b",
    r"\bname is\b",
    r"\bmy name\b",
]

_HISTORY_PREF_PATTERNS = [
    r"\bmy name is\b",
    r"\bi (prefer|like|love|hate|dislike)\b",
    r"\bmy (favorite|favourite)\b",
    r"\bremember that\b",
    r"\bcall me\b",
]


def _parse_choice(answers: Dict, qid: str):
    """Choice label plus confidence, zeroed when the answer is malformed."""
    ans = (answers or {}).get(qid) or {}
    label = str(ans.get("choice", "") or "").upper()
    try:
        conf = float(ans.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return label, conf


def _parse_noul(answers: Dict, qid: str) -> float:
    """P(true) for a noul answer, 0.0 when malformed."""
    ans = (answers or {}).get(qid) or {}
    try:
        return float(ans.get("noul", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _laya_route(
    label: str, confidence: float, has_docs: bool, threshold: float
) -> RouteDecision:
    """Gate the Laya label on confidence, else safe-default by docs.

    At or above threshold the Laya label wins (WEB has no serving branch
    yet, so it falls through to the docs default). Below threshold docs
    attached means RAG - empty retrieval fail-opens to DIRECT anyway -
    while no docs means DIRECT.
    """
    if label in ("DIRECT", "RAG") and confidence >= threshold:
        return RouteDecision(route=label, reason="laya")  # type: ignore[arg-type]
    if has_docs:
        return RouteDecision(route="RAG", reason="laya low-conf safe-default")
    return RouteDecision(route="DIRECT", reason="laya low-conf no docs")


EPISODIC_GATE_LIMIT = 10


class MemoryEvidence(NamedTuple):
    """Gate triple - overall flag plus per-store hits for the split."""

    needs_memory: bool
    sem_hit: bool
    epi_hit: bool


def _gate_episodic_rows(db, conversation_id, limit: int = EPISODIC_GATE_LIMIT):
    """Recent episodic rows for the gate, [] without chat or on error."""
    try:
        if db is None or conversation_id is None:
            return []
        return list(
            EpisodicMemoryRepository(db).list_recent_for_query(
                conversation_id, limit=limit
            )
            or []
        )
    except Exception as e:
        logger.debug("episodic gate skipped: %s", e)
        return []


def _episodic_overlap(query: str, epi_rows) -> bool:
    """True on token overlap with any recent summary, no vector."""
    toks = memory_tokens(query)
    if not toks:
        return False
    try:
        for row in epi_rows or []:
            if isinstance(row, str):
                text = row
            elif isinstance(row, dict):
                text = row.get("summary", "") or ""
            else:
                text = getattr(row, "summary", "") or ""
            if toks & memory_tokens(text):
                return True
    except Exception as e:
        logger.debug("episodic overlap skipped: %s", e)
        return False
    return False


def _stub_evidence(query, history, memories, db, conversation_id):
    """Stub sem hit plus cheap episodic overlap, no Laya call."""
    sem_hit = _needs_memory_stub(query, history, memories)
    epi_hit = _episodic_overlap(query, _gate_episodic_rows(db, conversation_id))
    return bool(sem_hit), bool(epi_hit)


def memory_evidence(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
    memories=None,
    db=None,
    conversation_id: Optional[uuid.UUID] = None,
    epi_rows=None,
) -> MemoryEvidence:
    """Gate triple - Laya first, stub plus episodic overlap below it.

    needs_memory is Laya True OR stub True OR episodic overlap True.
    sem_hit and epi_hit drive the mem_cap split in _combine_memory.
    Pass epi_rows to skip the pool fetch; no vector at gate time.
    """
    laya_hit = False
    try:
        state = {"query": query or "", "history": _history_text(history)}
        res = LayaService.get_instance().predict(
            state, {"needs_memory": NEEDS_MEMORY_QUESTION}
        )
        answers = (res or {}).get("answers") or {}
        laya_hit = (
            _parse_noul(answers, "needs_memory") >= config.LAYA_MEMORY_THRESHOLD
        )
    except Exception as e:
        logger.debug("laya memory gate skipped: %s", e)
    sem_hit = _needs_memory_stub(query, history, memories)
    if epi_rows is None:
        epi_rows = _gate_episodic_rows(db, conversation_id)
    epi_hit = _episodic_overlap(query, epi_rows)
    return MemoryEvidence(
        needs_memory=bool(laya_hit or sem_hit or epi_hit),
        sem_hit=bool(sem_hit),
        epi_hit=bool(epi_hit),
    )


def needs_memory(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
    memories=None,
    db=None,
    conversation_id: Optional[uuid.UUID] = None,
    epi_rows=None,
) -> bool:
    """Recall gate - Laya first, stub plus episodic overlap below it."""
    return bool(
        memory_evidence(
            query,
            history=history,
            memories=memories,
            db=db,
            conversation_id=conversation_id,
            epi_rows=epi_rows,
        ).needs_memory
    )


def _needs_memory_stub(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
    memories=None,
) -> bool:
    """Recall gate - keywords, history prefs, or semantic key overlap."""
    import re

    recent = list(history or [])[-2:]
    q = (query or "").lower()
    hist = " ".join(((t or {}).get("content") or "") for t in recent).lower()
    if any(re.search(p, f"{q} {hist}") for p in _MEMORY_PATTERNS):
        return True
    if hist and any(re.search(p, hist) for p in _HISTORY_PREF_PATTERNS):
        return True
    if memories:
        toks = memory_tokens(query)
        for m in memories:
            key = m.get("key", "") if isinstance(m, dict) else (getattr(m, "key", "") or "")
            if toks & memory_tokens(key):
                return True
    return False


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

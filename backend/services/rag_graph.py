"""Agentic chat orchestration - decide, retrieve, build. Sole RAG entry point.

Flow: query + history + conversation_id -> decide (DIRECT skips retrieval)
-> retrieve (scoped hybrid search) -> build (grounded or plain messages).
"""

import uuid
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session

from schemas.rag_schemas import RouteDecision, SearchHit
from config import config
from core.context_budget import top_k_for_ctx
from core.logging import get_logger
from services.decider import Decider, memory_evidence
from repository.semantic_repository import SemanticMemoryRepository, memory_embed_text
from services.llama_engine import EmbeddingEngine
from services.llm_service import LLMService
from services.rag_service import RagService, _fit_memory

logger = get_logger(__name__)


MEMORY_DOMINANT_SHARE = 0.7


def _memory_shares(
    sem_hit: bool, epi_hit: bool, cap: int
) -> Tuple[int, int, bool]:
    """(sem_cap, epi_cap, epi_first) split of mem_cap for gate evidence.

    Epi-only hit fits episodic first up to 70 pct, sem-only hit fits
    semantic first up to 70 pct; both/neither keeps semantic-first on
    the full cap. M5 topic buffer reuses this for its own carve.
    """
    cap = max(0, int(cap))
    dominant = int(cap * MEMORY_DOMINANT_SHARE)
    if epi_hit and not sem_hit:
        return (max(0, cap - dominant), dominant, True)
    if sem_hit and not epi_hit:
        return (dominant, max(0, cap - dominant), False)
    return (cap, cap, False)


def _combine_memory(
    sem_text: str,
    ep_texts: Optional[List[str]],
    sem_hit: bool = False,
    epi_hit: bool = False,
    cap: Optional[int] = None,
) -> str:
    """One memory block inside mem_cap, split per query evidence.

    Single-side input takes the full cap, as today. With both sides
    the dominant evidence fits first (up to 70 pct) and the other
    takes the remainder; both/neither keeps semantic-first.
    """
    limit = config.EFFECTIVE_MEMORY_TOKENS if cap is None else max(0, int(cap))
    sem_clean = (sem_text or "").strip()
    ep_lines = [
        f"earlier: {(s or '').strip()}" for s in ep_texts or [] if (s or "").strip()
    ]
    ep_joined = "\n".join(ep_lines).strip()
    if sem_clean and not ep_joined:
        return _fit_memory(sem_clean, limit)[0].strip()
    if ep_joined and not sem_clean:
        return _fit_memory(ep_joined, limit)[0].strip()
    if not sem_clean and not ep_joined:
        return ""
    sem_cap, epi_cap, epi_first = _memory_shares(sem_hit, epi_hit, limit)
    if epi_first:
        ep_block, epi_used = _fit_memory(ep_joined, epi_cap)
        sem_block, sem_used = _fit_memory(sem_clean, max(0, limit - epi_used))
    else:
        sem_block, sem_used = _fit_memory(sem_clean, sem_cap)
        ep_block, epi_used = _fit_memory(ep_joined, max(0, limit - sem_used))
    logger.debug(
        "memory split sem_hit=%s epi_hit=%s cap=%s sem_used=%s epi_used=%s",
        sem_hit,
        epi_hit,
        limit,
        sem_used,
        epi_used,
    )
    return "\n".join(p for p in (sem_block, ep_block) if p).strip()


class RagState(TypedDict, total=False):
    query: str
    history: List[Dict[str, str]]
    conversation_id: Optional[uuid.UUID]
    decision: RouteDecision
    hits: List[SearchHit]
    messages: List[Dict[str, str]]


class RagGraph:
    """Entry decider, then retrieve and build around one RagService."""

    def __init__(
        self,
        db: Session,
        rag: Optional[RagService] = None,
        decider: Optional[Decider] = None,
        llm: Optional[LLMService] = None,
        top_k: Optional[int] = None,
    ):
        # Shared embed slot per graph: unloaded after the run so only one
        # model is resident on the 8GB box. Injected fakes skip this.
        self._owns_engine = rag is None
        engine = EmbeddingEngine.get_instance("embed") if self._owns_engine else None
        self.rag = rag or RagService(db, engine=engine)
        self.decider = decider or Decider(db)
        self.llm = llm or LLMService()
        self.top_k = top_k if top_k is not None else top_k_for_ctx(config.EFFECTIVE_N_CTX)

        graph = StateGraph(RagState)
        graph.add_node("decide", self._decide)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("build", self._build)
        graph.set_entry_point("decide")
        graph.add_conditional_edges(
            "decide", self._route, {"rag": "retrieve", "direct": "build"}
        )
        graph.add_edge("retrieve", "build")
        graph.add_edge("build", END)
        self._app = graph.compile()

    def run(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        conversation_id: Optional[uuid.UUID] = None,
    ) -> Dict[str, Any]:
        """Run decider then retrieval/build; fail-open DIRECT on any error."""
        logger.debug("Graph run started - query='%.100s'", query)
        try:
            try:
                out = self._app.invoke(
                    {
                        "query": query,
                        "history": history or [],
                        "conversation_id": conversation_id,
                    }
                )
            finally:
                if self._owns_engine:
                    self.rag.engine.unload()
        except Exception as e:
            logger.warning("Graph failed - falling back to DIRECT: %s", e)
            return {
                "decision": RouteDecision(route="DIRECT", reason="graph fallback"),
                "hits": [],
                "messages": self.llm.build_chat_messages(history or [], query),
                "route": "DIRECT",
            }
        decision = out.get("decision") or RouteDecision(route="DIRECT", reason="empty")
        out["decision"] = decision
        out["route"] = decision.route
        logger.info(
            "RAG decision - route=%s, hits=%s, reason='%s'",
            decision.route,
            len(out.get("hits", [])),
            decision.reason,
        )
        return out

    def _decide(self, state: RagState) -> Dict[str, Any]:
        try:
            history = state.get("history") or []
            decision = self.decider.decide(
                state.get("query", ""),
                state.get("conversation_id"),
                history=history[-2:],
            )
        except Exception as e:
            logger.warning("Decider failed - falling back to DIRECT: %s", e)
            decision = RouteDecision(route="DIRECT", reason="decider error")

        return {"decision": decision}

    def _route(self, state: RagState) -> str:
        decision = state.get("decision")
        if decision is not None and decision.route == "RAG":
            return "rag"
        return "direct"

    def _retrieve(self, state: RagState) -> Dict[str, Any]:
        try:
            hits = self.rag.search(
                state.get("query", ""),
                top_k=self.top_k,
                conversation_id=state.get("conversation_id"),
            )
        except Exception as e:
            logger.warning("Retrieval failed - falling back to DIRECT: %s", e)
            return {
                "hits": [],
                "decision": RouteDecision(route="DIRECT", reason="retrieval error"),
            }

        decision = state.get("decision")

        if not hits and decision is not None and decision.route == "RAG":
            decision = RouteDecision(
                route="DIRECT", reason="no hits - ask with filename"
            )

        return {"hits": hits, "decision": decision}

    def _memory_text(
        self, query: str, conversation_id: Optional[uuid.UUID] = None
    ) -> Optional[str]:
        """Semantic facts plus episodic summaries in one block, None when none.

        Cheap overlap pools first; the embed engine stays untouched on a
        memory-less turn. Otherwise one batched embed (query plus capped
        candidates) feeds vector-blended recall; any embed failure falls
        back to overlap-only.
        """
        from services.episodic_service import EpisodicService

        db = self.rag.db
        try:
            sem_overlap = SemanticMemoryRepository(db).find_relevant(query, limit=5)
        except Exception as e:
            logger.debug("memory recall skipped: %s", e)
            sem_overlap = []
        try:
            sem_rows = list(SemanticMemoryRepository(db).list_all(limit=100) or [])
        except Exception as e:
            logger.debug("semantic pool skipped: %s", e)
            sem_rows = []
        try:
            if conversation_id is not None:
                epi_rows = list(
                    EpisodicService(db).repo.list_recent_for_query(
                        conversation_id, limit=10
                    )
                    or []
                )
            else:
                epi_rows = []
        except Exception as e:
            logger.debug("episodic pool skipped: %s", e)
            epi_rows = []
        evidence = memory_evidence(query, memories=sem_overlap, epi_rows=epi_rows)
        if not evidence.needs_memory and not epi_rows and not sem_rows:
            return None  # memory-less turn, engine untouched
        sem_items = []
        for row in sem_rows:
            key = getattr(row, "key", "") or ""
            text = memory_embed_text(f"{key}: {getattr(row, 'value', '') or ''}")
            if text.strip():
                sem_items.append((key, text))
        epi_items = []
        for row in epi_rows:
            text = memory_embed_text(getattr(row, "summary", "") or "")
            if text.strip():
                epi_items.append((str(getattr(row, "id", "")), text))
        vecs = None
        try:
            texts = (
                [query or ""]
                + [text for _, text in sem_items]
                + [text for _, text in epi_items]
            )
            vecs = self.rag.engine.embed(texts)
            if (
                not isinstance(vecs, list)
                or len(vecs) != len(texts)
                or not all(isinstance(v, list) and v for v in vecs)
            ):
                raise ValueError("embed shape mismatch")
        except Exception as e:
            logger.debug("memory embed skipped, overlap-only: %s", e)
            vecs = None
        if vecs is None:
            rows = sem_overlap
            ep_texts: List[str] = []
            if conversation_id is not None:
                try:
                    ep_texts = EpisodicService(db).recall(
                        conversation_id, query, limit=3
                    )
                except Exception as e:
                    logger.debug("episodic recall skipped, semantic-only: %s", e)
                    ep_texts = []
        else:
            query_vector = vecs[0]
            sem_vecs = {
                key: vec for (key, _), vec in zip(sem_items, vecs[1:])
            }
            epi_vecs = {
                key: vec
                for (key, _), vec in zip(epi_items, vecs[1 + len(sem_items):])
            }
            try:
                rows = SemanticMemoryRepository(db).find_relevant(
                    query,
                    limit=5,
                    query_vector=query_vector,
                    candidate_vectors=sem_vecs,
                )
            except Exception as e:
                logger.debug("memory recall skipped: %s", e)
                rows = sem_overlap
            ep_texts = []
            if conversation_id is not None:
                try:
                    ep_texts = EpisodicService(db).recall(
                        conversation_id,
                        query,
                        limit=3,
                        query_vector=query_vector,
                        candidate_vectors=epi_vecs,
                    )
                except Exception as e:
                    logger.debug("episodic recall skipped, semantic-only: %s", e)
                    ep_texts = []
        lines = [
            f"{r.key}: {r.value}"
            for r in rows or []
            if (getattr(r, "value", "") or "").strip()
        ]
        sem_text = "\n".join(lines).strip()
        text = _combine_memory(
            sem_text, ep_texts, sem_hit=evidence.sem_hit, epi_hit=evidence.epi_hit
        )
        return text or None

    def _build(self, state: RagState) -> Dict[str, List[Dict[str, str]]]:
        """Grounded or plain messages; memory keeps its carve first.

        Effective priority order: memory carve (sem/epi split per gate
        evidence in _combine_memory) first, then RAG hits, then history.
        Route RAG with hits builds grounded messages, otherwise plain
        chat. allocate() signature unchanged - needs_memory follows the
        memory block, so strong memory evidence never loses its carve
        to RAG hits.
        """
        query = state.get("query", "")
        history = state.get("history", [])
        decision = state.get("decision")
        hits = state.get("hits", [])
        memory_text = self._memory_text(query, state.get("conversation_id"))

        if decision is not None and decision.route == "RAG" and hits:
            messages = self.rag.build_messages(
                query, hits, history, memory_text=memory_text
            )
        else:
            messages = self.rag.build_messages(
                query, None, history, memory_text=memory_text
            )
        return {"messages": messages}

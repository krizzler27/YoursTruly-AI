"""Agentic chat orchestration - decide, retrieve, build. Sole RAG entry point.

Flow: query + history + conversation_id -> decide (DIRECT skips retrieval)
-> retrieve (scoped hybrid search) -> build (grounded or plain messages).
"""

import uuid
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session

from schemas.rag_schemas import RouteDecision
from core.logging import get_logger
from services.decider import Decider
from services.llama_engine import EmbeddingEngine
from services.llm_service import LLMService
from services.rag_service import RagService

logger = get_logger(__name__)


class RagState(TypedDict, total=False):
    query: str
    history: List[Dict[str, str]]
    conversation_id: Optional[uuid.UUID]
    decision: RouteDecision
    hits: List[Dict[str, Any]]
    messages: List[Dict[str, str]]


class RagGraph:
    """Entry decider, then retrieve and build around one RagService."""

    def __init__(
        self,
        db: Session,
        rag: Optional[RagService] = None,
        decider: Optional[Decider] = None,
        llm: Optional[LLMService] = None,
        top_k: int = 5,
    ):
        # Fresh embedder per graph: unloaded after the run so only one
        # model is resident on the 8GB box. Injected fakes skip this.
        self._owns_engine = rag is None
        engine = EmbeddingEngine() if self._owns_engine else None
        self.rag = rag or RagService(db, engine=engine)
        self.decider = decider or Decider(db)
        self.llm = llm or LLMService()
        self.top_k = top_k

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
            decision = self.decider.decide(
                state.get("query", ""), state.get("conversation_id")
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

    def _build(self, state: RagState) -> Dict[str, List[Dict[str, str]]]:
        query = state.get("query", "")
        history = state.get("history", [])
        decision = state.get("decision")
        hits = state.get("hits", [])

        if decision is not None and decision.route == "RAG" and hits:
            messages = self.rag.build_messages(query, hits, history)
        else:
            messages = self.llm.build_chat_messages(history, query)
        return {"messages": messages}

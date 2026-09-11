"""RAG branch — entry Decider routes here for local-document questions.

DIRECT answers skip retrieval; RAG fuses hybrid search into a budgeted
grounded prompt. Output is messages only; chat wiring is Phase 4.
"""

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session

from schemas.rag_schemas import RouteDecision
from services.decider import Decider
from services.llm_service import LLMService
from services.rag_service import RagService


class RagState(TypedDict, total=False):
    query: str
    history: List[Dict[str, str]]
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
        self.rag = rag or RagService(db)
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
        self, query: str, history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Run decider then retrieval/build; returns decision, hits, messages."""
        return self._app.invoke({"query": query, "history": history or []})

    def _decide(self, state: RagState) -> Dict[str, RouteDecision]:
        return {"decision": self.decider.decide(state.get("query", ""))}

    def _route(self, state: RagState) -> str:
        decision = state.get("decision")
        if decision is not None and decision.route == "RAG":
            return "rag"
        return "direct"

    def _retrieve(self, state: RagState) -> Dict[str, List[Dict[str, Any]]]:
        return {"hits": self.rag.search(state.get("query", ""), top_k=self.top_k)}

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

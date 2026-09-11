"""Agentic entry decider — which capability serves this query.

Routes today: DIRECT (answer from the model) and RAG (local documents).
WEB and other capabilities slot in as new route values plus branches.
Graders (hits, grounding, answer) land here as knobs when needed.
"""

from typing import Optional
import uuid

from sqlalchemy.orm import Session

from repository.document_repository import DocumentRepository
from schemas.rag_schemas import RouteDecision
from services.llm_service import LLMService
from services.prompt_manager import PromptManager


class Decider:
    """Entry decider — mechanical guards plus one structured SLM call."""

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
        self, query: str, conversation_id: Optional[uuid.UUID] = None
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

        inventory = "\n".join(
            f"- {d.filename}" + (f": {d.summary}" if (d.summary or "").strip() else "")
            for d in attached
        )

        messages = [
            {
                "role": "system",
                "content": PromptManager.render(
                    "rag_gate.j2", current_query=clean, attached_docs=inventory
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
            return result
        except RuntimeError:
            return RouteDecision(route="RAG", reason="decider fallback")

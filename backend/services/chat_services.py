from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
from datetime import datetime, timezone

from db.models import ConversationsModel, MessagesModel, SemanticMemoryModel
from core.logging import get_logger
from core.trace import traceable
from repository.chat_repository import ChatRepository
from repository.message_repository import MessageRepository
from services.rag_graph import RagGraph
from services.rag_service import RagService

logger = get_logger(__name__)


class ChatServices:
    """Orchestrates conversation + message persistence for chat/stream."""

    def __init__(self, db: Session):
        self.db = db
        self.chat_repo = ChatRepository(db)
        self.message_repo = MessageRepository(db)

    def ensure_conversation(
        self, conversation_id: Optional[uuid.UUID], title: Optional[str] = None
    ) -> ConversationsModel:
        """Return existing conversation or create new when id is None."""
        if conversation_id is None:
            clean_title = (title or "New chat").strip()[:32] or "New chat"
            conv = self.chat_repo.create(title=clean_title)
            logger.info("Conversation created - id=%s", conv.id)
            return conv
        conv = self.chat_repo.get_by_id(conversation_id)
        if conv is None:
            raise ValueError(f"Conversation {conversation_id} not found")
        clean = (title or "").strip()
        if clean and (conv.title or "").strip() in ("", "New chat"):
            self.chat_repo.update(conversation_id, title=clean[:32])
            conv = self.chat_repo.get_by_id(conversation_id)
        return conv

    def add_message(
        self, conversation_id: uuid.UUID, role: str, content: str
    ) -> MessagesModel:
        msg = self.message_repo.create(
            conversation_id=conversation_id, role=role, content=content
        )
        self.chat_repo.update(conversation_id, updated_at=datetime.now(timezone.utc))
        return msg

    def rename_conversation(self, conversation_id: uuid.UUID, title: str) -> ConversationsModel:
        """Rename conversation, storing full title up to 100 chars."""
        clean = (title or "").strip()
        if not clean:
            raise ValueError("Title cannot be empty")
        conv = self.chat_repo.update(conversation_id, title=clean)
        if conv is None:
            raise ValueError(f"Conversation {conversation_id} not found")
        return conv

    def set_tag(
        self, conversation_id: uuid.UUID, tag: Optional[str]
    ) -> ConversationsModel:
        """Tag a conversation, None or blank clears the tag."""
        conv = self.chat_repo.set_tag(conversation_id, tag)
        if conv is None:
            raise ValueError(f"Conversation {conversation_id} not found")
        logger.info("Conversation tag set - id=%s tag=%s", conv.id, conv.tag)
        return conv

    def set_project(
        self, conversation_id: uuid.UUID, project_id: Optional[uuid.UUID]
    ) -> ConversationsModel:
        """Link a conversation to a project, None clears the link."""
        conv = self.chat_repo.set_project(conversation_id, project_id)
        if conv is None:
            raise ValueError(f"Conversation {conversation_id} not found")
        logger.info("Conversation project set - id=%s project=%s", conv.id, conv.project_id)
        return conv

    def maybe_remember(self, query: str) -> Optional[SemanticMemoryModel]:
        """Store one explicit fact off the chat save path, None when skipped."""
        try:
            from services.semantic_writer import remember_explicit

            return remember_explicit(self.db, query)
        except Exception as e:
            logger.debug("maybe_remember skipped: %s", e)
            return None

    def delete_conversation(self, conversation_id: uuid.UUID) -> None:
        """Delete conversation, its messages (cascade) and its attached docs."""
        rag = RagService(self.db)
        for doc in rag.docs.list_by_conversation(conversation_id, limit=1000):
            try:
                rag.delete_document(doc.id)
            except ValueError:
                pass  # already gone
        try:
            from repository.episodic_repository import EpisodicMemoryRepository

            EpisodicMemoryRepository(self.db).delete_by_conversation(conversation_id)
        except Exception as e:
            logger.debug("episodic cleanup skipped: %s", e)
        ok = self.chat_repo.delete(conversation_id)
        if not ok:
            raise ValueError(f"Conversation {conversation_id} not found")

    def maybe_rollup(self, conversation_id: uuid.UUID) -> Optional[str]:
        """Sync episodic write; fail-open, never raises.

        Foreground-safe only on the cheap no-trigger path (history<=2 fast
        return). A triggered turn runs a full SLM summarize inline, so the
        request path prefers RollupJob.submit; kept for tests/back-compat.
        """
        try:
            history = self.get_history(conversation_id, limit=100)
            if len(history) <= 2:
                return None
            from core.context_budget import allocate
            from services.episodic_service import EpisodicService

            usable = allocate(route="DIRECT", needs_memory=False, query_tokens=0)["usable"]
            return EpisodicService(self.db).rollup_if_needed(
                conversation_id, history, usable
            )
        except Exception as e:
            logger.debug("episodic rollup skipped: %s", e)
            return None

    def get_history(self, conversation_id: uuid.UUID, limit: int = 5) -> List[dict]:
        """Return last `limit` messages as LLM-ready dicts."""
        rows = self.message_repo.list_by_conversation(conversation_id, limit=limit)
        return [{"role": r.role, "content": r.content} for r in rows]

    def list_conversations(self, limit: int = 50) -> List[ConversationsModel]:
        return self.chat_repo.list_recent(limit=limit)

    def list_messages(self, conversation_id: uuid.UUID, limit: int = 100) -> List[MessagesModel]:
        rows = self.message_repo.list_by_conversation(conversation_id, limit=limit)
        # ensure existence check
        if not self.chat_repo.get_by_id(conversation_id):
            raise ValueError(f"Conversation {conversation_id} not found")
        return rows

    @traceable(name="run_agentic")
    def run_agentic(
        self, query: str, history: List[dict], conversation_id: uuid.UUID
    ) -> dict:
        """Single entry to the agent graph (owns decide/retrieve/build)."""
        logger.debug("Chat dispatch started - query='%.100s'", query)
        out = RagGraph(self.db).run(query, history, conversation_id)
        logger.info("Chat ready - route=%s, hits=%s", out.get("route"), len(out.get("hits", [])))
        return out
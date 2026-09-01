from typing import Optional, List
import uuid

from sqlalchemy.orm import Session

from db.models import ConversationsModel, MessagesModel
from repository.chat_repository import ChatRepository
from repository.message_repository import MessageRepository

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
            return self.chat_repo.create(title=clean_title)
        conv = self.chat_repo.get_by_id(conversation_id)
        if conv is None:
            raise ValueError(f"Conversation {conversation_id} not found")
        return conv

    def add_message(
        self, conversation_id: uuid.UUID, role: str, content: str
    ) -> MessagesModel:
        return self.message_repo.create(
            conversation_id=conversation_id, role=role, content=content
        )

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
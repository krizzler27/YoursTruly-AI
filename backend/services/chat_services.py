from sqlalchemy.orm import Session
from typing import Optional, List
import uuid
from datetime import datetime, timezone

from db.models import ConversationsModel, MessagesModel
from repository.chat_repository import ChatRepository
from repository.message_repository import MessageRepository
from repository.lance_repository import LanceRepository
from services.decider import Decider
from services.llama_engine import EmbeddingEngine
from services.llm_service import LLMService
from services.rag_service import RagService

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

    def delete_conversation(self, conversation_id: uuid.UUID) -> None:
        """Delete conversation, its messages (cascade) and its attached docs."""
        rag = RagService(self.db)
        for doc in rag.docs.list_by_conversation(conversation_id, limit=1000):
            try:
                rag.delete_document(doc.id)
            except ValueError:
                pass  # already gone
        ok = self.chat_repo.delete(conversation_id)
        if not ok:
            raise ValueError(f"Conversation {conversation_id} not found")

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

    def prepare_messages(
        self,
        query: str,
        history: List[dict],
        conversation_id: uuid.UUID,
        llm: LLMService,
    ) -> tuple[List[dict], str]:
        """Decide DIRECT vs RAG and build the message list; fail-open DIRECT."""
        try:
            decision = Decider(self.db, llm=llm).decide(query, conversation_id).route
        except Exception as e:
            print(f"[RAG] decider failed, DIRECT: {e}")
            return LLMService.build_chat_messages(history or [], query), "DIRECT"

        if decision != "RAG":
            return LLMService.build_chat_messages(history or [], query), "DIRECT"

        try:
            return self.retrieve_grounded(query, history, conversation_id), "RAG"
        except Exception as e:
            print(f"[RAG] retrieval failed, DIRECT: {e}")
            return LLMService.build_chat_messages(history or [], query), "DIRECT"

    def retrieve_grounded(
        self, query: str, history: List[dict], conversation_id: uuid.UUID, top_k: int = 5
    ) -> List[dict]:
        """Embed one query, scoped search, budgeted ground; embedder unloaded after."""
        embedder = EmbeddingEngine()
        try:
            embedder.load()  # brief co-residency with chat; unloaded in finally
            rag = RagService(self.db, engine=embedder, lance=LanceRepository(self.db))
            hits = rag.search(query, top_k=top_k, conversation_id=conversation_id)
            return rag.build_messages(query, hits, history or [])
        finally:
            embedder.unload()
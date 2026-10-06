from typing import List, Optional

from sqlalchemy.orm import Session

from db.models import ConversationsModel
from core.logging import get_logger
from repository.base_repository import BaseRepository

logger = get_logger(__name__)

TOPIC_MAX_LENGTH = 64


def normalize_topic(topic: Optional[str]) -> Optional[str]:
    """Strip and cap a topic tag, blank or None clears to None."""
    if topic is None:
        return None
    clean = topic.strip()
    if not clean:
        return None
    return clean[:TOPIC_MAX_LENGTH]


class ChatRepository(BaseRepository[ConversationsModel]):

    def __init__(self, db: Session):
        super().__init__(ConversationsModel, db)

    def list_recent(self, limit: int = 50) -> List[ConversationsModel]:
        """List conversations newest first"""
        return (
            self.db.query(self.model)
            .order_by(self.model.updated_at.desc(), self.model.id.desc())
            .limit(limit)
            .all()
        )

    def set_topic(
        self, conversation_id, topic: Optional[str]
    ) -> Optional[ConversationsModel]:
        """Tag a conversation, None or blank clears the tag."""
        conv = self.get_by_id(conversation_id)
        if conv is None:
            return None
        conv.topic = normalize_topic(topic)
        self.db.commit()
        self.db.refresh(conv)
        return conv

    def list_by_topic(
        self, topic: str, limit: int = 10, exclude_id=None
    ) -> List[ConversationsModel]:
        """Sibling chats sharing one topic, newest first, minus one id."""
        clean = normalize_topic(topic)
        if not clean:
            return []
        q = self.db.query(self.model).filter(self.model.topic == clean)
        if exclude_id is not None:
            q = q.filter(self.model.id != exclude_id)
        return (
            q.order_by(self.model.updated_at.desc(), self.model.id.desc())
            .limit(max(1, int(limit)))
            .all()
        )
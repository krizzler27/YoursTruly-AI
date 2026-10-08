from typing import List, Optional

from sqlalchemy.orm import Session

from db.models import ConversationsModel
from core.logging import get_logger
from repository.base_repository import BaseRepository

logger = get_logger(__name__)

TAG_MAX_LENGTH = 64
TOPIC_MAX_LENGTH = TAG_MAX_LENGTH


def normalize_tag(tag: Optional[str]) -> Optional[str]:
    """Strip and cap a tag, blank or None clears to None."""
    if tag is None:
        return None
    clean = tag.strip()
    if not clean:
        return None
    return clean[:TAG_MAX_LENGTH]


def normalize_topic(topic: Optional[str]) -> Optional[str]:
    """Deprecated alias for normalize_tag."""
    return normalize_tag(topic)


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

    def set_tag(
        self, conversation_id, tag: Optional[str]
    ) -> Optional[ConversationsModel]:
        """Tag a conversation, None or blank clears the tag."""
        conv = self.get_by_id(conversation_id)
        if conv is None:
            return None
        conv.tag = normalize_tag(tag)
        self.db.commit()
        self.db.refresh(conv)
        return conv

    def set_topic(
        self, conversation_id, topic: Optional[str]
    ) -> Optional[ConversationsModel]:
        """Deprecated alias for set_tag."""
        return self.set_tag(conversation_id, topic)

    def list_by_tag(
        self, tag: str, limit: int = 10, exclude_id=None
    ) -> List[ConversationsModel]:
        """Sibling chats sharing one tag, newest first, minus one id."""
        clean = normalize_tag(tag)
        if not clean:
            return []
        q = self.db.query(self.model).filter(self.model.tag == clean)
        if exclude_id is not None:
            q = q.filter(self.model.id != exclude_id)
        return (
            q.order_by(self.model.updated_at.desc(), self.model.id.desc())
            .limit(max(1, int(limit)))
            .all()
        )

    def list_by_topic(
        self, topic: str, limit: int = 10, exclude_id=None
    ) -> List[ConversationsModel]:
        """Deprecated alias for list_by_tag."""
        return self.list_by_tag(topic, limit=limit, exclude_id=exclude_id)

    def set_project(
        self, conversation_id, project_id
    ) -> Optional[ConversationsModel]:
        """Link a conversation to a project, None clears the link."""
        conv = self.get_by_id(conversation_id)
        if conv is None:
            return None
        conv.project_id = project_id
        self.db.commit()
        self.db.refresh(conv)
        return conv

    def list_by_project(
        self, project_id, limit: int = 50, exclude_id=None
    ) -> List[ConversationsModel]:
        """Member chats of one project, newest first, minus one id."""
        if project_id is None:
            return []
        q = self.db.query(self.model).filter(self.model.project_id == project_id)
        if exclude_id is not None:
            q = q.filter(self.model.id != exclude_id)
        return (
            q.order_by(self.model.updated_at.desc(), self.model.id.desc())
            .limit(max(1, int(limit)))
            .all()
        )

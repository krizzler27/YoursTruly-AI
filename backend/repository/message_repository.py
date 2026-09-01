from typing import List
import uuid

from sqlalchemy.orm import Session

from db.models import MessagesModel
from repository.base_repository import BaseRepository

class MessageRepository(BaseRepository[MessagesModel]):

    def __init__(self, db: Session):
        super().__init__(MessagesModel, db)

    def list_by_conversation(
        self, conversation_id: uuid.UUID, limit: int = 5
    ) -> List[MessagesModel]:
        """Return last `limit` messages for a conversation, oldest first."""
        rows = (
            self.db.query(self.model)
            .filter(self.model.conversation_id == conversation_id)
            .order_by(self.model.created_at.desc(), self.model.id.desc())
            .limit(limit)
            .all()
        )
        return list(reversed(rows))
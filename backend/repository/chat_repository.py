from typing import List

from sqlalchemy.orm import Session

from db.models import ConversationsModel
from repository.base_repository import BaseRepository

class ChatRepository(BaseRepository[ConversationsModel]):

    def __init__(self, db: Session):
        super().__init__(ConversationsModel, db)

    def list_recent(self, limit: int = 50) -> List[ConversationsModel]:
        """List conversations newest first."""
        return (
            self.db.query(self.model)
            .order_by(self.model.created_at.desc(), self.model.id.desc())
            .limit(limit)
            .all()
        )
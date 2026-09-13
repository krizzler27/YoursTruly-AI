from typing import List

from sqlalchemy.orm import Session

from db.models import DocumentsModel
from repository.base_repository import BaseRepository


PENDING_PREFIX = "__pending__"  # internal ingest rows, never listed


class DocumentRepository(BaseRepository[DocumentsModel]):
    """CRUD over ingested file rows — retrieval hydration reads chunks."""

    def __init__(self, db: Session):
        super().__init__(DocumentsModel, db)

    def list_recent(self, limit: int = 100) -> List[DocumentsModel]:
        """List documents newest first, excluding internal pending rows."""
        return (
            self.db.query(self.model)
            .filter(~self.model.filename.startswith(PENDING_PREFIX))
            .order_by(self.model.created_at.desc(), self.model.id.desc())
            .limit(limit)
            .all()
        )

    def list_by_conversation(
        self, conversation_id, limit: int = 100
    ) -> List[DocumentsModel]:
        """List one chat's attached files newest first, excluding pending."""
        return (
            self.db.query(self.model)
            .filter(self.model.conversation_id == conversation_id)
            .filter(~self.model.filename.startswith(PENDING_PREFIX))
            .order_by(self.model.created_at.desc(), self.model.id.desc())
            .limit(limit)
            .all()
        )

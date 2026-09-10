from typing import List

from sqlalchemy.orm import Session

from db.models import DocumentsModel
from repository.base_repository import BaseRepository


class DocumentRepository(BaseRepository[DocumentsModel]):
    """CRUD over ingested file rows — retrieval hydration reads chunks."""

    def __init__(self, db: Session):
        super().__init__(DocumentsModel, db)

    def list_recent(self, limit: int = 100) -> List[DocumentsModel]:
        """List documents newest first."""
        return (
            self.db.query(self.model)
            .order_by(self.model.created_at.desc(), self.model.id.desc())
            .limit(limit)
            .all()
        )

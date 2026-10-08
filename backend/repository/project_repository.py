from typing import List, Optional
import uuid

from sqlalchemy.orm import Session

from db.models import ProjectsModel
from repository.base_repository import BaseRepository

PROJECT_NAME_MAX_LENGTH = 64


def normalize_project_name(name: Optional[str]) -> Optional[str]:
    """Strip and cap a project name, blank or None maps to None."""
    if name is None:
        return None
    clean = name.strip()
    if not clean:
        return None
    return clean[:PROJECT_NAME_MAX_LENGTH]


class ProjectRepository(BaseRepository[ProjectsModel]):

    def __init__(self, db: Session):
        super().__init__(ProjectsModel, db)

    def get_by_name(self, name: str) -> Optional[ProjectsModel]:
        """One project by exact name, None when absent."""
        clean = normalize_project_name(name)
        if not clean:
            return None
        return self.db.query(self.model).filter(self.model.name == clean).first()

    def get_or_create(self, name: str) -> ProjectsModel:
        """Return the named project, creating it with an empty summary."""
        clean = normalize_project_name(name)
        if not clean:
            raise ValueError("project name cannot be empty")
        row = self.get_by_name(clean)
        if row is None:
            row = self.model(name=clean, summary="")
            self.db.add(row)
            self.db.commit()
            self.db.refresh(row)
        return row

    def update_summary(self, project_id: uuid.UUID, summary: str) -> Optional[ProjectsModel]:
        """Replace one project summary, None when absent."""
        row = self.get_by_id(project_id)
        if row is None:
            return None
        row.summary = (summary or "").strip()
        self.db.commit()
        self.db.refresh(row)
        return row

    def list_all(self, limit: int = 100) -> List[ProjectsModel]:
        """All projects newest first, capped."""
        return (
            self.db.query(self.model)
            .order_by(self.model.updated_at.desc(), self.model.id.desc())
            .limit(max(1, int(limit)))
            .all()
        )

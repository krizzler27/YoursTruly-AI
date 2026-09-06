from typing import TypeVar, Generic, Type, Optional, List
import uuid

from sqlalchemy.orm import Session
from db.models import Base

# Type variable bound to SQLAlchemy Base models
ModelType = TypeVar("ModelType", bound=Base)

class BaseRepository(Generic[ModelType]):
    """Generic repository containing common CRUD operations for all tables."""
    
    def __init__(self, model: Type[ModelType], db: Session):
        self.model = model
        self.db = db

    def get_by_id(self, id: uuid.UUID) -> Optional[ModelType]:
        return self.db.query(self.model).filter(self.model.id == id).first()

    def list_all(self, skip: int = 0, limit: int = 100) -> List[ModelType]:
        return self.db.query(self.model).offset(skip).limit(limit).all()

    def create(self, **data) -> ModelType:
        obj = self.model(**data)
        self.db.add(obj)
        self.db.commit()
        self.db.refresh(obj)
        return obj

    def create_many(self, items_data: List[dict]) -> List[ModelType]:
        """Inserts multiple records in a single database transaction."""
        objects = [self.model(**data) for data in items_data]
        self.db.add_all(objects)
        self.db.commit()
        return objects


    def update(self, id: uuid.UUID, **data) -> Optional[ModelType]:
        obj = self.get_by_id(id)
        if not obj:
            return None
        for key, value in data.items():
            setattr(obj, key, value)
        self.db.commit()
        self.db.refresh(obj)
        return obj

    def update_many(self, ids: List[uuid.UUID], **values) -> int:
        """Updates specific fields for multiple IDs in 1 query."""
        rows_updated = (
            self.db.query(self.model)
            .filter(self.model.id.in_(ids))
            .update(values, synchronize_session=False)
        )
        self.db.commit()
        return rows_updated

    def delete(self, id: uuid.UUID) -> bool:
        obj = self.get_by_id(id)
        if not obj:
            return False
        self.db.delete(obj)
        self.db.commit()
        return True

    def delete_many(self, ids: List[uuid.UUID]) -> int:
        """Deletes multiple records matching a list of IDs in 1 query."""
        rows_deleted = (
            self.db.query(self.model)
            .filter(self.model.id.in_(ids))
            .delete(synchronize_session=False) # synchronize_session execs directly in db and ignores session memory
        )
        self.db.commit()
        return rows_deleted


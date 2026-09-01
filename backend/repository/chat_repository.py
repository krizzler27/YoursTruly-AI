from sqlalchemy.orm import Session

from db.models import ConversationsModel
from repository.base_repository import BaseRepository

class ChatRepository(BaseRepository[ConversationsModel]):

    def __init__(self, db: Session):
        super().__init__(ConversationsModel, db)
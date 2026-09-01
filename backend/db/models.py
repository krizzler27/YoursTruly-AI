from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from sqlalchemy import func, ForeignKey
from datetime import datetime, timezone
import uuid
import uuid6

class Base(DeclarativeBase):
    pass

class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=func.now()
    )

class ConversationsModel(Base, TimestampMixin):

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid6.uuid7        
    )
    title: Mapped[str]

class MessagesModel(Base, TimestampMixin):

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid6.uuid7        
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(index=True)
    content: Mapped[str] = mapped_column()
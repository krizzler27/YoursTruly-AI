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

class DocumentsModel(Base, TimestampMixin):
    """Ingested source files — one row per file, tracks index status."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid6.uuid7
    )
    filename: Mapped[str] = mapped_column(unique=True, index=True)
    chunk_count: Mapped[int] = mapped_column(default=0)
    status: Mapped[str] = mapped_column(default="pending")  # pending|indexed


class DocumentChunksModel(Base):
    """Chunk text source of truth. String PK shared with the LanceDB row id."""

    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    index: Mapped[int] = mapped_column(default=0)
    heading: Mapped[str] = mapped_column(default="")
    page: Mapped[int | None] = mapped_column(default=None)
    text: Mapped[str] = mapped_column()
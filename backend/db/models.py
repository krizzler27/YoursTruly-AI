from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from sqlalchemy import String, Text, func, ForeignKey, Index, UniqueConstraint
from datetime import datetime, timezone
from typing import Optional
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
    tag: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, default=None, index=True
    )
    project_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL"),
        nullable=True, default=None, index=True,
    )


class ProjectsModel(Base, TimestampMixin):
    """Shared project state - one summary row per project name."""

    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid6.uuid7
    )
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    summary: Mapped[str] = mapped_column(default="")

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
    """Ingested source files - one row per file, tracks index status."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid6.uuid7
    )
    filename: Mapped[str] = mapped_column(index=True)
    conversation_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=True, index=True
    )
    summary: Mapped[str] = mapped_column(default="")
    chunk_count: Mapped[int] = mapped_column(default=0)
    status: Mapped[str] = mapped_column(default="pending")  # pending|indexed

    __table_args__ = (
        UniqueConstraint("filename", "conversation_id"),
    )


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


class SemanticMemoryModel(Base, TimestampMixin):
    """Durable user facts - name, prefs, shared across chats."""

    __tablename__ = "semantic_memory"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid6.uuid7
    )
    key: Mapped[str] = mapped_column(index=True, unique=True)
    value: Mapped[str] = mapped_column(default="")
    embedding: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, default=None
    )


class EpisodicMemoryModel(Base, TimestampMixin):
    """Summarized older turns per chat - episodic recall window."""

    __tablename__ = "episodic_memory"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid6.uuid7
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    summary: Mapped[str] = mapped_column(default="")
    turn_start: Mapped[int] = mapped_column(default=0)
    turn_end: Mapped[int] = mapped_column(default=0)
    recall_count: Mapped[int] = mapped_column(default=0)
    last_recalled_at: Mapped[Optional[datetime]] = mapped_column(nullable=True, default=None)

    __table_args__ = (
        Index("ix_episodic_memory_conversation_created", "conversation_id", "created_at"),
    )
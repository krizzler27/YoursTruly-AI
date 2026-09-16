import uuid
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Chunk(BaseModel):
    """Unit flowing through load → split → embed → stores."""

    text: str
    heading: str = ""
    index: int = 0
    page: Optional[int] = None
    vector: List[float] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def check_text(cls, v: str) -> str:
        v = v.strip()

        if not v:
            raise ValueError("chunk text cannot be empty")

        return v


class SearchRequest(BaseModel):
    """POST /api/search body - query plus result budget."""

    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    conversation_id: Optional[uuid.UUID] = None

    @field_validator("query")
    @classmethod
    def check_query(cls, v: str) -> str:
        v = (v or "").strip()

        if not v:
            raise ValueError("query cannot be empty")

        return v


class SearchHit(BaseModel):
    """One fused retrieval hit, hydrated from sqlite."""

    id: str
    document_id: str
    index: int = 0
    heading: str = ""
    page: Optional[int] = None
    text: str
    score: float = 0.0

    model_config = {"from_attributes": True}


class DocumentResponse(BaseModel):
    """Ingested file row - CRUD view over DocumentsModel."""

    id: uuid.UUID
    filename: str
    chunk_count: int = 0
    status: str = "pending"
    conversation_id: Optional[uuid.UUID] = None
    summary: str = ""
    created_at: datetime
    updated_at: datetime
    queue_depth: int = 0  # jobs ahead in the ingest queue (202 only)

    model_config = {"from_attributes": True}


class RouteDecision(BaseModel):
    """Agentic entry decision - DIRECT answers from chat, RAG via local docs (WEB later)."""

    route: Literal["DIRECT", "RAG"]
    reason: str = ""

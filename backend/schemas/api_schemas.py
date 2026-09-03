from pydantic import BaseModel, field_validator
from typing import Optional, List
from datetime import datetime
import uuid

class ChatRequest(BaseModel):
    query: str
    context: Optional[List[str]] = None
    model: Optional[str] = None
    conversation_id: Optional[uuid.UUID] = None

    @field_validator("query")
    @classmethod
    def check_query(cls, v):
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v.strip()

    @field_validator("model")
    @classmethod
    def check_model(cls, v):
        if v is not None and not v.strip():
            raise ValueError("model cannot be empty string")
        return v.strip() if v else None


class ConversationResponse(BaseModel):
    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class MessageResponse(BaseModel):
    id: uuid.UUID
    conversation_id: uuid.UUID
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ModelDownloadRequest(BaseModel):
    repo_id: str
    filename: Optional[str] = None
    quant: Optional[str] = None

    @field_validator("repo_id")
    @classmethod
    def check_repo(cls, v):
        if not v or not v.strip():
            raise ValueError("repo_id cannot be empty")
        v = v.strip()
        # allow HF repo, search query, or known name
        return v

    @field_validator("filename")
    @classmethod
    def check_filename(cls, v):
        if v is None or v == "":
            return v
        v = v.strip()
        if "/" in v or "\\" in v or ".." in v:
            raise ValueError("Invalid filename")
        if not v.lower().endswith(".gguf"):
            raise ValueError("filename must be .gguf")
        return v

    @field_validator("quant")
    @classmethod
    def check_quant(cls, v):
        if v is None or v == "":
            return v
        v = v.strip().upper()
        return v
from pydantic import BaseModel, field_validator
from typing import Optional, List
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
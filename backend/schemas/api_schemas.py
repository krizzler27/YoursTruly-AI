from pydantic import BaseModel, field_validator, Field
from typing import Optional, List, Literal
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


# ---- llmfit catalog / recommend ----

SORT_ALIASES = {"vram": "mem", "speed": "tps", "ram": "mem", "memory": "mem"}
SORT_VALUES = {"score", "tps", "mem"}

UseCaseField = Literal["general", "coding", "reasoning", "chat", "multimodal"]

class CatalogRequest(BaseModel):
    """POST /api/catalog - replaces GET query with typed body. Defaults = trusted-only, perfect only."""

    limit: int = Field(default=20, ge=1, le=100, description="max models returned")
    sort: str = Field(default="score", description="score|tps|mem (vram alias supported)")
    providers: Optional[List[str]] = Field(default=None, description="null = trusted default, [] = trusted (shim), [meta,google] = filter")
    perfect_only: bool = Field(default=True)
    include_community: bool = Field(default=False)
    search: Optional[str] = Field(default=None, description="client-side fallback, not sent to llmfit")

    @field_validator("sort", mode="before")
    @classmethod
    def normalize_sort(cls, v) -> str:
        if not v:
            return "score"
        s = str(v).strip().lower()
        s = SORT_ALIASES.get(s, s)
        if s not in SORT_VALUES:
            raise ValueError("sort must be one of: score, tps, mem (vram alias)")
        return s

    @field_validator("providers", mode="after")
    @classmethod
    def normalize_providers(cls, v):
        if v is None:
            return None
        out = [p.strip().lower() for p in v if p and p.strip()]
        # [] is treated as null (trusted) by service - keep as [] for explicit shim
        return out

    @field_validator("search", mode="after")
    @classmethod
    def normalize_search(cls, v):
        if v is None or not v.strip():
            return None
        return v.strip()


class RecommendRequest(CatalogRequest):
    """POST /api/recommend - same as catalog + use_case filter (embedding excluded)."""

    use_case: Optional[UseCaseField] = Field(default=None, description="general|coding|reasoning|chat|multimodal")
    min_fit: Literal["perfect", "good", "marginal"] = Field(default="marginal")
    runtime: Literal["any", "mlx", "llamacpp"] = Field(default="any")


class QuantsRequest(BaseModel):
    model: str = Field(description="HF repo, ollama name, or llmfit catalog name e.g. google/gemma-3-4b-it or gemma3:4b")

    @field_validator("model")
    @classmethod
    def check_model(cls, v):
        if not v or not v.strip():
            raise ValueError("model cannot be empty")
        return v.strip()
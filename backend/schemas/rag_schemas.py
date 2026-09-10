from typing import List, Optional

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

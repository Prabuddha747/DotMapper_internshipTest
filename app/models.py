"""Pydantic request schemas. No response models: each query_engine operation
returns a different "data" shape, and this is trusted server-generated
output, not an input trust boundary — a rigid response model would only
force an artificial union type for no real safety gain."""
from pydantic import BaseModel, field_validator


class QueryRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        """Reject empty/whitespace-only questions before they reach the LLM."""
        value = value.strip()
        if not value:
            raise ValueError("question must not be empty")
        return value

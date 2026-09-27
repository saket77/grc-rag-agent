"""Public response types and the deliberately smaller generation contract."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

NOT_FOUND = "Not found in document"


class Citation(BaseModel):
    page: int | None = Field(description="One-based physical PDF page; null for JSON")
    excerpt: str


class AnswerResult(BaseModel):
    question: str
    answer: str
    status: Literal["found", "partial", "not_found"] = Field(
        description="Model-assessed evidence coverage, not a confidence score or verification."
    )
    citations: list[Citation]


class QAResponse(BaseModel):
    results: list[AnswerResult]


class GeneratedPartContent(BaseModel):
    """Model-written fields; the request schema supplies the part identity."""

    model_config = ConfigDict(extra="forbid", strict=True)
    coverage: Literal["full", "partial", "related_only", "none"] = Field(
        description=(
            "full when every requested fact is established; partial when at least one requested "
            "fact is established but another is missing; related_only when evidence concerns the "
            "topic but establishes none of the requested facts, including a different property, "
            "category, data class, or an unlinked subject-object relationship; none when no "
            "relevant evidence exists"
        )
    )
    answer: str = Field(
        description="Concise prose without chunk IDs, part IDs, or inline citation annotations."
    )
    evidence_chunk_ids: list[str]


class GeneratedPartAnswer(GeneratedPartContent):
    """Normalized internal result, after restoring server-owned part IDs."""

    part_id: str


class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    parts: list[GeneratedPartAnswer]

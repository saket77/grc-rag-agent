"""Public response types and the deliberately smaller generation contract."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

NOT_FOUND = "Not found in document"


class Citation(BaseModel):
    # todo: I dont like this we gotta figure out a way to support json citations like if hthe
    # knowledge base is json not pdf then we need to be able to give the user a way to see where
    # in that json that feild came
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
    status: Literal["supported", "partial", "not_found"]
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

    @property
    def supported(self) -> bool:
        return any(part.status != "not_found" for part in self.parts)

    @property
    def evidence_chunk_ids(self) -> list[str]:
        return [chunk_id for part in self.parts for chunk_id in part.evidence_chunk_ids]

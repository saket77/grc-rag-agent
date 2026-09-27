"""Public response types and the deliberately smaller generation contract."""

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
    citations: list[Citation]


class QAResponse(BaseModel):
    results: list[AnswerResult]


class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    supported: bool
    answer: str
    evidence_chunk_ids: list[str]

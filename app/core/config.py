"""Validated environment configuration; the answer model is intentionally fixed."""

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ANSWER_MODEL = "gpt-4o-mini"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", hide_input_in_errors=True)

    openai_api_key: SecretStr | None = None
    embedding_model: str = "text-embedding-3-small"
    max_document_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    max_questions_bytes: int = Field(default=128 * 1024, gt=0)
    max_request_bytes: int = Field(default=11 * 1024 * 1024, gt=0)
    max_questions: int = Field(default=50, gt=0)
    max_question_chars: int = Field(default=2000, gt=0)
    max_pdf_pages: int = Field(default=200, gt=0)
    max_extracted_chars: int = Field(default=1_000_000, gt=0)
    max_chunks: int = Field(default=2000, gt=0)
    max_json_depth: int = Field(default=64, gt=0)
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=400, ge=0)
    retrieval_k: int = Field(default=6, gt=0)
    embedding_batch_size: int = Field(default=64, gt=0, le=256)
    max_provider_calls: int = Field(default=4, gt=0)
    max_active_requests: int = Field(default=2, gt=0)
    worker_threads: int = Field(default=2, gt=0)
    request_timeout_seconds: float = Field(default=120, gt=0)
    provider_timeout_seconds: float = Field(default=30, gt=0)
    max_answer_tokens: int = Field(default=1200, gt=0)

    @model_validator(mode="after")
    def check_chunk_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

"""Allowlisted JSON logs: never stringify provider exceptions or request bodies."""

import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime

request_id_context: ContextVar[str | None] = ContextVar("request_id", default=None)
SAFE_FIELDS = (
    "request_id",
    "stage",
    "duration_ms",
    "status_code",
    "code",
    "question_count",
    "unique_question_count",
    "chunk_count",
    "document_bytes",
    "attempt",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "supported",
    "citation_count",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        result = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "event": record.msg,
            "request_id": request_id_context.get(),
        }
        for field in SAFE_FIELDS:
            if hasattr(record, field):
                result[field] = getattr(record, field)
        return json.dumps(result, ensure_ascii=True)


def configure_logging() -> None:
    logger = logging.getLogger("app")
    if not any(isinstance(h.formatter, JsonFormatter) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # These libraries can log parsed source strings, HTTP bodies, or provider failures.
    for name in ("pypdf", "openai", "httpx", "httpcore", "langchain", "langsmith"):
        third_party = logging.getLogger(name)
        third_party.handlers = [logging.NullHandler()]
        third_party.propagate = False

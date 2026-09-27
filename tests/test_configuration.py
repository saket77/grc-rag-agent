import json
import logging

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.observability import JsonFormatter, question_number_context, request_id_context


def test_configuration_reads_environment_without_exposing_secret(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-never-live")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
    settings = Settings(_env_file=None)
    assert settings.openai_api_key.get_secret_value() == "test-secret-never-live"
    assert "test-secret-never-live" not in repr(settings)
    assert settings.embedding_model == "text-embedding-3-large"


def test_configuration_validates_limits():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_questions=0)


def test_json_log_allows_metrics_but_omits_sensitive_extras_and_exception():
    token = request_id_context.set("request-123")
    question_token = question_number_context.set(3)
    try:
        record = logging.LogRecord("app", logging.INFO, "", 0, "generation_usage", (), None)
        record.input_tokens = 10
        record.operation = "answer_generation"
        record.citation_chars = 900
        record.document = "private document"
        record.api_key = "private key"
        record.exc_text = "private traceback"
        output = JsonFormatter().format(record)
        assert json.loads(output)["input_tokens"] == 10
        assert json.loads(output)["request_id"] == "request-123"
        assert json.loads(output)["question_number"] == 3
        assert json.loads(output)["operation"] == "answer_generation"
        assert json.loads(output)["citation_chars"] == 900
        assert "private" not in output
    finally:
        question_number_context.reset(question_token)
        request_id_context.reset(token)

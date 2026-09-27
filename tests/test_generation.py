"""Offline generator tests: mock the provider, retain schema and evidence validation."""

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from pydantic import SecretStr

from app.config import Settings
from app.errors import ServiceError
from app.generation import OpenAIAnswerGenerator, validate_answer
from app.models import NOT_FOUND, EvidenceQuote, GeneratedAnswer


@pytest.fixture
def chunks():
    return [
        Document(
            page_content="The production service is hosted on AWS. Backups are encrypted.",
            metadata={"chunk_id": "chunk-1", "page": 12},
        ),
        Document(
            page_content='{"encryption": "AES-256"}',
            metadata={"chunk_id": "chunk-2", "page": None},
        ),
    ]


def supported_answer(**changes):
    fields = {
        "supported": True,
        "answer": "The service is hosted on AWS.",
        "evidence": [EvidenceQuote(chunk_id="chunk-1", quote="hosted on AWS.")],
    }
    fields.update(changes)
    return GeneratedAnswer(**fields)


def test_citation_page_and_quote_are_resolved_from_source(chunks):
    result = validate_answer("Where?", chunks, supported_answer())
    assert result.model_dump() == {
        "question": "Where?",
        "answer": "The service is hosted on AWS.",
        "citations": [{"page": 12, "excerpt": "hosted on AWS."}],
    }


def test_json_citations_have_null_page_and_duplicates_are_removed(chunks):
    evidence = EvidenceQuote(chunk_id="chunk-2", quote='"encryption": "AES-256"')
    answer = supported_answer(answer="AES-256", evidence=[evidence, evidence])
    result = validate_answer("Encryption?", chunks, answer)
    assert len(result.citations) == 1
    assert result.citations[0].page is None
    assert result.citations[0].excerpt == '"encryption": "AES-256"'


def test_unsupported_always_returns_exact_contract(chunks):
    generated = GeneratedAnswer(supported=False, answer="I don't know", evidence=[])
    assert validate_answer("Unknown?", chunks, generated).model_dump() == {
        "question": "Unknown?",
        "answer": NOT_FOUND,
        "citations": [],
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"answer": " "},
        {"answer": NOT_FOUND},
        {"evidence": []},
        {"evidence": [EvidenceQuote(chunk_id="invented", quote="hosted on AWS.")]},
        {"evidence": [EvidenceQuote(chunk_id="chunk-1", quote="hosted on Azure.")]},
        {"evidence": [EvidenceQuote(chunk_id="chunk-1", quote="HOSTED ON AWS.")]},
        {"evidence": [EvidenceQuote(chunk_id="chunk-1", quote="  ")]},
        {"evidence": [EvidenceQuote(chunk_id="chunk-1", quote="a" * 1001)]},
    ],
)
def test_invalid_evidence_is_an_error_not_abstention(chunks, changes):
    with pytest.raises(ServiceError) as exc:
        validate_answer("Where?", chunks, supported_answer(**changes))
    assert exc.value.status_code == 502
    assert exc.value.code == "provider_response_invalid"


@pytest.mark.parametrize("page", [0, -1, "12", True])
def test_citation_page_must_be_one_based_integer(chunks, page):
    chunks[0].metadata["page"] = page
    with pytest.raises(ServiceError):
        validate_answer("Where?", chunks, supported_answer())


def test_provider_initialization_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ServiceError) as exc:
        OpenAIAnswerGenerator(Settings(_env_file=None))
    assert exc.value.status_code == 503
    assert exc.value.code == "provider_not_configured"


@pytest.fixture
def provider(monkeypatch):
    runnable = MagicMock()
    runnable.ainvoke = AsyncMock()
    chat = MagicMock()
    chat.return_value.with_structured_output.return_value = runnable
    monkeypatch.setattr("app.generation.ChatOpenAI", chat)
    generator = OpenAIAnswerGenerator(
        Settings(_env_file=None, openai_api_key=SecretStr("test-key-never-sent"))
    )
    return generator, runnable, chat


async def test_provider_uses_fixed_model_and_strict_schema(provider, chunks, caplog):
    generator, runnable, chat = provider
    generated = supported_answer()
    runnable.ainvoke.return_value = {
        "raw": AIMessage(
            content="private model output",
            response_metadata={"finish_reason": "stop"},
            usage_metadata={"input_tokens": 40, "output_tokens": 20, "total_tokens": 60},
        ),
        "parsed": generated,
        "parsing_error": None,
    }
    with caplog.at_level(logging.INFO, logger="app"):
        result = await generator.generate("Where is our private service?", chunks)
    assert result is generated
    kwargs = chat.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["base_url"] == "https://api.openai.com/v1"
    assert kwargs["max_retries"] == 0
    assert kwargs["temperature"] == 0
    assert kwargs["api_key"].get_secret_value() == "test-key-never-sent"
    chat.return_value.with_structured_output.assert_called_once_with(
        GeneratedAnswer, method="json_schema", strict=True, include_raw=True
    )
    messages = runnable.ainvoke.call_args.args[0]
    assert "untrusted data" in messages[0].content
    envelope = json.loads(messages[1].content)
    assert envelope["question"] == "Where is our private service?"
    assert envelope["chunks"][0] == {
        "chunk_id": "chunk-1",
        "text": chunks[0].page_content,
    }
    assert "private" not in caplog.text
    assert "test-key" not in caplog.text
    assert caplog.records[-1].total_tokens == 60


@pytest.mark.parametrize("failure", ["schema", "refusal", "length", "missing_parsed"])
async def test_invalid_provider_results_are_safe_errors(provider, chunks, failure):
    generator, runnable, _ = provider
    raw = AIMessage(content="private model output", response_metadata={"finish_reason": "stop"})
    response = {"raw": raw, "parsed": supported_answer(), "parsing_error": None}
    if failure == "schema":
        response["parsing_error"] = ValueError("private data in parser failure")
    elif failure == "refusal":
        raw.additional_kwargs["refusal"] = "I cannot answer."
    elif failure == "length":
        raw.response_metadata["finish_reason"] = "length"
    else:
        response["parsed"] = None
    runnable.ainvoke.return_value = response
    with pytest.raises(ServiceError) as exc:
        await generator.generate("Where?", chunks)
    assert exc.value.status_code == 502
    assert "private" not in exc.value.message
    runnable.ainvoke.assert_awaited_once()


async def test_provider_errors_propagate_for_shared_retry_policy(provider, chunks):
    generator, runnable, _ = provider
    runnable.ainvoke.side_effect = TimeoutError("provider timeout")
    with pytest.raises(TimeoutError):
        await generator.generate("Where?", chunks)
    runnable.ainvoke.assert_awaited_once()

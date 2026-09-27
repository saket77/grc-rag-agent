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
from app.models import NOT_FOUND, GeneratedAnswer, GeneratedPartAnswer
from app.planning import QuestionPart, QuestionPlan, build_question_plan


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
        "part_id": "part_1",
        "status": "supported",
        "answer": "The service is hosted on AWS.",
        "evidence_chunk_ids": ["chunk-1"],
    }
    fields.update(changes)
    return GeneratedAnswer(parts=[GeneratedPartAnswer(**fields)])


def unsupported_answer(**changes):
    fields = {
        "part_id": "part_1",
        "status": "not_found",
        "answer": NOT_FOUND,
        "evidence_chunk_ids": [],
    }
    fields.update(changes)
    return GeneratedAnswer(parts=[GeneratedPartAnswer(**fields)])


def plan(question):
    return build_question_plan(question)


def test_citation_page_and_full_chunk_are_resolved_from_source(chunks):
    result = validate_answer(plan("Where?"), chunks, supported_answer())
    assert result.model_dump() == {
        "question": "Where?",
        "answer": "The service is hosted on AWS.",
        "status": "found",
        "citations": [{"page": 12, "excerpt": chunks[0].page_content}],
    }


def test_json_citations_have_null_page_and_duplicates_are_removed(chunks):
    answer = supported_answer(answer="AES-256", evidence_chunk_ids=["chunk-2", "chunk-2"])
    result = validate_answer(plan("Encryption?"), chunks, answer)
    assert len(result.citations) == 1
    assert result.citations[0].page is None
    assert result.citations[0].excerpt == chunks[1].page_content


def test_unsupported_always_returns_exact_contract(chunks):
    generated = unsupported_answer()
    assert validate_answer(plan("Unknown?"), chunks, generated).model_dump() == {
        "question": "Unknown?",
        "answer": NOT_FOUND,
        "status": "not_found",
        "citations": [],
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"answer": " "},
        {"answer": NOT_FOUND},
        {"evidence_chunk_ids": []},
        {"evidence_chunk_ids": ["invented"]},
    ],
)
@pytest.mark.parametrize("status", ["supported", "partial"])
def test_invalid_evidence_is_an_error_not_abstention(chunks, changes, status):
    with pytest.raises(ServiceError) as exc:
        validate_answer(plan("Where?"), chunks, supported_answer(status=status, **changes))
    assert exc.value.status_code == 502
    assert exc.value.code == "provider_response_invalid"


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (["supported"], "found"),
        (["partial"], "partial"),
        (["not_found"], "not_found"),
        (["supported", "supported"], "found"),
        (["supported", "partial"], "partial"),
        (["supported", "not_found"], "partial"),
        (["not_found", "supported"], "partial"),
        (["partial", "not_found"], "partial"),
        (["partial", "partial"], "partial"),
        (["not_found", "not_found"], "not_found"),
    ],
)
def test_public_status_reflects_coverage_of_all_parts(chunks, statuses, expected):
    parts = tuple(
        QuestionPart(part_id=f"part_{index}", question=f"Question {index}?")
        for index in range(1, len(statuses) + 1)
    )
    question_plan = QuestionPlan(original_question="Complete question?", parts=parts)
    generated = GeneratedAnswer(
        parts=[
            GeneratedPartAnswer(
                part_id=part.part_id,
                status=status,
                answer=NOT_FOUND if status == "not_found" else "The service is hosted on AWS.",
                evidence_chunk_ids=[] if status == "not_found" else ["chunk-1"],
            )
            for part, status in zip(parts, statuses, strict=True)
        ]
    )

    result = validate_answer(question_plan, chunks, generated)

    assert result.status == expected
    assert [citation.model_dump() for citation in result.citations] == (
        [] if expected == "not_found" else [{"page": 12, "excerpt": chunks[0].page_content}]
    )


def test_unknown_evidence_id_logs_safe_diagnostic(caplog, chunks):
    generated = supported_answer(
        answer="Private generated answer", evidence_chunk_ids=["private-invented-id"]
    )

    with caplog.at_level(logging.WARNING, logger="app"), pytest.raises(ServiceError):
        validate_answer(plan("Private question"), chunks, generated)

    rejection = next(
        record for record in caplog.records if record.msg == "answer_response_rejected"
    )
    assert rejection.code == "evidence_chunk_id_unknown"
    assert rejection.stage == "citation_validation"
    assert rejection.evidence_index == 0
    assert "Private" not in caplog.text
    assert "invented" not in caplog.text


@pytest.mark.parametrize("page", [0, -1, "12", True])
def test_citation_page_must_be_one_based_integer(chunks, page):
    chunks[0].metadata["page"] = page
    with pytest.raises(ServiceError):
        validate_answer(plan("Where?"), chunks, supported_answer())


def test_partial_answer_preserves_supported_parts_and_names_missing_parts(chunks):
    generated = GeneratedAnswer(
        parts=[
            GeneratedPartAnswer(
                part_id="part_1",
                status="supported",
                answer="The service is hosted on AWS.",
                evidence_chunk_ids=["chunk-1"],
            ),
            GeneratedPartAnswer(
                part_id="part_2",
                status="not_found",
                answer=NOT_FOUND,
                evidence_chunk_ids=[],
            ),
        ]
    )

    result = validate_answer(plan("Which cloud? What is the retention period?"), chunks, generated)

    assert result.answer == (
        "The service is hosted on AWS. "
        "The provided evidence does not specify: What is the retention period."
    )
    assert [citation.model_dump() for citation in result.citations] == [
        {"page": 12, "excerpt": chunks[0].page_content}
    ]


def test_single_part_preserves_a_grounded_partial_answer(chunks):
    answer = "The production service is hosted on AWS; the backup provider is not specified."
    generated = GeneratedAnswer(
        parts=[
            GeneratedPartAnswer(
                part_id="part_1",
                status="partial",
                answer=answer,
                evidence_chunk_ids=["chunk-1"],
            )
        ]
    )

    result = validate_answer(
        plan("Which providers host production and backups?"), chunks, generated
    )

    assert result.answer == answer
    assert [citation.model_dump() for citation in result.citations] == [
        {"page": 12, "excerpt": chunks[0].page_content}
    ]


def test_numbered_option_question_uses_one_result_per_planned_part(chunks):
    generated = GeneratedAnswer(
        parts=[
            GeneratedPartAnswer(
                part_id="part_1",
                status="partial",
                answer="CPU monitoring is documented, but the APM label is not explicit.",
                evidence_chunk_ids=["chunk-1"],
            ),
            GeneratedPartAnswer(
                part_id="part_2",
                status="not_found",
                answer=NOT_FOUND,
                evidence_chunk_ids=[],
            ),
            GeneratedPartAnswer(
                part_id="part_3",
                status="not_found",
                answer=NOT_FOUND,
                evidence_chunk_ids=[],
            ),
        ]
    )

    result = validate_answer(
        plan("Which monitoring exists: 1. APM, 2. EUM, 3. DEM?"), chunks, generated
    )

    assert result.answer == (
        "CPU monitoring is documented, but the APM label is not explicit. "
        "EUM: Not specified in the provided evidence. "
        "DEM: Not specified in the provided evidence."
    )
    assert len(result.citations) == 1


def test_all_numbered_parts_not_found_retains_exact_fallback(chunks):
    question_plan = plan("Which controls: 1. Redundancy, 2. Failover?")
    generated = GeneratedAnswer(
        parts=[
            GeneratedPartAnswer(
                part_id=part.part_id, status="not_found", answer=NOT_FOUND, evidence_chunk_ids=[]
            )
            for part in question_plan.parts
        ]
    )

    result = validate_answer(question_plan, chunks, generated)

    assert result.answer == NOT_FOUND
    assert result.citations == []


@pytest.mark.parametrize(
    "generated",
    [
        GeneratedAnswer(parts=[]),
        GeneratedAnswer(
            parts=[
                GeneratedPartAnswer(
                    part_id="wrong",
                    status="not_found",
                    answer=NOT_FOUND,
                    evidence_chunk_ids=[],
                )
            ]
        ),
        unsupported_answer(answer="I do not know"),
        unsupported_answer(evidence_chunk_ids=["chunk-1"]),
    ],
)
def test_invalid_part_contract_is_rejected(chunks, generated):
    with pytest.raises(ServiceError):
        validate_answer(plan("Where?"), chunks, generated)


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
        result = await generator.generate(plan("Where is our private service?"), chunks)
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
    schema_text = json.dumps(GeneratedAnswer.model_json_schema())
    assert "evidence_chunk_ids" in schema_text
    assert '"quote"' not in schema_text
    messages = runnable.ainvoke.call_args.args[0]
    assert "untrusted data" in messages[0].content
    assert "Never generate quotations" in messages[0].content
    assert "Do not combine separate facts" in messages[0].content
    envelope = json.loads(messages[1].content)
    assert envelope["question"] == "Where is our private service?"
    assert envelope["parts"] == [{"part_id": "part_1", "question": "Where is our private service?"}]
    assert envelope["chunks"][0] == {
        "chunk_id": "chunk-1",
        "text": chunks[0].page_content,
    }
    assert "private" not in caplog.text
    assert "test-key" not in caplog.text
    assert caplog.records[-1].total_tokens == 60


async def test_display_labels_do_not_change_questions_sent_to_the_model(provider, chunks):
    generator, runnable, _ = provider
    question_plan = plan("Which controls: 1. Redundancy, 2. Failover?")
    generated = GeneratedAnswer(
        parts=[
            GeneratedPartAnswer(
                part_id=part.part_id, status="not_found", answer=NOT_FOUND, evidence_chunk_ids=[]
            )
            for part in question_plan.parts
        ]
    )
    runnable.ainvoke.return_value = {
        "raw": AIMessage(content="", response_metadata={"finish_reason": "stop"}),
        "parsed": generated,
        "parsing_error": None,
    }

    await generator.generate(question_plan, chunks)

    runnable.ainvoke.assert_awaited_once()
    envelope = json.loads(runnable.ainvoke.call_args.args[0][1].content)
    assert envelope["parts"] == [
        {"part_id": "part_1", "question": "Which controls Redundancy?"},
        {"part_id": "part_2", "question": "Which controls Failover?"},
    ]
    assert [part["question"] for part in envelope["parts"]] == list(
        question_plan.retrieval_queries[1:]
    )


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
        await generator.generate(plan("Where?"), chunks)
    assert exc.value.status_code == 502
    assert "private" not in exc.value.message
    runnable.ainvoke.assert_awaited_once()


async def test_provider_errors_propagate_for_shared_retry_policy(provider, chunks):
    generator, runnable, _ = provider
    runnable.ainvoke.side_effect = TimeoutError("provider timeout")
    with pytest.raises(TimeoutError):
        await generator.generate(plan("Where?"), chunks)
    runnable.ainvoke.assert_awaited_once()

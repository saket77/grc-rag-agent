"""Offline generator tests: mock the provider, retain schema and evidence validation."""

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from pydantic import SecretStr, ValidationError

from app.core.config import Settings
from app.core.errors import ServiceError
from app.rag.generation import OpenAIAnswerGenerator, build_response_model, validate_answer
from app.rag.planning import QuestionPart, QuestionPlan, build_question_plan
from app.rag.prompts import EXAMPLES, SYSTEM_PROMPT
from app.schemas.qa import NOT_FOUND, GeneratedAnswer, GeneratedPartAnswer


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


def wire_answer(question_plan, generated):
    return build_response_model(question_plan).model_validate(
        {"parts": {part.part_id: part.model_dump(exclude={"part_id"}) for part in generated.parts}}
    )


@pytest.mark.parametrize("part_count", [1, 2, 3, 8])
def test_response_schema_requires_exact_planned_parts(part_count):
    question_plan = plan(" ".join(f"Question {i}?" for i in range(part_count)))
    schema = build_response_model(question_plan).model_json_schema()
    parts = schema["$defs"]["PlannedParts"]
    content = schema["$defs"]["GeneratedPartContent"]
    expected_ids = [part.part_id for part in question_plan.parts]

    assert schema["required"] == ["parts"]
    assert list(parts["properties"]) == expected_ids
    assert parts["required"] == expected_ids
    assert set(content["required"]) == {"status", "answer", "evidence_chunk_ids"}
    assert "part_id" not in content["properties"]
    assert content["properties"]["status"]["enum"] == ["supported", "partial", "not_found"]
    for object_schema in (schema, parts, content):
        assert object_schema["additionalProperties"] is False


@pytest.mark.parametrize("invalid", ["missing", "extra", "status", "part_id", "old_list"])
def test_response_schema_rejects_invalid_parts(invalid):
    question_plan = plan("Which features: 1. Upload, 2. Export, 3. Search?")
    parts = {
        part.part_id: {"status": "not_found", "answer": NOT_FOUND, "evidence_chunk_ids": []}
        for part in question_plan.parts
    }
    if invalid == "missing":
        del parts["part_3"]
    elif invalid == "extra":
        parts["part_4"] = parts["part_1"].copy()
    elif invalid == "status":
        parts["part_1"]["status"] = "found"
    elif invalid == "part_id":
        parts["part_1"]["part_id"] = "part_1"
    else:
        parts = [{"part_id": key, **value} for key, value in parts.items()]

    with pytest.raises(ValidationError):
        build_response_model(question_plan).model_validate({"parts": parts})


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
    monkeypatch.setattr("app.rag.generation.ChatOpenAI", chat)
    generator = OpenAIAnswerGenerator(
        Settings(_env_file=None, openai_api_key=SecretStr("test-key-never-sent"))
    )
    return generator, runnable, chat


async def test_provider_uses_fixed_model_and_strict_schema(provider, chunks, caplog):
    generator, runnable, chat = provider
    generated = supported_answer()
    question_plan = plan("Where is our private service?")
    runnable.ainvoke.return_value = {
        "raw": AIMessage(
            content="private model output",
            response_metadata={"finish_reason": "stop"},
            usage_metadata={"input_tokens": 40, "output_tokens": 20, "total_tokens": 60},
        ),
        "parsed": wire_answer(question_plan, generated),
        "parsing_error": None,
    }
    with caplog.at_level(logging.INFO, logger="app"):
        result = await generator.generate(question_plan, chunks)
    assert result == generated
    kwargs = chat.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["base_url"] == "https://api.openai.com/v1"
    assert kwargs["max_retries"] == 0
    assert kwargs["temperature"] == 0
    assert kwargs["api_key"].get_secret_value() == "test-key-never-sent"
    binding = chat.return_value.with_structured_output
    binding.assert_called_once()
    assert binding.call_args.kwargs == {
        "method": "json_schema",
        "strict": True,
        "include_raw": True,
    }
    response_model = binding.call_args.args[0]
    schema_text = json.dumps(response_model.model_json_schema())
    assert "evidence_chunk_ids" in schema_text
    assert '"quote"' not in schema_text
    messages = runnable.ainvoke.call_args.args[0]
    assert messages[0].content == SYSTEM_PROMPT
    assert "untrusted data" in messages[0].content
    assert "free of internal IDs, quotations" in messages[0].content
    assert "do not invent details, relationships" in messages[0].content
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
        "parsed": wire_answer(question_plan, generated),
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


@pytest.mark.parametrize(
    ("example", "expected_status"),
    list(
        zip(
            EXAMPLES,
            ["found", "partial", "not_found"],
            strict=True,
        )
    ),
    ids=["supported", "partial", "not_found"],
)
async def test_prompt_examples_match_request_schema_and_public_contract(
    provider, example, expected_status
):
    generator, runnable, _ = provider
    envelope = example["input"]
    question_plan = plan(envelope["question"])
    assert envelope["parts"] == [
        {"part_id": part.part_id, "question": part.question} for part in question_plan.parts
    ]
    assert json.dumps(example, ensure_ascii=False) in SYSTEM_PROMPT
    assert len(example["output"]["parts"]) == 1
    assert example["output"]["parts"]["part_1"]["status"] == (
        "supported" if expected_status == "found" else expected_status
    )
    chunks = [
        Document(page_content=chunk["text"], metadata={"chunk_id": chunk["chunk_id"]})
        for chunk in envelope["chunks"]
    ]
    runnable.ainvoke.return_value = {
        "raw": AIMessage(content="", response_metadata={"finish_reason": "stop"}),
        "parsed": build_response_model(question_plan).model_validate(example["output"]),
        "parsing_error": None,
    }

    generated = await generator.generate(question_plan, chunks)
    result = validate_answer(question_plan, chunks, generated)

    runnable.ainvoke.assert_awaited_once()
    assert result.status == expected_status
    assert "example_source" not in result.answer
    assert "part_" not in result.answer
    assert [citation.excerpt for citation in result.citations] == (
        [] if expected_status == "not_found" else [chunks[0].page_content]
    )


async def test_provider_normalizes_response_in_plan_order(provider, chunks):
    generator, runnable, _ = provider
    question_plan = plan("Which cloud? Which encryption?")
    runnable.ainvoke.return_value = {
        "raw": AIMessage(content=""),
        "parsed": build_response_model(question_plan).model_validate(
            {
                "parts": {
                    "part_2": {
                        "status": "supported",
                        "answer": "AES-256",
                        "evidence_chunk_ids": ["chunk-2"],
                    },
                    "part_1": {
                        "status": "supported",
                        "answer": "AWS",
                        "evidence_chunk_ids": ["chunk-1"],
                    },
                }
            }
        ),
        "parsing_error": None,
    }

    generated = await generator.generate(question_plan, chunks)

    assert [part.part_id for part in generated.parts] == ["part_1", "part_2"]
    assert [part.answer for part in generated.parts] == ["AWS", "AES-256"]


async def test_concurrent_questions_have_independent_schemas(provider, chunks):
    generator, _, chat = provider
    runnables = []

    def bind(response_model, **kwargs):
        async def invoke(messages, **kwargs):
            await asyncio.sleep(0)
            envelope = json.loads(messages[1].content)
            return {
                "raw": AIMessage(content=""),
                "parsed": response_model.model_validate(
                    {
                        "parts": {
                            part["part_id"]: {
                                "status": "not_found",
                                "answer": NOT_FOUND,
                                "evidence_chunk_ids": [],
                            }
                            for part in envelope["parts"]
                        }
                    }
                ),
                "parsing_error": None,
            }

        runnable = MagicMock(ainvoke=AsyncMock(side_effect=invoke))
        runnables.append(runnable)
        return runnable

    chat.return_value.with_structured_output.side_effect = bind
    plans = [plan("Which cloud?"), plan("Which cloud? Which region? Which backup region?")]

    results = await asyncio.gather(*(generator.generate(item, chunks) for item in plans))

    assert [[part.part_id for part in result.parts] for result in results] == [
        ["part_1"],
        ["part_1", "part_2", "part_3"],
    ]
    assert chat.return_value.with_structured_output.call_count == 2
    for runnable in runnables:
        runnable.ainvoke.assert_awaited_once()


async def test_provider_revalidates_parsed_model_against_current_plan(provider, chunks, caplog):
    generator, runnable, _ = provider
    runnable.ainvoke.return_value = {
        "raw": AIMessage(content="private model output"),
        "parsed": wire_answer(plan("Another question?"), supported_answer()),
        "parsing_error": None,
    }

    with caplog.at_level(logging.WARNING, logger="app"), pytest.raises(ServiceError) as exc:
        await generator.generate(plan("Where? When?"), chunks)

    assert exc.value.status_code == 502
    assert caplog.records[-1].code == "schema_validation_failed"
    assert "private" not in caplog.text
    runnable.ainvoke.assert_awaited_once()


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

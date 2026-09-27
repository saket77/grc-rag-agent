"""Generate a structured answer, then resolve citations against retrieved text only."""

import json
import logging
from typing import Protocol

from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import tracing_context
from openai import ContentFilterFinishReasonError, LengthFinishReasonError
from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from app.core.config import ANSWER_MODEL, Settings
from app.core.errors import ServiceError
from app.rag.planning import QuestionPlan
from app.rag.prompts import SYSTEM_PROMPT
from app.schemas.qa import (
    NOT_FOUND,
    AnswerResult,
    Citation,
    GeneratedAnswer,
    GeneratedPartAnswer,
    GeneratedPartContent,
)

logger = logging.getLogger("app")


def build_response_model(plan: QuestionPlan) -> type[BaseModel]:
    """Require each planned part exactly once; the model cannot invent another part."""
    config = ConfigDict(extra="forbid", strict=True)
    parts_model = create_model(
        "PlannedParts",
        __config__=config,
        **{part.part_id: (GeneratedPartContent, ...) for part in plan.parts},
    )
    return create_model("PlannedAnswer", __config__=config, parts=(parts_model, ...))


class AnswerGenerator(Protocol):
    async def generate(self, plan: QuestionPlan, chunks: list[Document]) -> GeneratedAnswer: ...


def _invalid_response(
    reason: str,
    *,
    stage: str = "generation",
    evidence_index: int | None = None,
    finish_reason: str | None = None,
) -> ServiceError:
    # Log only controlled metadata. Never include parser exceptions, prompts, model output,
    # questions, answers, quotes, source text, filenames, or credentials.
    details: dict[str, str | int | bool] = {"stage": stage, "code": reason}
    optional = {
        "evidence_index": evidence_index,
        "finish_reason": finish_reason,
    }
    details.update({key: value for key, value in optional.items() if value is not None})
    logger.warning("answer_response_rejected", extra=details)
    return ServiceError(
        502,
        "provider_response_invalid",
        "The answer provider returned an invalid or unverifiable response.",
    )


class OpenAIAnswerGenerator:
    """One generation call; shared concurrency, retries, and deadlines live in orchestration."""

    def __init__(self, settings: Settings):
        if (
            settings.openai_api_key is None
            or not settings.openai_api_key.get_secret_value().strip()
        ):
            raise ServiceError(
                503,
                "provider_not_configured",
                "Set OPENAI_API_KEY before requesting document answers.",
            )
        self._model = ChatOpenAI(
            model=ANSWER_MODEL,
            api_key=settings.openai_api_key,
            base_url="https://api.openai.com/v1",
            temperature=0,
            max_retries=0,
            timeout=settings.provider_timeout_seconds,
            max_tokens=settings.max_answer_tokens,
            use_responses_api=False,
            callbacks=[],
            verbose=False,
            cache=False,
        )

    async def aclose(self) -> None:
        """Release the SDK clients owned by this generator during application shutdown."""
        await self._model.root_async_client.close()
        self._model.root_client.close()

    async def generate(self, plan: QuestionPlan, chunks: list[Document]) -> GeneratedAnswer:
        response_model = build_response_model(plan)
        structured = self._model.with_structured_output(
            response_model, method="json_schema", strict=True, include_raw=True
        )
        envelope = {
            "question": plan.original_question,
            "parts": [{"part_id": part.part_id, "question": part.question} for part in plan.parts],
            "chunks": [
                {"chunk_id": chunk.metadata["chunk_id"], "text": chunk.page_content}
                for chunk in chunks
            ],
        }
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(envelope, ensure_ascii=False)),
        ]
        try:
            # Compliance documents must not be sent to ambient LangSmith tracing endpoints.
            with tracing_context(enabled=False):
                response = await structured.ainvoke(messages, config={"callbacks": []})
        except OutputParserException:
            raise _invalid_response("schema_parse_failed") from None
        except ValidationError:
            raise _invalid_response("schema_validation_failed") from None
        except LengthFinishReasonError:
            raise _invalid_response("finish_reason_length", finish_reason="length") from None
        except ContentFilterFinishReasonError:
            raise _invalid_response(
                "finish_reason_content_filter", finish_reason="content_filter"
            ) from None

        if not isinstance(response, dict):
            raise _invalid_response("response_not_mapping")
        raw = response.get("raw")
        if not isinstance(raw, AIMessage):
            raise _invalid_response("raw_message_missing")
        self._log_usage(raw)
        if response.get("parsing_error") is not None:
            raise _invalid_response("schema_parse_failed")
        if raw.additional_kwargs.get("refusal"):
            raise _invalid_response("model_refused")
        finish_reason = raw.response_metadata.get("finish_reason")
        if finish_reason not in (None, "stop"):
            safe_finish_reason = (
                finish_reason
                if finish_reason in {"length", "content_filter", "tool_calls"}
                else "other"
            )
            raise _invalid_response("finish_reason_invalid", finish_reason=safe_finish_reason)
        parsed = response.get("parsed")
        if not isinstance(parsed, BaseModel):
            raise _invalid_response("parsed_answer_missing")
        try:
            parts = response_model.model_validate(parsed.model_dump()).parts.model_dump()
        except ValidationError:
            raise _invalid_response("schema_validation_failed") from None
        return GeneratedAnswer(
            parts=[
                GeneratedPartAnswer(part_id=part.part_id, **parts[part.part_id])
                for part in plan.parts
            ]
        )

    @staticmethod
    def _log_usage(raw: AIMessage) -> None:
        usage = raw.usage_metadata or {}
        safe_usage = {
            key: value
            for key in ("input_tokens", "output_tokens", "total_tokens")
            if type(value := usage.get(key)) is int and value >= 0
        }
        logger.info("generation_usage", extra={"event": "generation_usage", **safe_usage})


def validate_answer(
    plan: QuestionPlan, chunks: list[Document], generated: GeneratedAnswer
) -> AnswerResult:
    """Validate every question part and resolve model-selected IDs to source citations."""
    if not isinstance(generated, GeneratedAnswer):
        raise _invalid_response("generated_answer_wrong_type", stage="citation_validation")
    try:
        generated = GeneratedAnswer.model_validate(generated.model_dump())
    except ValidationError:
        raise _invalid_response(
            "generated_answer_schema_invalid", stage="citation_validation"
        ) from None
    expected_parts = plan.parts
    expected_ids = {part.part_id for part in expected_parts}
    generated_by_id: dict[str, GeneratedPartAnswer] = {}
    for part in generated.parts:
        if part.part_id in generated_by_id:
            raise _invalid_response("answer_part_id_duplicate", stage="citation_validation")
        generated_by_id[part.part_id] = part
    if set(generated_by_id) != expected_ids:
        raise _invalid_response("answer_part_ids_invalid", stage="citation_validation")

    answered_parts = [part for part in generated.parts if part.status != "not_found"]
    if not answered_parts:
        for part in generated.parts:
            if part.status == "not_found" and (
                part.answer.strip() != NOT_FOUND or part.evidence_chunk_ids
            ):
                raise _invalid_response("unsupported_part_has_answer", stage="citation_validation")
        return AnswerResult(
            question=plan.original_question, answer=NOT_FOUND, status="not_found", citations=[]
        )

    sources = {}
    for chunk in chunks:
        chunk_id = chunk.metadata.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise _invalid_response("source_chunk_id_invalid", stage="citation_validation")
        if chunk_id in sources:
            raise _invalid_response("source_chunk_id_duplicate", stage="citation_validation")
        sources[chunk_id] = chunk

    answers: list[str] = []
    citations: list[Citation] = []
    seen: set[str] = set()
    for expected_part in expected_parts:
        part = generated_by_id[expected_part.part_id]
        if part.status == "not_found":
            if part.answer.strip() != NOT_FOUND or part.evidence_chunk_ids:
                raise _invalid_response("unsupported_part_has_answer", stage="citation_validation")
            if expected_part.label:
                answers.append(f"{expected_part.label}: Not specified in the provided evidence.")
            else:
                missing_question = expected_part.question.rstrip(" ?.!")
                answers.append(f"The provided evidence does not specify: {missing_question}.")
            continue
        answer = part.answer.strip()
        if not answer:
            raise _invalid_response("supported_answer_blank", stage="citation_validation")
        if answer == NOT_FOUND:
            raise _invalid_response("supported_answer_is_fallback", stage="citation_validation")
        if not part.evidence_chunk_ids:
            raise _invalid_response(
                "supported_answer_missing_chunk_ids", stage="citation_validation"
            )
        answers.append(answer)
        for evidence_index, chunk_id in enumerate(part.evidence_chunk_ids):
            source = sources.get(chunk_id)
            if source is None:
                raise _invalid_response(
                    "evidence_chunk_id_unknown",
                    stage="citation_validation",
                    evidence_index=evidence_index,
                )
            if not source.page_content.strip():
                raise _invalid_response(
                    "source_chunk_blank",
                    stage="citation_validation",
                    evidence_index=evidence_index,
                )
            page = source.metadata.get("page")
            if page is not None and (type(page) is not int or page < 1):
                raise _invalid_response(
                    "page_metadata_invalid",
                    stage="citation_validation",
                    evidence_index=evidence_index,
                )
            if chunk_id not in seen:
                citations.append(Citation(page=page, excerpt=source.page_content))
                seen.add(chunk_id)
    return AnswerResult(
        question=plan.original_question,
        answer=" ".join(answers),
        status=(
            "found" if all(part.status == "supported" for part in generated.parts) else "partial"
        ),
        citations=citations,
    )

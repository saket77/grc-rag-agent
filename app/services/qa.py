"""Orchestrate one document and many questions without embedding the document again."""

import asyncio
import logging
from time import perf_counter

import httpx
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langsmith import tracing_context

from app.core.config import Settings
from app.core.errors import ServiceError
from app.core.observability import question_number_context
from app.core.runtime import ProviderRunner, WorkerPool
from app.rag.generation import AnswerGenerator, OpenAIAnswerGenerator, validate_answer
from app.rag.ingestion import parse_document, parse_questions, split_documents
from app.rag.planning import QuestionPlan, build_question_plan
from app.rag.retrieval import IndexBuilder, merge_retrieval_results
from app.schemas.qa import AnswerResult, QAResponse

logger = logging.getLogger("app")


class QAService:
    def __init__(
        self,
        settings: Settings,
        embeddings: Embeddings | None = None,
        generator: AnswerGenerator | None = None,
    ):
        self.settings = settings
        self.embeddings = embeddings
        self.generator = generator
        self.provider = ProviderRunner(settings)
        self.workers = WorkerPool(settings.worker_threads)
        self.http_client: httpx.AsyncClient | None = None
        self.sync_http_client: httpx.Client | None = None
        self.owned_generator: OpenAIAnswerGenerator | None = None

    def _prepare(self, question_bytes: bytes, document_bytes: bytes, kind: str):
        questions = parse_questions(question_bytes, self.settings)
        documents = parse_document(document_bytes, kind, self.settings)
        chunks = split_documents(documents, self.settings)
        return questions, chunks

    def _configure_clients(self) -> None:
        if self.embeddings is not None and self.generator is not None:
            return
        key = self.settings.openai_api_key
        if key is None or not key.get_secret_value().strip():
            raise ServiceError(
                503, "missing_api_key", "Set OPENAI_API_KEY in the environment or local .env file."
            )
        if self.embeddings is None:
            self.http_client = httpx.AsyncClient(trust_env=False)
            self.sync_http_client = httpx.Client(trust_env=False)
            self.embeddings = OpenAIEmbeddings(
                model=self.settings.embedding_model,
                api_key=key.get_secret_value(),
                base_url="https://api.openai.com/v1",
                max_retries=0,
                request_timeout=self.settings.provider_timeout_seconds,
                chunk_size=self.settings.embedding_batch_size,
                check_embedding_ctx_length=False,
                http_async_client=self.http_client,
                http_client=self.sync_http_client,
            )
        if self.generator is None:
            self.owned_generator = OpenAIAnswerGenerator(self.settings)
            self.generator = self.owned_generator

    async def answer(self, question_bytes: bytes, document_bytes: bytes, kind: str) -> QAResponse:
        # Uploaded reports must not flow to an ambient LangSmith tracing configuration.
        with tracing_context(enabled=False):
            return await self._answer(question_bytes, document_bytes, kind)

    async def _answer(self, question_bytes: bytes, document_bytes: bytes, kind: str) -> QAResponse:
        started = perf_counter()
        questions, chunks = await self.workers.run(
            self._prepare, question_bytes, document_bytes, kind
        )
        unique_questions = list(dict.fromkeys(questions))
        logger.info(
            "ingestion_complete",
            extra={
                "stage": "ingestion",
                "duration_ms": round((perf_counter() - started) * 1000, 2),
                "question_count": len(questions),
                "unique_question_count": len(unique_questions),
                "chunk_count": len(chunks),
                "document_bytes": len(document_bytes),
            },
        )
        self._configure_clients()
        assert self.embeddings is not None and self.generator is not None
        builder = IndexBuilder(self.embeddings, self.settings, self.provider, self.workers)
        started = perf_counter()
        index = await builder.build(chunks)
        logger.info(
            "index_complete",
            extra={
                "stage": "embedding",
                "duration_ms": round((perf_counter() - started) * 1000, 2),
                "chunk_count": len(chunks),
            },
        )
        try:
            started = perf_counter()
            question_plans = {
                question: build_question_plan(question) for question in unique_questions
            }
            unique_retrieval_queries = list(
                dict.fromkeys(
                    query
                    for question in unique_questions
                    for query in question_plans[question].retrieval_queries
                )
            )
            query_vectors = await builder.embed(
                unique_retrieval_queries, operation="question_embedding"
            )
            search_rows = await self.workers.run(
                index.search, query_vectors, self.settings.retrieval_k
            )
            rows_by_query = dict(zip(unique_retrieval_queries, search_rows, strict=True))
            contexts = []
            for question in unique_questions:
                queries = question_plans[question].retrieval_queries
                rows = [rows_by_query[query] for query in queries]
                context = merge_retrieval_results(rows, self.settings.retrieval_k)
                contexts.append(context)
            logger.info(
                "retrieval_complete",
                extra={
                    "stage": "retrieval",
                    "duration_ms": round((perf_counter() - started) * 1000, 2),
                },
            )

            async def answer_one(question_number: int, plan: QuestionPlan, context) -> AnswerResult:
                token = question_number_context.set(question_number)
                answer_started = perf_counter()
                logger.info(
                    "answer_task_started",
                    extra={"stage": "generation", "context_chunk_count": len(context)},
                )
                try:
                    generated = await self.provider.call(
                        lambda: self.generator.generate(plan, context),
                        operation="answer_generation",
                        item_count=len(context),
                    )
                    result = validate_answer(plan, context, generated)
                    logger.info(
                        "answer_task_complete",
                        extra={
                            "stage": "generation",
                            "duration_ms": round((perf_counter() - answer_started) * 1000, 2),
                            "supported": result.status != "not_found",
                            "selected_chunk_count": len(result.citations),
                            "citation_count": len(result.citations),
                            "citation_chars": sum(
                                len(citation.excerpt) for citation in result.citations
                            ),
                        },
                    )
                    return result
                except asyncio.CancelledError:
                    logger.info(
                        "answer_task_cancelled",
                        extra={
                            "stage": "generation",
                            "duration_ms": round((perf_counter() - answer_started) * 1000, 2),
                        },
                    )
                    raise
                except ServiceError as exc:
                    logger.warning(
                        "answer_task_failed",
                        extra={
                            "stage": "generation",
                            "duration_ms": round((perf_counter() - answer_started) * 1000, 2),
                            "code": exc.code,
                        },
                    )
                    raise
                finally:
                    question_number_context.reset(token)

            started = perf_counter()
            tasks: dict[str, asyncio.Task] = {}
            try:
                async with asyncio.TaskGroup() as group:
                    plans = [question_plans[question] for question in unique_questions]
                    pairs = zip(plans, contexts, strict=True)
                    for question_number, (plan, context) in enumerate(pairs, start=1):
                        tasks[plan.original_question] = group.create_task(
                            answer_one(question_number, plan, context)
                        )
            except* ServiceError as group_error:
                raise group_error.exceptions[0] from None
            logger.info(
                "generation_complete",
                extra={
                    "stage": "generation",
                    "duration_ms": round((perf_counter() - started) * 1000, 2),
                },
            )
            return QAResponse(results=[tasks[question].result() for question in questions])
        finally:
            index.close()

    async def close(self) -> None:
        self.workers.close()
        if self.http_client:
            await self.http_client.aclose()
        if self.sync_http_client:
            self.sync_http_client.close()
        if self.owned_generator:
            await self.owned_generator.aclose()

"""Orchestrate one document and many questions without embedding the document again."""

import asyncio
import logging
from time import perf_counter

import httpx
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langsmith import tracing_context

from app.config import Settings
from app.errors import ServiceError
from app.generation import AnswerGenerator, OpenAIAnswerGenerator, validate_answer
from app.ingestion import parse_document, parse_questions, split_documents
from app.models import AnswerResult, QAResponse
from app.retrieval import IndexBuilder
from app.runtime import ProviderRunner, WorkerPool

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
            query_vectors = await builder.embed(unique_questions)
            contexts = await self.workers.run(
                index.search, query_vectors, self.settings.retrieval_k
            )
            logger.info(
                "retrieval_complete",
                extra={
                    "stage": "retrieval",
                    "duration_ms": round((perf_counter() - started) * 1000, 2),
                },
            )

            async def answer_one(question, context) -> AnswerResult:
                result = await self.provider.call(
                    lambda: self.generator.generate(question, context)
                )
                return validate_answer(question, context, result)

            started = perf_counter()
            tasks: dict[str, asyncio.Task] = {}
            try:
                async with asyncio.TaskGroup() as group:
                    for question, context in zip(unique_questions, contexts, strict=True):
                        tasks[question] = group.create_task(answer_one(question, context))
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

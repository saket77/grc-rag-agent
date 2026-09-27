"""Deterministic local providers for end-to-end API tests; no credentials or network."""

import asyncio

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.rag.planning import QuestionPlan
from app.schemas.qa import NOT_FOUND, GeneratedAnswer, GeneratedPartAnswer


class DeterministicEmbeddings(Embeddings):
    def __init__(self):
        self.batches: list[list[str]] = []

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        topics = (
            ("cloud", "host", "aws", "azure", "provider"),
            ("encrypt", "aes-256"),
            ("retention", "30 days"),
        )
        return [float(any(term in lowered for term in topic)) for topic in topics] + [0.05]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return self.embed_documents(texts)


class GroundedGenerator:
    def __init__(
        self,
        *,
        delay: float = 0,
        delays_by_question: dict[str, float] | None = None,
        error: Exception | None = None,
        gate: asyncio.Event | None = None,
        invalid_chunk_id: bool = False,
    ):
        self.delay = delay
        self.delays_by_question = delays_by_question or {}
        self.error = error
        self.gate = gate
        self.invalid_chunk_id = invalid_chunk_id
        self.started = asyncio.Event()
        self.calls: list[tuple[QuestionPlan, list[Document]]] = []
        self.completed_questions: list[str] = []
        self.active = 0
        self.peak_active = 0
        self.cancelled = 0

    async def generate(self, plan: QuestionPlan, chunks: list[Document]) -> GeneratedAnswer:
        question = plan.original_question
        self.calls.append((plan, chunks))
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        self.started.set()
        try:
            if self.gate is not None:
                await self.gate.wait()
            delay = self.delays_by_question.get(question, self.delay)
            if delay:
                await asyncio.sleep(delay)
            if self.error:
                raise self.error
            result = self._answer(plan, chunks)
            self.completed_questions.append(question)
            return result
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.active -= 1

    def _answer(self, plan: QuestionPlan, chunks: list[Document]) -> GeneratedAnswer:
        results = []
        for part in plan.parts:
            question_lower = part.question.lower()
            facts = []
            if "cloud" in question_lower or "host" in question_lower:
                facts = [
                    ("AWS", "The service is hosted on AWS."),
                    ("Azure", "The service is hosted on Azure."),
                ]
            elif "encrypt" in question_lower:
                facts = [("AES-256", "Data is encrypted using AES-256.")]
            elif "retention" in question_lower:
                facts = [("30 days", "Retention is 30 days.")]
            generated = GeneratedPartAnswer(
                part_id=part.part_id,
                status="not_found",
                answer=NOT_FOUND,
                evidence_chunk_ids=[],
            )
            for marker, answer in facts:
                source = next((chunk for chunk in chunks if marker in chunk.page_content), None)
                if source is not None:
                    generated = GeneratedPartAnswer(
                        part_id=part.part_id,
                        status="supported",
                        answer=answer,
                        evidence_chunk_ids=[
                            "invented" if self.invalid_chunk_id else source.metadata["chunk_id"]
                        ],
                    )
                    break
            results.append(generated)
        return GeneratedAnswer(parts=results)

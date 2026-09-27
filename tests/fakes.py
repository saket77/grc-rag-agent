"""Deterministic local providers for end-to-end API tests; no credentials or network."""

import asyncio

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.models import NOT_FOUND, EvidenceQuote, GeneratedAnswer


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
        invalid_citation: str | None = None,
    ):
        self.delay = delay
        self.delays_by_question = delays_by_question or {}
        self.error = error
        self.gate = gate
        self.invalid_citation = invalid_citation
        self.started = asyncio.Event()
        self.calls: list[tuple[str, list[Document]]] = []
        self.completed_questions: list[str] = []
        self.active = 0
        self.peak_active = 0
        self.cancelled = 0

    async def generate(self, question: str, chunks: list[Document]) -> GeneratedAnswer:
        self.calls.append((question, chunks))
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
            result = self._answer(question, chunks)
            self.completed_questions.append(question)
            return result
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.active -= 1

    def _answer(self, question: str, chunks: list[Document]) -> GeneratedAnswer:
        question_lower = question.lower()
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
        for quote, answer in facts:
            for chunk in chunks:
                if quote in chunk.page_content:
                    return GeneratedAnswer(
                        supported=True,
                        answer=answer,
                        evidence=[
                            EvidenceQuote(
                                chunk_id="invented"
                                if self.invalid_citation == "id"
                                else chunk.metadata["chunk_id"],
                                quote="fabricated passage"
                                if self.invalid_citation == "quote"
                                else quote,
                            )
                        ],
                    )
        return GeneratedAnswer(supported=False, answer=NOT_FOUND, evidence=[])

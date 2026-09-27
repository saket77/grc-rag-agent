"""Request-owned native FAISS and BM25 indexes with deterministic rank fusion."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from heapq import nsmallest

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.core.config import Settings
from app.core.errors import ServiceError
from app.core.runtime import ProviderRunner, WorkerPool

MIN_CANDIDATES = 20
CANDIDATE_MULTIPLIER = 4
RRF_OFFSET = 60
BM25_K1 = 1.2
BM25_B = 0.75
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
# Normalize a small set of ordinary inflections and one direct communication synonym. This keeps
# lexical matching deterministic without creating extra retrieval queries or provider calls.
_LEXICAL_ALIASES = {
    "accessed": "access",
    "accesses": "access",
    "accessing": "access",
    "disclosed": "disclose",
    "discloses": "disclose",
    "disclosing": "disclose",
    "inform": "notify",
    "informed": "notify",
    "informing": "notify",
    "informs": "notify",
    "managed": "manage",
    "manages": "manage",
    "managing": "manage",
    "notification": "notify",
    "notifications": "notify",
    "notified": "notify",
    "notifies": "notify",
    "notifying": "notify",
    "parties": "party",
    "processed": "process",
    "processes": "process",
    "processing": "process",
    "providers": "provider",
    "retained": "retain",
    "retains": "retain",
    "retaining": "retain",
    "stored": "store",
    "stores": "store",
    "storing": "store",
    "vendors": "vendor",
}


def checked_vectors(vectors: list[list[float]], count: int) -> np.ndarray:
    try:
        result = np.asarray(vectors, dtype=np.float32, order="C")
        if (
            result.ndim != 2
            or result.shape[0] != count
            or result.shape[1] == 0
            or not np.isfinite(result).all()
        ):
            raise ValueError
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            norms = np.linalg.norm(result, axis=1)
        if not np.isfinite(norms).all() or (norms == 0).any():
            raise ValueError
        return result
    except (ValueError, TypeError, OverflowError):
        raise ServiceError(
            502, "invalid_embeddings", "The embedding service returned invalid vectors."
        ) from None


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(
        _LEXICAL_ALIASES.get(token, token)
        for match in _TOKEN.finditer(text)
        if (token := match.group().casefold())
    )


@dataclass(frozen=True)
class LexicalIndex:
    """Small request-local Okapi BM25 index over source chunk text."""

    document_count: int
    postings: dict[str, tuple[tuple[int, float], ...]]

    @classmethod
    def build(cls, documents: tuple[Document, ...]) -> LexicalIndex:
        term_counts: list[Counter[str]] = []
        lengths: list[int] = []
        document_frequencies: Counter[str] = Counter()
        for document in documents:
            terms = _tokens(document.page_content)
            lengths.append(len(terms))
            counts = Counter(terms)
            term_counts.append(counts)
            document_frequencies.update(counts.keys())
        document_count = len(documents)
        average_length = sum(lengths) / document_count if document_count else 0.0
        postings: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
        for position, counts in enumerate(term_counts):
            for term, term_frequency in counts.items():
                frequency = document_frequencies[term]
                # A term in every chunk cannot discriminate between candidates.
                if frequency == document_count:
                    continue
                inverse_frequency = math.log(
                    1 + (document_count - frequency + 0.5) / (frequency + 0.5)
                )
                length_ratio = lengths[position] / average_length
                denominator = term_frequency + BM25_K1 * (1 - BM25_B + BM25_B * length_ratio)
                score = inverse_frequency * term_frequency * (BM25_K1 + 1) / denominator
                postings[term].append((position, score))
        return cls(
            document_count=document_count,
            postings={term: tuple(values) for term, values in postings.items()},
        )

    def search(self, query: str, limit: int) -> list[int]:
        if limit <= 0 or not self.document_count:
            return []
        scores: defaultdict[int, float] = defaultdict(float)
        for term in dict.fromkeys(_tokens(query)):
            matches = self.postings.get(term)
            if not matches:
                continue
            for position, score in matches:
                scores[position] += score
        return [
            position
            for position, _ in nsmallest(
                limit, scores.items(), key=lambda item: (-item[1], item[0])
            )
        ]


def reciprocal_rank_fusion(rankings: list[list[int]], limit: int) -> list[int]:
    """Fuse rank-only candidate lists with stable, deterministic tie breaking."""
    scores: defaultdict[int, float] = defaultdict(float)
    first_seen: dict[int, int] = {}
    sequence = 0
    for ranking in rankings:
        seen_in_ranking: set[int] = set()
        for rank, position in enumerate(ranking, start=1):
            if position in seen_in_ranking:
                continue
            seen_in_ranking.add(position)
            if position not in first_seen:
                first_seen[position] = sequence
                sequence += 1
            scores[position] += 1 / (RRF_OFFSET + rank)
    return sorted(
        scores,
        key=lambda position: (-scores[position], first_seen[position], position),
    )[:limit]


@dataclass
class DocumentIndex:
    index: faiss.IndexFlatIP | None
    documents: tuple[Document, ...]
    lexical: LexicalIndex | None

    def _vector_positions(self, vectors: list[list[float]], k: int) -> list[list[int]]:
        faiss.omp_set_num_threads(1)
        index = self.index
        if index is None:
            raise RuntimeError("Index is closed")
        matrix = checked_vectors(vectors, len(vectors))
        if matrix.shape[1] != index.d:
            raise ServiceError(502, "invalid_embeddings", "Embedding dimensions do not match.")
        matrix = matrix.copy()
        faiss.normalize_L2(matrix)
        _, positions = index.search(matrix, min(k, len(self.documents)))
        return [[int(position) for position in row if position != -1] for row in positions]

    def vector_search(self, vectors: list[list[float]], k: int) -> list[list[Document]]:
        """Expose vector-only retrieval for equivalence tests and the microbenchmark."""
        documents = self.documents
        return [
            [documents[position] for position in row] for row in self._vector_positions(vectors, k)
        ]

    def search(
        self, queries: list[str], vectors: list[list[float]], k: int
    ) -> list[list[Document]]:
        if len(queries) != len(vectors):
            raise ValueError("Queries and vectors must have the same length")
        lexical = self.lexical
        if lexical is None:
            raise RuntimeError("Index is closed")
        candidate_count = min(len(self.documents), max(MIN_CANDIDATES, CANDIDATE_MULTIPLIER * k))
        vector_rows = self._vector_positions(vectors, candidate_count)
        documents = self.documents
        contexts = []
        for query, vector_positions in zip(queries, vector_rows, strict=True):
            lexical_positions = lexical.search(query, candidate_count)
            fused = reciprocal_rank_fusion([vector_positions, lexical_positions], k)
            contexts.append([documents[position] for position in fused])
        return contexts

    def close(self) -> None:
        # Drop ownership without mutating a native index a cancelled worker may still read.
        self.index = None
        self.documents = ()
        self.lexical = None


def build_index(chunks: list[Document], vectors: list[list[float]]) -> DocumentIndex:
    faiss.omp_set_num_threads(1)
    matrix = checked_vectors(vectors, len(chunks)).copy()
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    documents = tuple(chunks)
    return DocumentIndex(index=index, documents=documents, lexical=LexicalIndex.build(documents))


def merge_retrieval_results(rows: list[list[Document]], k: int) -> list[Document]:
    """Keep the original top-k, then fairly add derived-query results up to 2k."""
    if not rows:
        raise ValueError("At least the original retrieval result is required")
    limit = 2 * k
    selected: list[Document] = []
    seen: set[str] = set()

    def add(document: Document) -> None:
        chunk_id = document.metadata.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise RuntimeError("Retrieved source document has no chunk ID")
        if chunk_id not in seen and len(selected) < limit:
            selected.append(document)
            seen.add(chunk_id)

    for document in rows[0][:k]:
        add(document)
    for rank in range(k):
        for row in rows[1:]:
            if rank < len(row):
                add(row[rank])
    return selected


class IndexBuilder:
    def __init__(
        self,
        embeddings: Embeddings,
        settings: Settings,
        provider: ProviderRunner,
        workers: WorkerPool,
    ):
        self.embeddings = embeddings
        self.settings = settings
        self.provider = provider
        self.workers = workers

    async def embed(self, texts: list[str], *, operation: str = "embedding") -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch_number, start in enumerate(
            range(0, len(texts), self.settings.embedding_batch_size), start=1
        ):
            batch = texts[start : start + self.settings.embedding_batch_size]
            values = await self.provider.call(
                lambda batch=batch: self.embeddings.aembed_documents(batch),
                operation=operation,
                batch_number=batch_number,
                item_count=len(batch),
            )
            # Validate each response before combining batches; missing items cannot shift IDs.
            await self.workers.run(checked_vectors, values, len(batch))
            vectors.extend(values)
        return vectors

    async def build(self, chunks: list[Document]) -> DocumentIndex:
        vectors = await self.embed(
            [chunk.page_content for chunk in chunks], operation="document_embedding"
        )
        return await self.workers.run(build_index, chunks, vectors)


# Keep FAISS's CPU parallelism bounded alongside the application's own worker pool.
faiss.omp_set_num_threads(1)

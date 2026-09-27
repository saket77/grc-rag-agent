import faiss
import numpy as np
import pytest
from langchain_core.documents import Document

from app.core.config import Settings
from app.core.errors import ServiceError
from app.core.runtime import ProviderRunner, WorkerPool
from app.rag.planning import build_question_plan
from app.rag.retrieval import (
    DocumentIndex,
    IndexBuilder,
    build_index,
    checked_vectors,
    merge_retrieval_results,
)
from tests.fakes import DeterministicEmbeddings


@pytest.mark.parametrize(
    "vectors",
    [
        [],
        [[1.0], [2.0]],
        [[float("nan")]],
        [[float("inf")]],
        [[0.0]],
        [["oops"]],
        [[1e38, 1e38]],
        [[1e-30, 1e-30]],
    ],
)
def test_invalid_embedding_responses_are_rejected(vectors):
    with pytest.raises(ServiceError) as caught:
        checked_vectors(vectors, 1)
    assert caught.value.status_code == 502


@pytest.mark.parametrize("neighbors", [1, 6])
def test_batched_search_matches_individual_search_and_uses_one_native_call(monkeypatch, neighbors):
    embeddings = DeterministicEmbeddings()
    texts = ["Hosted on AWS", "Uses AES-256", "Retain for 30 days"]
    chunks = [
        Document(page_content=text, metadata={"chunk_id": str(index), "page": index + 1})
        for index, text in enumerate(texts)
    ]
    store = build_index(chunks, embeddings.embed_documents(texts), embeddings)
    queries = embeddings.embed_documents(["What encryption?", "Which cloud?", "Retention?"])
    expected = [store.similarity_search_by_vector(vector, k=neighbors) for vector in queries]
    native_search = store.index.search
    seen_shapes = []

    def observed_search(matrix, count):
        seen_shapes.append(matrix.shape)
        np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1, rtol=1e-6)
        return native_search(matrix, count)

    monkeypatch.setattr(store.index, "search", observed_search)
    index = DocumentIndex(store)
    try:
        assert index.search(queries, neighbors) == expected
        assert seen_shapes == [(3, 4)]
    finally:
        index.close()


async def test_embedding_batches_and_actual_worker_faiss_thread_limit():
    settings = Settings(_env_file=None, embedding_batch_size=2)
    embeddings = DeterministicEmbeddings()
    workers = WorkerPool(1)
    builder = IndexBuilder(embeddings, settings, ProviderRunner(settings), workers)
    chunks = [
        Document(page_content=text, metadata={"chunk_id": str(i), "page": i + 1})
        for i, text in enumerate(["Hosted on AWS", "Uses AES-256", "Retain for 30 days"])
    ]
    index = None
    try:
        index = await builder.build(chunks)
        assert list(map(len, embeddings.batches)) == [2, 1]
        assert await workers.run(faiss.omp_get_max_threads) == 1
        vectors = await builder.embed(["What encryption?"])
        result = await workers.run(index.search, vectors, 1)
        assert result[0][0].metadata["page"] == 2
    finally:
        if index:
            index.close()
        workers.close()


def test_query_dimension_mismatch_has_clear_provider_error():
    chunks = [Document(page_content="AWS", metadata={"chunk_id": "one", "page": 1})]
    index = DocumentIndex(build_index(chunks, [[1.0, 0.0]], DeterministicEmbeddings()))
    try:
        with pytest.raises(ServiceError) as caught:
            index.search([[1.0]], 1)
        assert caught.value.code == "invalid_embeddings"
    finally:
        index.close()


@pytest.mark.parametrize(
    ("question", "expected_parts", "expected_queries"),
    [
        (
            "Do you notify clients during incidents? What is the notification SLA?",
            [
                "Do you notify clients during incidents?",
                "What is the notification SLA?",
            ],
            [
                "Do you notify clients during incidents? What is the notification SLA?",
                "Do you notify clients during incidents?",
                "What is the notification SLA?",
            ],
        ),
        (
            "Is personal information handled by third parties? If yes, describe.",
            ["Is personal information handled by third parties? If yes, describe."],
            [
                "Is personal information handled by third parties? If yes, describe.",
            ],
        ),
        (
            "Which monitoring exists: 1. APM, 2. EUM, 3. DEM?",
            [
                "Which monitoring exists APM?",
                "Which monitoring exists EUM?",
                "Which monitoring exists DEM?",
            ],
            [
                "Which monitoring exists: 1. APM, 2. EUM, 3. DEM?",
                "Which monitoring exists APM?",
                "Which monitoring exists EUM?",
                "Which monitoring exists DEM?",
            ],
        ),
        (
            "Which cloud provider?",
            ["Which cloud provider?"],
            ["Which cloud provider?"],
        ),
    ],
)
def test_one_question_plan_drives_model_parts_and_retrieval_queries(
    question, expected_parts, expected_queries
):
    plan = build_question_plan(question)

    assert [part.question for part in plan.parts] == expected_parts
    assert list(plan.retrieval_queries) == expected_queries
    assert list(plan.retrieval_queries) == list(
        dict.fromkeys([plan.original_question, *(part.question for part in plan.parts)])
    )


def test_retrieval_merge_preserves_original_then_round_robins_derived_results():
    def hit(chunk_id):
        return Document(page_content=chunk_id, metadata={"chunk_id": chunk_id})

    rows = [
        [hit("original-1"), hit("shared")],
        [hit("shared"), hit("derived-1")],
        [hit("derived-2"), hit("derived-3")],
    ]

    assert [document.metadata["chunk_id"] for document in merge_retrieval_results(rows, k=2)] == [
        "original-1",
        "shared",
        "derived-2",
        "derived-1",
    ]


def test_numbered_labels_preserve_option_text_without_changing_search_queries():
    question = "Which controls exist: 1. Encryption at rest, 2. Encryption in transit?"
    plan = build_question_plan(question)

    assert [part.label for part in plan.parts] == ["Encryption at rest", "Encryption in transit"]
    assert list(plan.retrieval_queries) == [
        question,
        "Which controls exist Encryption at rest?",
        "Which controls exist Encryption in transit?",
    ]
    assert build_question_plan("Which controls exist?").parts[0].label is None


async def test_provider_construction_and_shutdown_are_offline():
    from app.services.qa import QAService

    service = QAService(Settings(_env_file=None, openai_api_key="test-not-live"))
    try:
        service._configure_clients()
        async_client = service.http_client
        sync_client = service.sync_http_client
    finally:
        await service.close()
    assert async_client.is_closed
    assert sync_client.is_closed

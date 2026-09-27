import faiss
import numpy as np
import pytest
from langchain_core.documents import Document

from app.core.config import Settings
from app.core.errors import ServiceError
from app.core.runtime import ProviderRunner, WorkerPool
from app.rag.planning import build_question_plan
from app.rag.retrieval import (
    IndexBuilder,
    LexicalIndex,
    build_index,
    checked_vectors,
    merge_retrieval_results,
    reciprocal_rank_fusion,
)
from tests.fakes import DeterministicEmbeddings


def documents(*texts):
    return [
        Document(page_content=text, metadata={"chunk_id": str(index), "page": index + 1})
        for index, text in enumerate(texts)
    ]


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
def test_batched_vector_search_matches_individual_search_and_uses_one_native_call(neighbors):
    embeddings = DeterministicEmbeddings()
    chunks = documents("Hosted on AWS", "Uses AES-256", "Retain for 30 days")
    vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
    queries = embeddings.embed_documents(["What encryption?", "Which cloud?", "Retention?"])
    index = build_index(chunks, vectors)
    expected = [index.vector_search([vector], neighbors)[0] for vector in queries]
    native = index.index
    seen_shapes = []

    class ObservedIndex:
        d = native.d

        @staticmethod
        def search(matrix, count):
            seen_shapes.append(matrix.shape)
            np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1, rtol=1e-6)
            return native.search(matrix, count)

    index.index = ObservedIndex()
    try:
        assert index.vector_search(queries, neighbors) == expected
        assert seen_shapes == [(3, 4)]
    finally:
        index.close()


def test_lexical_search_is_case_insensitive_ranked_and_deterministic():
    chunks = tuple(
        documents(
            "Generic vendor policy.",
            "Vendors process information assets.",
            "Vendors access, process, store, and manage information assets.",
        )
    )
    lexical = LexicalIndex.build(chunks)

    assert lexical.search("VENDORS process store information assets", 3) == [2, 1]
    assert lexical.search("term-not-present", 3) == []


def test_lexical_search_normalizes_common_verbs_without_expanding_queries():
    chunks = tuple(
        documents(
            "The team will inform necessary parties without undue delay.",
            "Providers are accessing, processing, storing, or managing information assets.",
            "Unrelated background.",
        )
    )
    lexical = LexicalIndex.build(chunks)

    assert lexical.search("notification timing", 3) == [0]
    assert lexical.search("vendors accessed and processed stored information assets", 3)[0] == 1


def test_reciprocal_rank_fusion_deduplicates_and_breaks_ties_stably():
    assert reciprocal_rank_fusion([[0, 1, 2, 2], [2, 1, 3]], 4) == [2, 1, 0, 3]
    assert reciprocal_rank_fusion([[4, 5], []], 2) == [4, 5]


def test_hybrid_search_promotes_strong_lexical_match_when_vector_rank_is_weak():
    chunks = documents(
        "Generic policy background.",
        "Unrelated third party review.",
        "Vendors process and store information assets.",
    )
    index = build_index(chunks, [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    try:
        result = index.search(
            ["Which vendors process and store information assets?"],
            [[1.0, 0.0]],
            1,
        )
        assert result == [[chunks[2]]]
    finally:
        index.close()


def test_hybrid_search_falls_back_to_vector_order_without_lexical_matches():
    chunks = documents("Hosted on AWS", "Uses AES-256")
    index = build_index(chunks, [[1.0, 0.0], [0.0, 1.0]])
    try:
        assert index.search(["unseen vocabulary"], [[1.0, 0.0]], 6) == [[chunks[0], chunks[1]]]
    finally:
        index.close()


@pytest.mark.parametrize(("neighbors", "expected_candidates"), [(2, 20), (6, 24)])
def test_hybrid_search_uses_bounded_candidate_depth(neighbors, expected_candidates):
    chunks = documents(*(f"record {number}" for number in range(30)))
    vectors = [[1.0, number / 100] for number in range(30)]
    index = build_index(chunks, vectors)
    native = index.index
    seen_counts = []

    class ObservedIndex:
        d = native.d

        @staticmethod
        def search(matrix, count):
            seen_counts.append(count)
            return native.search(matrix, count)

    index.index = ObservedIndex()
    try:
        assert len(index.search(["unseen vocabulary"], [[1.0, 0.0]], neighbors)[0]) == neighbors
        assert seen_counts == [expected_candidates]
    finally:
        index.close()


def test_hybrid_search_validates_query_count_and_releases_owned_state():
    chunks = documents("Hosted on AWS")
    index = build_index(chunks, [[1.0, 0.0]])
    with pytest.raises(ValueError):
        index.search([], [[1.0, 0.0]], 1)
    index.close()
    assert index.index is None and index.documents == () and index.lexical is None
    with pytest.raises(RuntimeError):
        index.search(["cloud"], [[1.0, 0.0]], 1)


async def test_embedding_batches_and_actual_worker_faiss_thread_limit():
    settings = Settings(_env_file=None, embedding_batch_size=2)
    embeddings = DeterministicEmbeddings()
    workers = WorkerPool(1)
    builder = IndexBuilder(embeddings, settings, ProviderRunner(settings), workers)
    chunks = documents("Hosted on AWS", "Uses AES-256", "Retain for 30 days")
    index = None
    try:
        index = await builder.build(chunks)
        assert list(map(len, embeddings.batches)) == [2, 1]
        assert await workers.run(faiss.omp_get_max_threads) == 1
        queries = ["What encryption?"]
        vectors = await builder.embed(queries)
        result = await workers.run(index.search, queries, vectors, 1)
        assert result[0][0].metadata["page"] == 2
    finally:
        if index:
            index.close()
        workers.close()


def test_query_dimension_mismatch_has_clear_provider_error():
    chunks = documents("AWS")
    index = build_index(chunks, [[1.0, 0.0]])
    try:
        with pytest.raises(ServiceError) as caught:
            index.search(["cloud"], [[1.0]], 1)
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
            ["Is personal information handled by third parties? If yes, describe."],
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

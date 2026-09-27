"""Compare batched FAISS retrieval with the previous per-question loop, offline."""

import argparse
import json
from statistics import median
from time import perf_counter

import numpy as np
from langchain_core.documents import Document

from app.rag.retrieval import DocumentIndex, build_index, checked_vectors
from tests.fakes import DeterministicEmbeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=2000)
    parser.add_argument("--questions", type=int, default=50)
    parser.add_argument("--dimensions", type=int, default=1536)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if min(args.documents, args.questions, args.dimensions, args.repeats) < 1:
        parser.error("All sizes and repeat counts must be positive.")
    random = np.random.default_rng(42)
    vectors = random.normal(size=(args.documents, args.dimensions)).astype(np.float32).tolist()
    queries = random.normal(size=(args.questions, args.dimensions)).astype(np.float32).tolist()
    chunks = [
        Document(page_content=f"Synthetic record {number}", metadata={"chunk_id": str(number)})
        for number in range(args.documents)
    ]
    store = build_index(chunks, vectors, DeterministicEmbeddings())
    index = DocumentIndex(store)

    def previous_loop():
        matrix = checked_vectors(queries, len(queries))
        return [store.similarity_search_by_vector(vector.tolist(), k=6) for vector in matrix]

    def batched():
        return index.search(queries, 6)

    try:
        if previous_loop() != batched():
            raise RuntimeError("Retrieved documents differ between implementations")
        samples = {"previous_loop": [], "batched": []}
        for repeat in range(args.repeats):
            operations = [("previous_loop", previous_loop), ("batched", batched)]
            if repeat % 2:
                operations.reverse()
            for name, operation in operations:
                started = perf_counter()
                operation()
                samples[name].append((perf_counter() - started) * 1000)
        medians = {name: round(median(values), 3) for name, values in samples.items()}
        print(
            json.dumps(
                {
                    **vars(args),
                    "native_threads": 1,
                    "median_ms": medians,
                    "speedup": round(medians["previous_loop"] / medians["batched"], 2),
                    "results_match": True,
                    "scope": "Retrieval only; excludes ingestion, embeddings, and generation",
                },
                indent=2,
            )
        )
    finally:
        index.close()


if __name__ == "__main__":
    main()

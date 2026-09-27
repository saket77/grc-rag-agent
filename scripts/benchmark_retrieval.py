"""Measure vector batching and deterministic hybrid-search overhead offline."""

import argparse
import json
from statistics import median
from time import perf_counter

import numpy as np
from langchain_core.documents import Document

from app.rag.retrieval import build_index


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
    query_vectors = (
        random.normal(size=(args.questions, args.dimensions)).astype(np.float32).tolist()
    )
    queries = [
        f"Which synthetic record matches question {number}?" for number in range(args.questions)
    ]
    chunks = [
        Document(page_content=f"Synthetic record {number}", metadata={"chunk_id": str(number)})
        for number in range(args.documents)
    ]
    index = build_index(chunks, vectors)

    def vector_loop():
        return [index.vector_search([vector], 6)[0] for vector in query_vectors]

    def vector_batch():
        return index.vector_search(query_vectors, 6)

    def hybrid():
        return index.search(queries, query_vectors, 6)

    try:
        if vector_loop() != vector_batch():
            raise RuntimeError("Retrieved documents differ between implementations")
        samples = {"vector_loop": [], "vector_batch": [], "hybrid": []}
        for repeat in range(args.repeats):
            operations = [
                ("vector_loop", vector_loop),
                ("vector_batch", vector_batch),
                ("hybrid", hybrid),
            ]
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
                    "vector_batch_speedup": round(
                        medians["vector_loop"] / medians["vector_batch"], 2
                    ),
                    "hybrid_overhead_ms": round(medians["hybrid"] - medians["vector_batch"], 3),
                    "vector_results_match": True,
                    "scope": (
                        "Retrieval only; hybrid includes BM25 and rank fusion but excludes "
                        "ingestion, embeddings, and generation"
                    ),
                },
                indent=2,
            )
        )
    finally:
        index.close()


if __name__ == "__main__":
    main()

# Zania Document QA

A document-grounded question-answering service for PDF and JSON uploads. It combines hybrid
BM25/vector retrieval with structured evidence coverage and source-owned citations so unsupported
questionnaire claims remain explicit rather than being inferred from adjacent controls.

## What the service does

- Accepts one PDF or JSON document and a JSON array of questions.
- Normalizes and chunks the source without crossing PDF pages or JSON records.
- Embeds the document once and searches all unique question vectors in one native FAISS call.
- Fuses semantic and lexical rankings deterministically with reciprocal-rank fusion.
- Separates full, partial, related-only, and absent evidence internally.
- Returns public found, partial, or not_found results in the submitted question order.
- Lets the model select server-issued chunk IDs; the server resolves citation pages and excerpts.
- Bounds uploads, concurrent requests, provider calls, retries, and request deadlines.

The answer model is fixed to gpt-4o-mini; embeddings default to text-embedding-3-small.

## Run with Docker

Requirements:

- Docker Desktop or Docker Engine with Compose v2
- An OpenAI API key

    cp .env.example .env
    # Set OPENAI_API_KEY in .env
    docker compose up --build

Open:

- Upload UI: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- API documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health check: [http://127.0.0.1:8000/healthz](http://127.0.0.1:8000/healthz)

The container runs as a non-root user and binds the host port to loopback. Stop it with:

    docker compose down

## Local development

Python 3.12 is required.

    make install
    make doctor
    make test
    make check
    make dev

Tests use deterministic providers and block external sockets. A live /qa request uses the
configured OpenAI providers and incurs normal API usage.

## API

POST /qa accepts exactly two multipart/form-data file fields:

- questions: a .json file containing a nonempty array of nonblank strings.
- document: a text-extractable .pdf or a .json object/array.

    curl --fail-with-body http://127.0.0.1:8000/qa \
      -F 'questions=@examples/questions.json;type=application/json' \
      -F 'document=@examples/document.json;type=application/json'

A successful response preserves question order:

    {
      "results": [
        {
          "question": "Which cloud providers do you rely on?",
          "answer": "The production service is hosted on AWS.",
          "status": "found",
          "citations": [
            {
              "page": null,
              "excerpt": "The production service is hosted on AWS."
            }
          ]
        },
        {
          "question": "How frequently do you conduct penetration tests?",
          "answer": "Not found in document",
          "status": "not_found",
          "citations": []
        }
      ]
    }

| Status | Meaning |
| --- | --- |
| found | Every requested part is supported by retrieved evidence. |
| partial | At least one requested fact is supported, but another fact or qualifier is missing. |
| not_found | No requested fact is supported; related background does not count as partial support. |

Coverage is not a confidence score. The server validates source identity and citation provenance;
semantic correctness is measured separately with live regression evaluation.

PDF citations use one-based physical page numbers. JSON citations have page null and quote the
deterministically normalized source chunk.

Every response includes X-Request-ID. Errors use a sanitized envelope. Provider failures are never
converted into not_found. A request is all-or-nothing: one failed answer task cancels its siblings.

## Architecture

The request path is:

1. Validate uploads, normalize source text, and build page/record-local chunks.
2. Batch document embeddings into a request-owned native FAISS index.
3. Build a request-owned BM25 inverted index over the same source text.
4. Create deterministic original and explicit-part retrieval queries.
5. Embed all unique queries together and search FAISS once with the full matrix.
6. Fuse semantic and lexical candidates with reciprocal-rank fusion.
7. Merge bounded context and make one strict structured generation call per unique question.
8. Validate coverage and selected chunk IDs, then resolve citations from source-owned chunks.

### Retrieval

For every deterministic original/part query the service:

1. searches a bounded candidate depth in one FAISS matrix call;
2. ranks lexical candidates with Okapi BM25;
3. fuses the two rank lists with reciprocal-rank fusion; and
4. keeps the original query's top k, then round-robins unique part-query results up to 2k.

This retains semantic recall while giving exact compliance terms such as acronyms, policy names,
and data-handling verbs a direct retrieval path. It adds no provider call.

### Answer coverage

The private generation schema distinguishes:

- full: every requested fact and relationship is established;
- partial: at least one requested fact is established, but another is missing;
- related_only: evidence is topical but answers a different property or relationship; and
- none: no relevant evidence exists.

Only full and partial may select evidence IDs. Related-only and absent evidence become public
not_found. The model never supplies authoritative excerpts or page numbers.

## Measured performance

The final fixed retrieval benchmark (2,000 documents, 50 questions, 1,536 dimensions) measured
19.350 ms for per-query vector search, 17.104 ms for one matrix FAISS search, and 17.323 ms for full
hybrid retrieval. Matrix search was 1.13× faster, and BM25 plus reciprocal-rank fusion added 0.219 ms.

In the ten-run live regression, mean end-to-end latency improved from 16.321 s to 13.178 s. All ten
requests passed the semantic and citation gates. See [EVALUATION.md](EVALUATION.md) for fixture
hashes, methodology, final results, and limitations.

## Verification

    make check
    make test
    .venv/bin/python -m scripts.benchmark_retrieval
    bash .github/ci/e2e.sh

CI also builds the Python package, validates Compose, builds the production image, and smoke-tests
health, OpenAPI, UI assets, and deterministic multipart /qa behavior.

The retrieval benchmark reports vector-loop time, one-call matrix-search time, and hybrid-search
overhead separately. It excludes parsing, provider embeddings, and generation.

The public Nave SOC 2 report and appendix questions under examples are evaluation fixtures, not
production-specific rules.

## Configuration

All settings are validated in app/core/config.py and documented in .env.example.

| Setting | Default |
| --- | --- |
| MAX_DOCUMENT_BYTES | 10 MiB |
| MAX_QUESTIONS_BYTES | 128 KiB |
| MAX_QUESTIONS / MAX_QUESTION_CHARS | 50 / 2,000 |
| MAX_PDF_PAGES / MAX_EXTRACTED_CHARS | 200 / 1,000,000 |
| MAX_CHUNKS / MAX_JSON_DEPTH | 2,000 / 64 |
| CHUNK_SIZE / CHUNK_OVERLAP | 1,000 / 400 characters |
| RETRIEVAL_K / EMBEDDING_BATCH_SIZE | 6 / 64 |
| MAX_PROVIDER_CALLS / MAX_ACTIVE_REQUESTS | 4 / 2 per process |
| REQUEST_TIMEOUT_SECONDS / PROVIDER_TIMEOUT_SECONDS | 120 / 30 seconds |

## Current limitations

- PDF ingestion does not perform OCR or reconstruct complex table layouts.
- JSON citations identify normalized chunks, not exact source paths or character ranges.
- There is no authentication, durable tenancy, persistent vector store, or cross-request cache.
- Limits are per process; multiple Uvicorn workers multiply memory and concurrency limits.
- Citation identity validation does not mathematically prove semantic entailment.
- Live evaluation on one public SOC 2 fixture measures a regression case, not broad-document
  generalization or production readiness.

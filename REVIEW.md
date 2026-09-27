# Backend review and reading plan

Review date: 2026-09-27. This is an implementation review, not an evaluator's score or a claim of production readiness. The original review ran 123 tests. Follow-up work adds regression coverage for the defects below, native FAISS batch search, deterministic query decomposition, structured partial answers, and a minimal upload UI.

Current verification: **193 offline tests pass**, and the full Python lint and formatting checks pass. The credential-free HTTP CI smoke test and Python package build also pass; the built wheel imports and serves its health endpoint and UI assets. The latest generation changes add generic valid-response examples and an exact per-request part schema. A subsequent ten-request live evaluation returned HTTP 200 every time (mean 11.715 seconds), with no invalid-part errors. Its public statuses were consistently `partial, partial, found, not_found, not_found`. Q3 remained correct and Q4 correctly abstained, but Q1 still overclaimed notification criteria, Q2 labeled background as partial support, and Q5 omitted the documented monitoring signals. All 56 citation instances matched their source pages. Full retrieved contexts were not captured; valid response shape and source identity do not establish semantic correctness.

Historical evaluation (before the latest generation changes): a ten-run live evaluation against the supplied five-question SOC 2 fixture produced the same support pattern and citation pages on every run: Q1, Q3, and Q5 answered; Q2 and Q4 abstained. Selected retrieval contexts were stable; one Q1 derived query swapped ranks four and five without changing the selected set. Mean end-to-end time was 12.27 seconds on the review machine. The evaluated 1,000/400 chunking produced 324 chunks and six document-embedding batches, versus 273 chunks and five batches at the prior 1,000/200 setting. Credentials and full-text diagnostic traces were not retained. Those results are not a validation of the current generation prompt.

## Bug ledger

Fixed entries retain their original reproduction so a reviewer can understand the before/after behavior. Proposed feature work is tracked separately in [NEXT_STEPS.md](docs/NEXT_STEPS.md).

| ID | Priority | Status | Regression evidence |
| --- | --- | --- | --- |
| B01 | P2 | Fixed: reject lone surrogates before providers | Ingestion tests for nested keys/values and valid emoji; endpoint tests assert zero provider calls |
| B02 | P2 | Fixed: preserve meaningful empty fields | Root-object and array-record tests with empty list/object, null, empty string, false, and zero |
| B03 | P3 | Fixed: preserve decimal values | Decimal/exponent/underflow tests verify numeric round trips; unsupported numbers fail clearly |
| B04 | P3 | Fixed: reject unsafe embedding norms | Overflowing/underflowing float32 norm tests |
| B05 | P2 | Fixed: normalize PDF extraction whitespace | Real-PDF ingestion and endpoint tests verify clean source/citation text before providers |
| P01 | Improvement | Implemented: native batch search | One-search-call assertion and equivalence to individual retrieval, including fewer-than-k documents |
| P02 | Improvement | Implemented: deterministic subquery retrieval and partial answers | Structural decomposition/merge tests, strict part validation, and ten-run live behavior matrix |
| U01 | Requirement | Implemented: minimal upload UI | Static routes tested offline; browser upload success/loading/abstention and validation error verified with fake providers |
| J01 | Improvement | Open: preserve JSON parent context and expose source paths | Concrete nested-pages reproduction and acceptance criteria in NEXT_STEPS.md |
| D01 | Maintenance | Open: evaluate FAISS integration migration | The pinned langchain-community integration emits a deprecation warning; current tests pass |

### B01 — Fixed: reject lone Unicode surrogates before paid calls

Location: `app/rag/ingestion.py`, `_load_json` and `parse_questions`.

Before the fix, JSON bytes `b'["\\ud800"]'` decoded successfully, and `json.loads` produced a string that could not be encoded as UTF-8. Question validation accepted it. An endpoint reproduction with fake providers executed document embedding, question embedding, and generation, then returned **500** when encoding the response. JSON document values and keys had the same validation gap.

Implemented: validate all decoded string values and object keys, rejecting lone surrogates with **422 before any provider call**. Valid surrogate pairs (such as escaped emoji) remain supported. Tests cover questions, nested values/keys, and valid Unicode.

### B02 — Fixed: do not discard JSON fields that represent absence

Location: `app/rag/ingestion.py`, `_has_content` and `_parse_json_document`.

Before the fix, given `[{"third_party_processors": []}, {"hosting": "AWS"}]`, only the hosting record survived ingestion. The processor record was discarded because its leaf values were empty. An explicit empty list can be relevant compliance evidence; a nonempty object carries meaning in its keys.

Implemented: retain nonempty objects even when their values are empty lists, null, or empty strings; still reject truly empty documents. Tests cover root-object and array-record forms. `null` is preserved, not automatically interpreted as a factual "no."

### B03 — Fixed: preserve JSON numeric precision

Location: `app/rag/ingestion.py`, `_finite_decimal` and JSON normalization.

Before the fix, `{"availability_percent":99.999999999999999}` normalized to `{"availability_percent":100.0}` because the parser converted the number to a binary float. That changed source evidence before retrieval and citation validation.

Implemented: parse decimal literals as `Decimal` and serialize them with pinned `simplejson`, retaining JSON numeric types and exact decimal values. This small dependency avoids custom JSON encoder internals. The previous rejection of non-finite and float-range-overflowing inputs remains; unsupported Decimal exponents also produce 422. Tests cover high precision, scientific notation, and values that would underflow binary floats.

### B04 — Fixed: validate vector norms

Before the fix, `checked_vectors` accepted finite elements whose float32 norm overflowed, such as `[1e38, 1e38]`; FAISS normalization then produced a zero vector. Non-finite and zero norms are now rejected before indexing or searching, including underflow. Healthy OpenAI embeddings are not expected to have these scales.

### B05 — Fixed: normalize PDF extraction whitespace

Location: `app/rag/ingestion.py`, `_normalize_pdf_text` and `_parse_pdf`.

Before the fix, `pypdf` treated individually positioned words in the sample SOC 2 tables as separate lines. Those extraction artifacts entered chunking, embeddings, model context, and server-resolved citations. The frontend's `white-space: pre-wrap` then displayed nearly every word on its own line, but the raw `/qa` JSON confirmed that the source of the defect was ingestion rather than rendering.

Implemented: collapse PDF extractor whitespace to one canonical space before counting extracted characters and creating page documents. Page metadata and source words remain intact; visual PDF layout is not reconstructed. Tests cover normalized ingestion and the final endpoint citation.

## Challenge assessment

| Rubric area | Evidence in this implementation | Remaining work / caveat |
| --- | --- | --- |
| Backend correctness — 15 | `/qa`, both formats, multiple questions, stable ordering and JSON schema; review defects fixed with regressions | Nested JSON context retention remains an improvement |
| Robustness — 15 | Byte/page/question/chunk limits, receive-time body limit, request/provider deadlines, sanitized errors | Parser threads cannot be forcibly stopped; hostile PDFs need stronger isolation |
| Code structure — 15 | Route, orchestration, ingestion, retrieval, generation, and resource controls are separated | Review whether each abstraction earns its complexity; no agent framework is needed |
| Tests — 15 | Offline tests; endpoint tests use real parsing/splitting/FAISS/citation checks and fake providers, with review regressions | Mocks do not establish answer quality |
| Performance — 15 | One request-owned index, batched embeddings and native FAISS search, duplicate reuse, bounded calls; reproducible retrieval benchmark | No cross-request cache; retrieval microbenchmark excludes provider latency |
| Container + observability — 10 | Non-root Docker image, health check, secret exclusions, JSON stage/request logs | Prior implementation smoke test passed; no public deployment or load benchmark |
| Grounding — 10 | Fixed `gpt-4o-mini`, retrieved text, strict schema, server-resolved chunk-ID citations, abstention | Selected source identity is not semantic entailment; run live-model evaluation after configuring the key |
| Minimal frontend — 5 | `/` uploads both files, displays answers/citations/errors and request IDs; `/docs` remains available | Browser flow tested with fake providers; live-model smoke remains pending |

FAISS is the selected vector store. It lives in this Python process alongside the chunk text/metadata; the requirement does not imply a separate database server. The implementation uses LangChain splitting, embedding/generation integrations, and its FAISS wrapper. `text-embedding-3-small` supplies vectors; only `gpt-4o-mini` writes answers, under the model-restriction interpretation documented in the README.

## Reading order: follow one request

1. **`examples/questions.json`, `examples/document.json`, then `app/schemas/qa.py`.** Understand the input, the public output, and the internal evidence schema. The internal model includes chunk IDs; clients receive page numbers/excerpts.
2. **`app/main.py`, then the JSON success test in `tests/test_api.py`.** FastAPI is the HTTP layer: parse two uploaded files, call the service, return a response. `create_app` lets tests inject fake dependencies.
3. **`app/services/qa.py`.** Read `answer` as the overall recipe. Note where the document is processed once, questions are deduplicated, work is concurrent, and original ordering is restored.
4. **`app/rag/ingestion.py` with `tests/test_ingestion.py`.** Bytes become page/record text, then chunks with IDs and source locations. Review the three JSON fixes alongside their regression tests, then examine the nested-pages improvement in NEXT_STEPS.md.
5. **`app/rag/retrieval.py` with `tests/test_retrieval.py`.** Text becomes vectors; FAISS ranks chunks; retrieved chunks still contain their text. Read `IndexBuilder`, then `DocumentIndex.search`. The answer model never receives the vector coordinates.
6. **`app/rag/generation.py` and `app/rag/prompts.py` with `tests/test_generation.py`.** Review the generic input/output examples and the request-specific schema: the model must return exactly the planned part keys. Follow normalization, unsupported-answer handling, and chunk-ID citation resolution. Tests cover missing/extra parts, concurrent plans, and example contract validity; mocks do not prove that a live model chooses the right status. Ask whether each selected source chunk supports the entire answer, not just whether the ID exists.
7. **`app/core/runtime.py`, `app/core/middleware.py`, `app/core/config.py`, and `app/core/observability.py`.** These are operational boundaries: simultaneous calls/requests, deadlines, safe logs, and input limits. Their tests explain why cancellation and resource ownership are more careful than a small prototype.

Do not start by reading every dependency import or all concurrency plumbing. Trace the ordinary JSON request first, then the failure paths.

## Suggested review sessions

1. **Run and observe (10–15 minutes):** `make doctor`, `make test`, `make dev`; inspect `/` and `/docs`. With a key configured locally, upload the synthetic files. Check the expected AWS answer, absent penetration-test answer, citation excerpts, and JSON log events. Live requests consume credits.
2. **Review a small change (20–30 minutes):** choose one fixed bug, read its regression test first, and explain why the old code failed. Then read the smallest corresponding change. For a next implementation, use the concrete JSON-context criteria in NEXT_STEPS.md.
3. **Challenge the boundaries (15–20 minutes):** read endpoint tests for duplicate questions, malformed uploads, capacity rejection, and timeouts. Confirm that provider failure differs from "Not found in document."
4. **Assess quality before polish:** use a real permitted SOC 2 sample and a small answer key containing supported, unsupported, cross-page, and numeric questions. Record retrieval misses separately from unsupported generated claims. Fix measured gaps before adding caching, Deep Agents, streaming, or specialized chunking.
5. **Submission finish:** review outstanding ledger entries, run the live-model evaluation and tests/container smoke, and review Git's staged files for secrets. Do not commit `.env`.

# Zania document question answering

A Python 3.12 backend that answers a JSON list of questions from an uploaded PDF or JSON document. It uses LangChain for splitting, embeddings, FAISS retrieval, and structured answer generation. Fully and partially supported answers include source-owned excerpts; unsupported questions return exactly `Not found in document`. Each answer exposes its model-assessed evidence coverage as `found`, `partial`, or `not_found`.

The backend includes a minimal HTML/JavaScript upload client at `/`: select the two files, submit them, and review answers and source excerpts. It runs from the same FastAPI server with no Node build or extra frontend service. Streaming, agent planning, and persistent document storage are deferred. FastAPI's `/docs` remains available for inspecting the API contract.

## Run locally

Open **this repository directory** (`zania-assignment/grc-rag-agent`), not just its parent `interview-prep`. On macOS/Linux with Python 3.12 and Make installed:

```bash
make install
make env
make doctor
```

`make install` is this repo's equivalent of `npm install`: it creates `.venv` if needed and installs the pinned dependencies there. Every Make command uses `.venv/bin/python` explicitly, so **you do not need to activate the environment**. `make env` preserves an existing `.env`. `make doctor` checks imports and reports whether a key is configured without showing it or making network calls.

Open `.env` in your editor and set `OPENAI_API_KEY` to the provided key. Do not put the key in source, a request, a screenshot, or a shell command recorded in history. `.env` and local variants are excluded from Git and Docker build context; `.env.example` contains placeholders only. Environment variables take precedence over `.env`. Restart the server after changing `.env`.

```bash
make dev
```

Visit [the upload client](http://127.0.0.1:8000/). Select `examples/questions.json` and `examples/document.json`, then choose **Answer questions**. The page shows elapsed time, answers, citations, and the request ID; failed requests display the server's error and allow a retry. Generated content is rendered as text, never inserted as HTML. You can also use [the interactive API docs](http://127.0.0.1:8000/docs): expand `POST /qa`, select **Try it out**, upload the files, and select **Execute**. The UI, `/healthz`, and `/docs` work without an API key; a valid QA request without a configured key receives a clear `503`. With a valid key, executing `/qa` makes billable OpenAI calls. The terminal shows JSON events for ingestion, retrieval, generation, and request completion.

Stop with Ctrl-C. If port 8000 is occupied, use `make dev PORT=8001` and visit port 8001. In another terminal, `make sample` sends the example files (also accepts `PORT=8001`). `make test` exercises the full pipeline offline with deterministic model replacements.

### Docker Compose setup

The repository includes a non-root Python 3.12 image and a Compose service for the shortest repeatable setup:

```bash
make env
# Set OPENAI_API_KEY in .env, then:
docker compose up --build
```

Visit [the upload client](http://127.0.0.1:8000/) or [API docs](http://127.0.0.1:8000/docs). Compose passes `.env` at runtime but does not copy it into the image, binds the API to loopback, and uses the image's `/healthz` health check. Set `PORT=8001 docker compose up --build` to change the host port. Stop with Ctrl-C followed by `docker compose down`. The container can start without `.env` for health/docs checks, but `POST /qa` returns `503` until a key is configured.

### Python setup, coming from JavaScript

| JavaScript concept | This repository |
| --- | --- |
| `package.json` dependencies | `pyproject.toml` |
| Lockfile | `requirements.txt` and `requirements-dev.txt` |
| Project-local installed packages (`node_modules`) | `.venv`, which also contains its own Python executable |
| `npm install` | `make install` |
| `npm run dev` / `npm test` | `make dev` / `make test` |

Python imports belong to the interpreter executing the program. Installing packages into `.venv` does not install them into the global `python3`. If you prefer plain commands, the equivalent setup/run sequence is:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m uvicorn app.main:app --reload --no-access-log
```

### Missing imports in your editor

For VS Code (or an editor using its Python extension), open `grc-rag-agent.code-workspace` or this repository folder. The included `.vscode/settings.json` points to the project environment. If imports remain underlined, use **Python: Select Interpreter** and select **`.venv/bin/python` inside this repository**; a previously selected interpreter may override the default. Ensure the Python extension is installed. If you keep the parent folder open, select the nested repository's interpreter explicitly.

`make doctor` is the runtime check: if it succeeds while the editor still reports missing imports, check the editor's interpreter selection and reload its window to refresh the language server. See [VS Code's interpreter selection guidance](https://code.visualstudio.com/docs/python/environments#_select-an-environment) and [Python's virtual environment documentation](https://docs.python.org/3.12/library/venv.html).

For a guided code walkthrough, rubric assessment, and the bug ledger, read [REVIEW.md](REVIEW.md). Remaining improvements are in [NEXT_STEPS.md](docs/NEXT_STEPS.md).

### Which OpenAI models are used?

Both models use the same `OPENAI_API_KEY`, subject to that key's project/model permissions:

| Job | Model | Output |
| --- | --- | --- |
| Convert source chunks and questions for similarity search | `text-embedding-3-small` (configurable with `EMBEDDING_MODEL`) | Numerical vectors |
| Read retrieved text and write the answer | `gpt-4o-mini` (fixed) | Structured part answers and selected chunk IDs |

This interprets the challenge's “gpt-4o-mini only” instruction as applying to answer generation. An embedding model is a separate component needed for semantic retrieval, not a second answer writer. The provided key must permit both calls. See the [OpenAI embeddings guide](https://developers.openai.com/api/docs/guides/embeddings) and [Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs).

## API

`POST /qa` accepts exactly two `multipart/form-data` file fields:

- `questions`: `.json` containing a nonempty array of nonblank strings.
- `document`: `.pdf` containing extractable text, or `.json` containing an object or array.

```bash
curl --fail-with-body http://127.0.0.1:8000/qa \
  -F 'questions=@examples/questions.json;type=application/json' \
  -F 'document=@examples/document.json;type=application/json'
```

For PDF, replace the document argument with `-F 'document=@report.pdf;type=application/pdf'`. These are real model calls once a key is configured. The example report is synthetic; its penetration-testing question intentionally has no supporting evidence.

Successful responses use the required `results` shape:

```json
{
  "results": [
    {
      "question": "Which cloud providers do you rely on?",
      "answer": "The production service is hosted on AWS.",
      "status": "found",
      "citations": [{"page": null, "excerpt": "The production service is hosted on AWS."}]
    },
    {
      "question": "How frequently do you conduct penetration tests?",
      "answer": "Not found in document",
      "status": "not_found",
      "citations": []
    },
    {
      "question": "Which providers host production and backups?",
      "answer": "Production uses AWS; the backup provider is not specified.",
      "status": "partial",
      "citations": [{"page": null, "excerpt": "The production service is hosted on AWS."}]
    }
  ]
}
```

This is an illustrative subset; a real response includes every submitted question in its original order. Identical question strings reuse the same result. The model selects server-issued chunk IDs and never generates citation text or page numbers. The server resolves each selected ID to the complete source chunk, so an excerpt can be up to the configured chunk size (1,000 characters by default). PDF citations use the physical page position starting at 1, not the printed page label. JSON has no pages, so `page` is `null`; its excerpts refer to deterministic JSON normalization (sorted keys, two-space indentation, Unicode preserved). Decimal values retain their precision through `Decimal` and `simplejson`; numbers remain JSON numbers, not strings. Non-finite values and numbers outside supported numeric ranges are rejected. Named fields with empty lists, nulls, or empty strings are retained as source evidence; a null does not automatically mean "no."

The required per-answer `status` is aggregated from the model's part assessments:

| Answer status | Meaning |
| --- | --- |
| `found` | Every question part is fully supported. |
| `partial` | At least some requested information is supported, but one or more parts are incomplete or unsupported. Citations for the supported content are retained. |
| `not_found` | No part is supported; the exact fallback and empty citations are returned. |

The upload UI displays this status and its meaning beside each answer. These are model-assessed coverage labels, not confidence scores or an independent guarantee of correctness. Source-ID validation ensures excerpts come from retrieved chunks; users still need to inspect whether those excerpts support the claims. `not_found` describes missing evidence, not a negative answer. The prompt applies the same rule to every document: partial answers must establish some requested information, not merely offer related background.

Every response has an `X-Request-ID` header. Errors are sanitized JSON:

```json
{"error":{"code":"missing_api_key","message":"Set OPENAI_API_KEY in the environment or local .env file.","request_id":"..."}}
```

| Status | Meaning |
| --- | --- |
| `413` | Byte, question, page, extracted-text, nesting, or chunk limit exceeded |
| `415` | Unsupported request/file format or conflicting extension and media type |
| `422` | Malformed uploads, invalid question JSON, corrupt/encrypted/textless PDF, empty source |
| `502` | Invalid provider response, unknown evidence chunk ID, or rejected upstream request |
| `503` | Missing/invalid key, unavailable model/provider, rate limiting, or local capacity full |
| `504` | Provider timeout or whole-request deadline exceeded |
| `500` | Unexpected internal failure, without exposing exception details |

Request execution is all-or-nothing: provider errors are not represented as unsupported answers or partial request success. A successful question may still contain a grounded partial answer, even when it has only one planned part. Correctly reporting missing evidence is different from a failed model call.

## Architecture and decisions

```text
POST /qa
  → bounded upload and validation
  → page/record extraction → chunks with source metadata
  → batched embeddings → request-owned in-memory FAISS index
  → one deterministic question plan → one batched FAISS search
  → original top-k plus bounded, deduplicated subquery evidence
  → concurrent gpt-4o-mini structured part answers
  → server-side chunk-ID resolution → results in original order
```

For code review, start at `app/main.py` (the HTTP contract), then `app/services/qa.py` (the complete workflow). The service calls `ingestion`, `retrieval`, and `generation`; `runtime` provides shared resource controls and `middleware` bounds incoming requests. Tests demonstrate the intended behavior at each boundary.

```text
app/
  main.py       # HTTP entry point; the only file at this level
  services/     # QA orchestration
  rag/          # Ingestion, question planning, retrieval, generation, prompts
  core/         # Configuration, errors, middleware, runtime, observability
  schemas/      # Public responses and internal generated-answer types
  static/       # Upload UI
```

`app` is a namespace package; its Python subpackages have `__init__.py` files.

- **Question parts are a shared contract:** retrieval always precedes generation. No model decides how to decompose or search, and no planner, grader, or subagent adds model calls. One deterministic plan supplies the exact question parts to both retrieval and structured generation, so evidence is searched for every explicit `?` clause or numbered choice instead of relying on one broad embedding to represent a multipart question. Dependent directives such as “If yes, describe” remain attached to the original question. Duplicate original questions and derived queries are computed once, while response order and one result per submitted question are preserved. One normal generation call is made per unique original question; only transient provider failures can cause one retry.
- **Native multi-question FAISS search:** a normalized vector index and LangChain document store are built once per request and reused for all questions. Embeddings may use several bounded provider batches, but all resulting unique question-part vectors are passed as one matrix to one `store.index.search(matrix, k)` call. FAISS returns one result row per query in the same order; the following Python loop only maps integer positions to documents and does not perform more similarity searches. Each final context keeps the original question's top `k`, then adds unique part-query results round-robin up to `2k`. Missing neighbors (`-1`) are skipped. This batching reduces repeated Python/native boundary overhead without changing per-question ranking semantics. Similarity uses L2 distance on normalized vectors, which has the same ranking as cosine similarity; changing FAISS index type would not improve semantic recall for this exact search.
- **Extraction and chunking:** PDF extractor whitespace is normalized to one space before limits, chunking, embeddings, generation, and citation display. Recursive character splitting starts at 1,000 characters with 400-character target overlap; the larger measured overlap keeps nearby section headings with continuation evidence. Chunks cannot cross PDF pages or JSON records and retain character offsets into the normalized source representation. A root JSON object is one record, while top-level array items are separate records. Nested arrays such as `pages` are not yet recognized as record boundaries. See [NEXT_STEPS.md](docs/NEXT_STEPS.md) for the remaining JSON-context work.
- **Grounding and partial answers:** each generation call gets a strict response schema whose `parts` object requires exactly the IDs in its question plan, with no extra keys. Each value contains `status` (`supported`, `partial`, or `not_found`), `answer`, and `evidence_chunk_ids`. The server normalizes these keyed results into plan order, validates the part set and IDs, resolves citations from source-owned chunks, and preserves grounded partial answers regardless of part count. Generic input/output examples in `app/rag/prompts.py` illustrate full support, partial support, missing evidence, and unsupported inferences. A partial answer must address the requested property and identify what remains unspecified; merely related facts do not answer it. When every part is `not_found`, the server returns the exact fallback with no citations. Numbered options retain display-only labels for concise missing-part messages; retrieval and generation still use the same full question text. The prompt and answer-field description restrict internal IDs to the structured evidence field. The public API and number of model calls are unchanged. Schema validation establishes response shape and source identity, not semantic correctness or guaranteed prose compliance.
- **Concurrency:** async provider I/O shares a process-wide semaphore; document and question embeddings are batched. Synchronous extraction, splitting, and FAISS work run in a bounded thread pool. On one generation failure, sibling generation tasks are cancelled. Results retain input order.
- **Privacy:** the service does not persist uploads or indexes. The multipart parser may temporarily spool large uploads to disk; handles close after success/failure. The total request body is buffered in memory only after enforcing its byte limit during receipt. LangSmith tracing is disabled for the pipeline, and logs contain operational metadata rather than report contents, questions, answers, filenames, or credentials. Document text is still sent to OpenAI for embeddings and retrieved passages for generation.

### Configurable limits

Settings live in `app/core/config.py`; all are environment-configurable except the fixed answer model. `.env.example` lists them. Limits must be positive, and chunk overlap must be smaller than chunk size.

| Setting | Default |
| --- | --- |
| `MAX_DOCUMENT_BYTES` | 10 MiB |
| `MAX_QUESTIONS_BYTES` | 128 KiB |
| `MAX_REQUEST_BYTES` | 11 MiB including multipart framing |
| `MAX_QUESTIONS` / `MAX_QUESTION_CHARS` | 50 / 2,000 |
| `MAX_PDF_PAGES` / `MAX_EXTRACTED_CHARS` | 200 / 1,000,000 |
| `MAX_CHUNKS` / `MAX_JSON_DEPTH` | 2,000 / 64 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1,000 / 400 characters |
| `RETRIEVAL_K` / `EMBEDDING_BATCH_SIZE` | 6 / 64 |
| `MAX_PROVIDER_CALLS` / `MAX_ACTIVE_REQUESTS` | 4 / 2 per process |
| `WORKER_THREADS` | 2; each FAISS operation uses one OpenMP thread |
| `REQUEST_TIMEOUT_SECONDS` / `PROVIDER_TIMEOUT_SECONDS` | 120 / 30 seconds |
| `MAX_ANSWER_TOKENS` | 1,200 |

The request deadline includes upload receipt, multipart parsing, queueing, retries, and generation. A transient connection, rate-limit, or upstream server failure gets at most one retry with a short backoff; timeouts and invalid generated outputs are not retried. Native SDK retries are disabled so they cannot multiply retries.

Application events are JSON logs with a generated request ID, stage timings, counts, status, and generation token usage when supplied. Uvicorn startup/shutdown messages remain ordinary server logs; access logs are disabled by the documented commands. External parser/provider logging is suppressed to avoid accidentally logging source material.

### Limits of this first pass

- PDF extraction has no OCR or specialized table/layout reconstruction. Whitespace normalization makes positioned text readable but does not reconstruct table columns, and page-local splitting may lose relationships that span pages. JSON records split into several chunks may also lose distant context.
- No authentication or durable tenancy model is implemented. Run locally for the challenge; a public deployment needs appropriate access controls and service-wide resource management. Limits are per worker process, so multiple Uvicorn workers multiply them.
- A Python thread cannot be forcibly stopped. Cancelling an HTTP request leaves already-running parsing/index work alive until it completes; its worker slot stays occupied, and no native index is reset while a worker might read it. Byte/page/text limits help bound ordinary workloads but do not provide hard CPU/memory isolation for hostile PDFs. Process isolation is a future hardening step.
- Mocked tests establish pipeline behavior, not live-model answer quality or perfect resistance to prompt injection. Evaluate real sample questions for retrieval recall, faithful answers, correct abstention, citation accuracy, latency, and cost before claiming production quality.
- Re-uploading a document rebuilds its index. A future in-process cache should use a key of `(document SHA-256, embedding model, ingestion/chunking version and settings)`, bounded memory/LRU eviction, TTL, duplicate-build prevention, and access/tenant isolation. Index construction is separated in `IndexBuilder` so this can be added without changing `/qa`.

## Tests and containers

```bash
make test
make check
```

Tests use synthetic PDF/JSON inputs, deterministic fake embeddings, and mocked generation. Endpoint tests still run real ingestion, chunking, FAISS search, question decomposition, context merging, part validation, and citation resolution. The test fixture blocks external socket connections, and no live API key is required. Coverage includes malformed input, resource limits, partial answers, abstention, invalid citations, provider errors, timeouts, ordering, duplicate questions, concurrency, and request isolation.

GitHub Actions adds a real-HTTP end-to-end boundary: it starts Uvicorn, calls the actual `/healthz`, `/openapi.json`, and multipart `/qa` endpoints with `curl`, and runs the production ingestion, planning, indexing, batched FAISS retrieval, validation, middleware, and serialization path. Only embeddings and answer generation are dependency-injected deterministic providers, so CI needs no credentials and cannot make billable model calls. The check also requires JSON stage events with latency fields and verifies that the sample's three question embeddings are sent as one provider batch. A separate job validates Compose, builds the production image, and smoke-tests its health and OpenAPI endpoints.

To reproduce the offline retrieval comparison:

```bash
.venv/bin/python -m scripts.benchmark_retrieval
```

On the local review machine, 2,000 synthetic document vectors, 50 question vectors, 1,536 dimensions, one native thread, and 15 repetitions measured median retrieval time of **19.565 ms** for 50 repeated searches and **16.320 ms** for one matrix search (**1.20×**). The script first verifies identical per-question documents and ordering. This isolates FAISS retrieval and Python/native call overhead; ingestion, embedding, question planning, context merging, and generation are excluded, so it is not an end-to-end latency claim. Results vary by machine and workload. FAISS documents [multi-vector search](https://github.com/facebookresearch/faiss/wiki) as a supported optimization.

For a direct Docker run without Compose:

```bash
docker build -t zania-grc-rag:local .
docker run --rm --env-file .env -p 127.0.0.1:8000:8000 zania-grc-rag:local
```

The container uses Python 3.12, runs as a non-root user, and includes a `/healthz` health check. The image contains neither `.env` nor uploaded documents. For a credential-free container smoke test, omit `--env-file .env`; health and API docs remain available.

`requirements.txt` and `requirements-dev.txt` pin runtime and test dependencies. Use the lockfiles for normal installs; `pyproject.toml` declares compatible development ranges. To deliberately refresh the locks:

```bash
pip-compile --strip-extras --no-emit-index-url -o requirements.txt pyproject.toml
pip-compile --strip-extras --no-emit-index-url -c requirements.txt --extra dev -o requirements-dev.txt pyproject.toml
```

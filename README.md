# Zania document question answering

A Python 3.12 backend that answers a JSON list of questions from an uploaded PDF or JSON document. It uses LangChain for splitting, embeddings, FAISS retrieval, and structured answer generation. Every supported answer includes a verified source excerpt; unsupported questions return exactly `Not found in document`.

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

For a guided code walkthrough, rubric assessment, and the bug ledger, read [REVIEW.md](REVIEW.md). The proposed next improvements and folder structure are in [NEXT_STEPS.md](docs/NEXT_STEPS.md).

### Which OpenAI models are used?

Both models use the same `OPENAI_API_KEY`, subject to that key's project/model permissions:

| Job | Model | Output |
| --- | --- | --- |
| Convert source chunks and questions for similarity search | `text-embedding-3-small` (configurable with `EMBEDDING_MODEL`) | Numerical vectors |
| Read retrieved text and write the answer | `gpt-4o-mini` (fixed) | Structured answer and evidence quotes |

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
      "citations": [{"page": null, "excerpt": "The production service is hosted on AWS."}]
    },
    {
      "question": "How frequently do you conduct penetration tests?",
      "answer": "Not found in document",
      "citations": []
    }
  ]
}
```

This is an illustrative subset; a real response includes every submitted question in its original order. Identical question strings reuse the same result. PDF citations use the physical page position starting at 1, not the printed page label. JSON has no pages, so `page` is `null`; its excerpts refer to deterministic JSON normalization (sorted keys, two-space indentation, Unicode preserved). Decimal values retain their precision through `Decimal` and `simplejson`; numbers remain JSON numbers, not strings. Non-finite values and numbers outside supported numeric ranges are rejected. Named fields with empty lists, nulls, or empty strings are retained as source evidence; a null does not automatically mean "no."

Every response has an `X-Request-ID` header. Errors are sanitized JSON:

```json
{"error":{"code":"missing_api_key","message":"Set OPENAI_API_KEY in the environment or local .env file.","request_id":"..."}}
```

| Status | Meaning |
| --- | --- |
| `413` | Byte, question, page, extracted-text, nesting, or chunk limit exceeded |
| `415` | Unsupported request/file format or conflicting extension and media type |
| `422` | Malformed uploads, invalid question JSON, corrupt/encrypted/textless PDF, empty source |
| `502` | Invalid provider response, fabricated citation, or rejected upstream request |
| `503` | Missing/invalid key, unavailable model/provider, rate limiting, or local capacity full |
| `504` | Provider timeout or whole-request deadline exceeded |
| `500` | Unexpected internal failure, without exposing exception details |

Results are all-or-nothing: provider errors are not represented as unsupported answers or partial success. Correctly reporting missing evidence is different from a failed model call.

## Architecture and decisions

```text
POST /qa
  → bounded upload and validation
  → page/record extraction → chunks with source metadata
  → batched embeddings → request-owned in-memory FAISS index
  → batched embeddings of unique questions → top-k source chunks
  → concurrent gpt-4o-mini structured answers
  → exact citation checks → results in original order
```

For code review, start at `app/main.py` (the HTTP contract), then `app/service.py` (the complete workflow). The service calls `ingestion`, `retrieval`, and `generation`; `runtime` provides shared resource controls and `middleware` bounds incoming requests. Tests demonstrate the intended behavior at each boundary.

- **Two-step RAG:** retrieval always precedes generation. No model decides whether to search, and no planner, grader, or subagent adds model calls. One normal generation call is made per unique question; only transient provider failures can cause one retry.
- **In-memory FAISS:** a normalized vector index and LangChain document store are built once per request and reused for all questions. All query vectors are normalized and searched in one native FAISS batch; its result rows are mapped back through the LangChain document store in question order. Missing neighbors (`-1` when fewer than `k` chunks exist) are skipped. Similarity uses L2 distance on normalized vectors, which has the same ranking as cosine similarity. Source text and metadata remain alongside vectors; the LLM receives text, not vector values. No arbitrary similarity cutoff is treated as proof of absence.
- **Chunking:** recursive character splitting starts at 1,000 characters with 200-character target overlap. Chunks cannot cross PDF pages or JSON records and retain character offsets. A root JSON object is one record, which can produce many chunks; top-level array items are separate records. Nested arrays such as `pages` are not yet recognized as record boundaries, so a chunk may lose its enclosing title or organization. See the concrete reproduction and acceptance criteria in [NEXT_STEPS.md](docs/NEXT_STEPS.md). This is a baseline for evaluation against real SOC 2 reports, not an optimal setting established by benchmarking.
- **Grounding:** strict structured output carries a supported flag, answer, chunk IDs, and verbatim evidence. Only retrieved IDs and exact nonblank source quotes are accepted. Page numbers come from server metadata. Invalid citations fail with `502`; insufficient evidence produces the prescribed fallback. These checks establish source identity and quote accuracy, not semantic proof that every generated claim is entailed.
- **Concurrency:** async provider I/O shares a process-wide semaphore; document and question embeddings are batched. Synchronous extraction, splitting, and FAISS work run in a bounded thread pool. On one generation failure, sibling generation tasks are cancelled. Results retain input order.
- **Privacy:** the service does not persist uploads or indexes. The multipart parser may temporarily spool large uploads to disk; handles close after success/failure. The total request body is buffered in memory only after enforcing its byte limit during receipt. LangSmith tracing is disabled for the pipeline, and logs contain operational metadata rather than report contents, questions, answers, filenames, or credentials. Document text is still sent to OpenAI for embeddings and retrieved passages for generation.

### Configurable limits

Settings live in `app/config.py`; all are environment-configurable except the fixed answer model. `.env.example` lists them. Limits must be positive, and chunk overlap must be smaller than chunk size.

| Setting | Default |
| --- | --- |
| `MAX_DOCUMENT_BYTES` | 10 MiB |
| `MAX_QUESTIONS_BYTES` | 128 KiB |
| `MAX_REQUEST_BYTES` | 11 MiB including multipart framing |
| `MAX_QUESTIONS` / `MAX_QUESTION_CHARS` | 50 / 2,000 |
| `MAX_PDF_PAGES` / `MAX_EXTRACTED_CHARS` | 200 / 1,000,000 |
| `MAX_CHUNKS` / `MAX_JSON_DEPTH` | 2,000 / 64 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1,000 / 200 characters |
| `RETRIEVAL_K` / `EMBEDDING_BATCH_SIZE` | 6 / 64 |
| `MAX_PROVIDER_CALLS` / `MAX_ACTIVE_REQUESTS` | 4 / 2 per process |
| `WORKER_THREADS` | 2; each FAISS operation uses one OpenMP thread |
| `REQUEST_TIMEOUT_SECONDS` / `PROVIDER_TIMEOUT_SECONDS` | 120 / 30 seconds |
| `MAX_ANSWER_TOKENS` | 1,200 |

The request deadline includes upload receipt, multipart parsing, queueing, retries, and generation. A transient connection, rate-limit, or upstream server failure gets at most one retry with a short backoff; timeouts and invalid generated outputs are not retried. Native SDK retries are disabled so they cannot multiply retries.

Application events are JSON logs with a generated request ID, stage timings, counts, status, and generation token usage when supplied. Uvicorn startup/shutdown messages remain ordinary server logs; access logs are disabled by the documented commands. External parser/provider logging is suppressed to avoid accidentally logging source material.

### Limits of this first pass

- PDF extraction has no OCR or specialized table/layout reconstruction. Page-local splitting may lose relationships that span pages. JSON records split into several chunks may also lose distant context.
- No authentication or durable tenancy model is implemented. Run locally for the challenge; a public deployment needs appropriate access controls and service-wide resource management. Limits are per worker process, so multiple Uvicorn workers multiply them.
- A Python thread cannot be forcibly stopped. Cancelling an HTTP request leaves already-running parsing/index work alive until it completes; its worker slot stays occupied, and no native index is reset while a worker might read it. Byte/page/text limits help bound ordinary workloads but do not provide hard CPU/memory isolation for hostile PDFs. Process isolation is a future hardening step.
- Mocked tests establish pipeline behavior, not live-model answer quality or perfect resistance to prompt injection. Evaluate real sample questions for retrieval recall, faithful answers, correct abstention, citation accuracy, latency, and cost before claiming production quality.
- Re-uploading a document rebuilds its index. A future in-process cache should use a key of `(document SHA-256, embedding model, ingestion/chunking version and settings)`, bounded memory/LRU eviction, TTL, duplicate-build prevention, and access/tenant isolation. Index construction is separated in `IndexBuilder` so this can be added without changing `/qa`.

## Tests and containers

```bash
make test
make check
```

Tests use synthetic PDF/JSON inputs, deterministic fake embeddings, and mocked generation. Endpoint tests still run real ingestion, chunking, FAISS search, and citation validation. The test fixture blocks external socket connections, and no live API key is required. Coverage includes malformed input, resource limits, abstention, invalid citations, provider errors, timeouts, ordering, duplicate questions, concurrency, and request isolation.

To reproduce the offline retrieval comparison:

```bash
.venv/bin/python -m scripts.benchmark_retrieval
```

On the local review machine, 2,000 synthetic document vectors, 50 queries, 1,536 dimensions, one native thread, and 15 repetitions measured median retrieval time of **27.796 ms** for the prior loop and **21.960 ms** for the batch (**1.27×**). The script verifies matching retrieved documents before timing. This is a local retrieval-only microbenchmark, not an end-to-end latency claim; provider calls are excluded. Results vary by machine and workload. FAISS documents [multi-vector search](https://github.com/facebookresearch/faiss/wiki) as a supported optimization.

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

# Improvements to review one at a time

The follow-up work implements the validation/evidence fixes, native FAISS batch search, a minimal upload UI, and the folder layout below. JSON-context changes remain a design decision, not an implemented feature.

## 1. Preserve JSON context and expose precise JSON citations

One parsed `Document` is not necessarily one vector: the current splitter already breaks a large root object into many chunks. The remaining problem is losing structural context when a title or parent field is far from a retrieved passage.

Reproduced with a root object containing `organization: "Acme"` and a `pages` array. Its first item has `title: "Production hosting"` and a long `text` field (`"Background policy. "` repeated 100 times, followed by `"The production service is hosted on AWS."`). A second item contains encryption text. Current defaults produce **one record and five chunks**. The chunk containing AWS has neither `Acme` nor `Production hosting`, and its source path is only `$`.

The implementation is deliberately undecided. [DESIRED_FEATURES.md](DESIRED_FEATURES.md) records the
online research, the limits reproduced in LangChain's JSON `create_documents` path, citation-range
alternatives, and the comparison spike required before choosing an approach.

Desired result, pending that spike:

1. Recognize a nested array of records as a boundary while leaving small scalar arrays such as `providers: ["AWS", "Azure"]` together. Start with the observed `pages` shape; do not assume every arbitrary array represents pages.
2. Carry selected enclosing fields (organization, section/title) and the exact original JSON path into each derived record. Define a size budget for repeated parent context so a huge parent cannot multiply memory and token usage.
3. Split oversized record text with the existing character splitter, preserving that parent context and source path for every resulting chunk.
4. Extend JSON citations with a server-resolved location. The leading candidate is an exact character range within an identified normalized JSON record, not a model-generated path. Do not treat an uploaded JSON `page` field as a verified PDF page number.
5. Document exactly which normalized representation citations quote. Keep locator/header text distinguishable from source quotations, and retain the original array positions even when empty entries are skipped.

Acceptance tests: root object and root array behavior; nested page objects; two pages with identical text but distinct identities; long page text whose answer appears near the end; empty arrays that are evidence; keys containing dots/slashes/quotes; unsupported questions; and maximum expanded characters/chunks. Compare retrieval on the same labeled questions before and after the change.

## 2. A folder structure with meaningful boundaries

Implemented layout (`main.py` is the only file directly under `app`):

```text
app/
  main.py
  core/
    config.py
    errors.py
    middleware.py
    runtime.py          # WorkerPool and ProviderRunner
    observability.py
  services/
    qa.py               # Request orchestration
  rag/
    ingestion.py
    planning.py         # Shared QuestionPlan for retrieval and generation
    retrieval.py
    generation.py
    prompts.py          # Generation instructions and generic response examples
  schemas/
    qa.py               # Public responses and internal generated-answer types
  static/
    index.html
    app.js
    styles.css
```

The Python subpackages include `__init__.py`; `app` itself is a namespace package.
The move preserved retrieval and runtime behavior and passed the existing offline suite before
the generation changes. `core/runtime.py` retains bounded synchronous work and asynchronous
provider concurrency, timeouts, and retries. No new retrieval algorithm accompanies this refactor.

## 3. What checked_vectors actually protects

FAISS expects a rectangular matrix of 32-bit floats: one row per input text and one column per embedding dimension. `checked_vectors` converts the provider result to that representation and checks:

- The number of returned vectors matches the number of texts sent.
- Every vector has the same nonzero number of dimensions.
- All coordinates are finite numbers.
- Each float32 norm is finite and nonzero, so normalization is meaningful.

It does not create embeddings or judge whether their meaning is correct. `DocumentIndex.search` also checks query dimensions against the index. The vector store is FAISS plus LangChain's accompanying chunk-text/metadata store, all in process.

## 4. Evaluation completed and remaining work

The latest generation change adds generic response examples and a request-specific keyed-parts
schema to prevent extra or missing answer parts. All ten subsequent live requests returned HTTP
200 with stable public statuses. Q3 remained correct and Q4 correctly abstained; Q1 still overclaimed,
Q2 mislabeled background as partial support, and Q5 omitted the documented monitoring signals.
Capture current contexts and test generation against fixed evidence before more retrieval changes.
The older results below do not establish the latest prompt's answer quality.

A first fixed live evaluation now covers five SOC 2 questions across ten runs. It records retrieved source identities, support/abstention behavior, citation pages, provider-call count, and elapsed time. Deterministic decomposition plus 1,000/400 chunking restored the incident-notification evidence; a `k=6/12/20` sweep showed that deeper vector retrieval alone did not. A BM25/vector experiment improved exact policy-term recall but did not retrieve the direct timing evidence, so hybrid retrieval was not shipped. A 700/140 chunking experiment also failed that recall gate.

The retained overlap keeps the incident heading and its timing sentence in one chunk. It raises this fixture from 273 to 324 chunks and adds one document-embedding batch, so future evaluation should confirm that tradeoff on a broader 10–20 question set with reserved examples that did not drive this change. Add supported facts, unsupported questions, empty-list evidence, decimals, repeated sections, and nested-page evidence before claiming general answer quality.

One normal generation call per unique original question remains the baseline. Explicit parts are evaluated inside that structured call; batching several original questions into one call would still expand context, complicate source assignment, and couple retries/failures. Re-upload caching remains separate and needs bounded memory, model/chunking-version keys, eviction, and duplicate-build prevention.

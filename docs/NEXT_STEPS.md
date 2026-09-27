# Improvements to review one at a time

The first follow-up implements the four known validation/evidence bugs, native FAISS batch search, and a minimal upload UI. This document records the next design decisions; the proposed folder moves and JSON-context changes are not implemented yet.

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

Suggested follow-up refactor, after reviewing the behavioral changes:

```text
app/
  __init__.py
  main.py
  config.py
  errors.py
  api/
    __init__.py
    schemas.py          # Public question/answer/citation response shapes
    middleware.py
  rag/
    __init__.py
    service.py
    ingestion.py
    retrieval.py
    generation.py
    schemas.py          # Internal generated answer and evidence shapes
  infrastructure/
    __init__.py
    execution.py        # Current runtime.py: WorkerPool and ProviderRunner
    logging.py
  static/
    index.html
    app.js
    styles.css
```

`runtime.py` currently has two responsibilities: execute synchronous parsing/index work in bounded threads, and control asynchronous provider calls with concurrency limits, timeouts, and retry policy. `execution.py` makes that role clearer. File moves should be one behavior-preserving change with updated imports and the same tests, not combined with a new retrieval algorithm.

## 3. What checked_vectors actually protects

FAISS expects a rectangular matrix of 32-bit floats: one row per input text and one column per embedding dimension. `checked_vectors` converts the provider result to that representation and checks:

- The number of returned vectors matches the number of texts sent.
- Every vector has the same nonzero number of dimensions.
- All coordinates are finite numbers.
- Each float32 norm is finite and nonzero, so normalization is meaningful.

It does not create embeddings or judge whether their meaning is correct. `DocumentIndex.search` also checks query dimensions against the index. The vector store is FAISS plus LangChain's accompanying chunk-text/metadata store, all in process.

## 4. What would make the submission stronger

Build a small evaluation set before adding another agent or combining all questions into one generation call. For 10–20 labeled questions, include supported facts, unsupported questions, empty-list evidence, decimals, repeated sections, and nested-page evidence. Record retrieved source identities, whether the answer is supported, correct abstention, citation accuracy, provider-call count, and elapsed time. Real semantic answer quality requires a separate explicitly run live-model evaluation.

Keep the question set fixed when comparing chunking approaches, and reserve a few examples that did not drive the changes. An explanation such as "this example lost its page title; this change preserved the context; these tests and measurements show the result" demonstrates engineering judgment more clearly than adding complexity without evidence.

One normal generation call per unique question remains the baseline. Batching several answers into one call may reduce request overhead but expands context, complicates source assignment, and couples retries/failures. Measure token usage, latency, and answer quality before adopting it. Re-upload caching also remains separate and needs bounded memory, model/chunking-version keys, eviction, and duplicate-build prevention.

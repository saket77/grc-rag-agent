# Final Evaluation

## Evaluated revision and fixtures

- Implementation commit: `ee4c9abd7f53f17c7af15a5774a7542a99e42516`
- Nave SOC 2 PDF SHA-256: `7d1fa9b6efbaeeb6f30d1af720a806e9061a60cbd6896d0e3620e8a78ede6003`
- Appendix questions SHA-256: `5a845407a37efb753153e9f62e1f08be9ab435fb5c92f234977a61714f6cb940`

The PDF and questions in `examples/` are public evaluation fixtures. Production retrieval and
generation contain no fixture-specific page, chunk, company, or answer override.

## Methodology

The fixed PDF and five questions were submitted to the live local `/qa` endpoint ten times in
sequence. Every run re-uploaded and re-indexed the document. The active worker was verified to use
the evaluated checkout, and source hashes remained unchanged for the full batch. The baseline used
the same endpoint, fixtures, configuration, and ten-run method at commit
`430324f5b57ec02b9e635006349d6aed339c1ba0`.

Offline acceptance included Ruff lint and format checks, 221 deterministic tests, the real-HTTP
deterministic-provider test, wheel build/import, Compose validation, a non-root container smoke
test, `git diff --check`, and a tracked-file secret scan.

## Final results

All ten requests returned HTTP 200. There were no schema or provider-validation failures, citation
identity leaks, or internal chunk/part ID leaks.

| Question | Stable public result | Citations | Ten-run semantic result |
| --- | --- | --- | --- |
| Incident notification criteria and SLA | `partial` | page 21 | Never claims defined client-notification criteria; always preserves “without undue delay” and states that a formal/numeric SLA is unspecified. |
| Third-party handling of personal information | `not_found` | none | Never claims that third parties handle personal information; explains that the requested relationship is not established. |
| Cloud providers | `found` | page 17 | Identifies only GCP as the cloud hosting provider. |
| Primary and backup geography | `not_found` | none | Does not turn GCP, multi-zone deployment, or backup architecture into a geographic location. |
| APM, EUM, and DEM | `partial` | page 21 | Always reports CPU, memory, application-error, and uptime monitoring; qualifies the APM label/scope and leaves EUM and DEM unsupported. |

Every returned excerpt was verified verbatim against its claimed one-based physical PDF page. The
hybrid stage adds no provider call; one generation call per unique question and the existing batched
question-embedding operation are preserved.

## Latency

| Measurement | Baseline | Final |
| --- | ---: | ---: |
| Mean live `/qa` latency, ten runs | 16.321 s | 13.178 s |
| Live range | 12.999–28.183 s | 11.874–14.601 s |

Mean live latency improved by 19.3%, remaining comfortably inside the 25% regression limit.

On the fixed offline 2,000-document/50-question/1,536-dimension retrieval benchmark, median
per-query vector search took 19.350 ms, one matrix FAISS search took 17.104 ms (1.13× faster), and
full hybrid search took 17.323 ms. BM25 plus reciprocal-rank fusion therefore added 0.219 ms over
matrix FAISS, below the 50 ms limit.

## Limitations

- This is a narrow regression fixture, not a broad-domain or production-readiness benchmark.
- Generation remains probabilistic even with temperature zero; the ten-run result measures observed
  stability, not a formal semantic guarantee.
- PDFs require extractable text; OCR and complex table reconstruction are not included.
- JSON citations resolve normalized chunks rather than exact JSON paths or character ranges.
- The service has no authentication, durable tenancy, persistent vector store, or cross-request
  cache; process-local limits multiply with additional workers.

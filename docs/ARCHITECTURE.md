# Architecture (Evolving)

## Status
Initial placeholder. Do not treat this outline as a final architecture until the dataset is inspected and an approved plan is implemented.

## Candidate pipeline stages
1. Source loading.
2. Normalization and validation.
3. Metadata extraction/preservation.
4. Chunking.
5. Embedding.
6. Storage.
7. Query preprocessing.
8. Trusted tenant/role/access-tier resolution.
9. Database-side prefiltering.
10. Dense and (if justified) sparse retrieval.
11. Fusion (e.g. RRF, if supported by experiments).
12. Optional reranking.
13. Result formatting and traceability to source documents.
14. Evaluation and observability.

## Candidate components
- Loader(s)
- Normalizer
- Chunker
- Embedding adapter
- Storage/repository layer
- Retrieval engine
- Ranking/fusion
- CLI
- Evaluation harness
- Tests

## Architecture decisions
Record accepted decisions in `DECISIONS.md`, including rationale, alternatives, trade-offs, and date. Revise this document when implemented architecture materially changes.

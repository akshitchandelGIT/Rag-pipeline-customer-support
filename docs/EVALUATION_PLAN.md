# Retrieval Evaluation Plan

## Principle
Measure retrieval quality and security separately. A fluent answer is not proof of correct retrieval.

## Progressive corpus testing
- Stage 0: A small, manually inspected set of about five documents. Create questions with expected document/chunk IDs and evidence.
- Stage 1: Expand the corpus gradually (e.g. 10, 25, 50, then larger subsets as feasible). Preserve the same evaluation queries where applicable.
- Stage 2: Evaluate the full corpus and adversarial/security cases.
- Stage 3: Repeat tests after preprocessing, chunking, embedding, indexing, filtering, fusion, or reranking changes.

## Quality metrics (select based on dataset and task)
- Recall@k: proportion of relevant items found in top-k.
- Precision@k: proportion of top-k results that are relevant.
- MRR: rank of the first relevant result.
- Hit rate@k.
- Context/evidence coverage and qualitative inspection.
- Latency (record environment, corpus size, query set, and percentile; do not claim targets without measurement).

## Evaluation record
For every experiment, record:
- Date and experiment ID.
- Corpus/sample size and version.
- Preprocessing/chunking settings.
- Embedding model and version.
- Retrieval/filter settings.
- Query set and ground-truth labels.
- Metrics and qualitative errors.
- Test command and reproducibility notes.

## Security tests — required
- End_Customer only receives public results.
- Support_Agent receives public and internal_staff, never admin_only.
- Org_Admin receives all authorized tiers.
- Org A query never receives Org B chunks.
- Cross-tenant and unauthorized-tier leakage target: zero.
- Filters applied in database query before ranking, not in Python post-filtering.
- Prompt-injection-like instructions in retrieved documents must not alter trusted access context.

## Ground truth
Ground truth must be explicit and reviewable: query, expected relevant document/chunk IDs, tenant, allowed tier(s), and notes. Do not make up relevance labels after seeing results without marking them as retrospective.

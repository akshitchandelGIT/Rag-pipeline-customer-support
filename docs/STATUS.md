# Project Status

## Current milestone
Milestone M5 — Real Embedding Inference (COMPLETE: In-process CPU-only `sentence-transformers/all-MiniLM-L6-v2` implemented with 384-d vectors, exact WordPiece token counting, bulk truncation warnings, strict query length validation, and full-corpus integration benchmark verified. Database Storage & pgvector integration is next).

## Completed
- Local repository initialized and remote configured.
- Project guidelines and security invariants established in `AGENTS.md`.
- Milestone M1 Plan (`docs/plans/01-dataset-recon-plan.md`) reviewed and approved.
- Milestone M1 Specification (`docs/specs/SPEC-01-dataset-recon.md`) reviewed and approved.
- Diagnostic reconnaissance scanner (`docs/experiments/m1_recon_scanner.py`) implemented using Python 3 standard library only.
- Comprehensive unit test harness (`tests/test_m1_recon_scanner.py`) implemented with 13/13 passing tests using temporary test fixtures.
- PDF detection limitation, multi-channel PII safety, and lightweight frontmatter handling verified.
- Dataset reconnaissance documentation updated in `docs/DATASET_RECON.md`.
- Milestone M2 Plan (`docs/plans/02-chunking-plan.md`) and Specification (`docs/specs/SPEC-02-chunking.md`) audited and approved.
- Milestone M3 Implementation Plan (`docs/plans/03-chunking-implementation-plan.md`) audited, approved, and reconciled with Shared Team Contract v3.0.
- Milestone M3 Phase 1 delivered:
  - Package structure initialized: `src/rag/__init__.py` and `src/rag/chunking/__init__.py`.
  - Foundational data structures, exception hierarchy, raw UUID v4 validator, deterministic UUID v5 chunk generator, and token estimation implemented in `src/rag/chunking/models.py`.
  - Phase 1 models verified with 10 passing unit tests.
- Milestone M3 Phase 2 delivered:
  - Bounded structural Markdown FSM parser implemented in `src/rag/chunking/parser.py`.
  - Frontmatter extraction and strict metadata validation (`EmptyDocumentError`, `MissingFrontmatterError`, `MalformedFrontmatterError`, `InvalidMetadataError`, `InvalidAccessLevelError`).
  - ATX heading parsing, section hierarchy builder, and canonical `heading_path` construction.
  - Dual code fence suppression (both backticks ``` and tildes ~~~) completely bypassing heading extraction inside code blocks.
  - Markdown table row grouping and list item preservation.
  - Graceful EOF recovery on unclosed code blocks emitting structured log warnings without content leakage.
  - Phase 2 parser verified with 16 unit tests, including 100% parsing and validation across all 10 raw corpus documents under `data/raw/`.
- Milestone M3 Phase 3 delivered:
  - Section-aware chunking engine, breadcrumb injection, and fallback protocol implemented in `src/rag/chunking/engine.py`.
  - Breadcrumb formatting with canonical `heading_path` decoupling (presentation markers like `(Table Part 2)` never mutate canonical metadata, Invariant F-01).
  - H2 section topic isolation and standalone H2 isolation (<60 tokens kept independent without merging into sibling H2s).
  - Small H2 intro forward-merging (<60 tokens merged into first H3 child chunk) and substantial H2 intro independence (>=60 tokens emitted independently).
  - Markdown table handling with atomic preservation, budget flush, row splitting on `\n|...|\n` boundaries, and repeated column headers/dividers on continuation chunks.
  - Two-Zone fallback protocol: Zone 1 (400-800 tokens) emitted intact (`is_oversized=True`, `is_split_item=False`); Zone 2 (>800 tokens) split on sentence boundaries (`is_oversized=True`, `is_split_item=True`) with explicit continuation markers.
  - Zone 2 whitespace retention and exact character-level reassembly (`"".join(sub_chunks) == original_source_text`, Invariant F-03).
  - Deterministic UUID v5 chunk IDs, sequential 0-based chunk indices, SHA-256 content hashes, and 100% metadata inheritance.
  - Non-fatal structured logging with zero raw content/PII leakage.
  - Top-level orchestrator `chunk_document` and package exports in `src/rag/chunking/__init__.py`.
  - Phase 3 engine verified with 13 unit tests, including end-to-end processing across all 10 raw documents under `data/raw/` (yielding 113 total valid chunks).
- Milestone M3 Phase 4 delivered:
  - Authoritative automated test suite implemented in `tests/test_m3_chunking.py` using Python standard library `unittest` with pure in-memory synthetic fixtures.
  - Full test matrix T-01 through T-21 implemented and mapped to SPEC-02 and M3 implementation plan (T-01 through T-21 + logging privacy + corpus smoke test).
  - All 23 unit tests passing cleanly in 0.022s (`python3 -m unittest tests/test_m3_chunking.py -v`).
  - Independent read-only audit of Phase 4 completed with APPROVE verdict.
- Milestone M3 Phase 5 delivered:
  - Empirical Stage 0 Smoke Verification executed across all 10 raw documents in `data/raw/` (5 Org A, 5 Org B).
  - 10/10 documents successfully parsed and chunked with 0 unhandled exceptions.
  - 113 total chunks generated (56 Org A, 57 Org B).
  - Token size statistics: Min 38 tokens, Max 169 tokens, Mean 90.2 tokens, Median 83.0 tokens.
  - All 113 natural chunks fit well within the 400-token ceiling (100% $\le 169$ tokens; 0 oversized chunks in the raw dataset).
  - 100% sequential index continuity ($0 \dots N-1$).
  - 100% metadata fidelity (`org_id`, `document_id`, `access_level`, `title`, `category`, `source`).
  - 100% tenant isolation: Org A chunks strictly carry Org A UUID, Org B chunks strictly carry Org B UUID, 0 cross-tenant contamination.
  - 100% determinism: Re-running chunk generation produces identical chunk IDs, SHA-256 hashes, and token estimates.
  - Logging privacy: 0 unhandled warnings or errors logged during standard corpus ingestion.
- Milestone M4 Phase 1 delivered:
  - Foundational data structures, exception hierarchy, immutable configuration dataclass (`EmbeddingConfig` defaulted to `all-MiniLM-L6-v2`, 384 dimensions), and deeply immutable vector container (`EmbeddedChunk`) in `src/rag/embeddings/models.py`.
- Milestone M4 Phase 2 delivered:
  - Model-independent client abstraction (`BaseEmbeddingClient`) and offline deterministic mock client (`DeterministicMockEmbeddingClient`) in `src/rag/embeddings/client.py`.
- Milestone M4 Phase 3 delivered:
  - Pipeline orchestrator (`EmbeddingPipeline`) in `src/rag/embeddings/pipeline.py` enforcing tenant-segregated batching, 5-step indexed 1:1 positional reassembly, and runtime query vectorization (`embed_query`).
- Milestone M4 Phase 5 delivered:
  - Full-corpus Stage 0 smoke verification executed across all 10 raw documents (113 chunks) using the offline deterministic mock client, verifying 100% count match, 100% positional ordering, 100% tenant batch isolation, 384-dimensional finite unit vectors, and 100% metadata fidelity in 0.080s.
  - Authoritative test harness in `tests/test_m4_embeddings.py` expanded to 34 unit and smoke tests (70 total tests across repository).
  - Note: Mock vectors are non-semantic synthetic projections for structural verification; real model weights have not been downloaded.
- Milestone M4 Phase 6 delivered:
  - Staged Shared Team Contract Amendment RFC (`docs/contracts/RFC-01-embedding-dimension-amendment.md`) proposing Contract update to `embedding vector(384)` with `sentence-transformers/all-MiniLM-L6-v2`.
- Milestone M4 Architectural Reconciliation:
  - Reconciled planning documents and specifications with authoritative Shared Team Contract in `docs/CONTRACTS.md` (Section 5 officially locks `embedding VECTOR(384), -- all-MiniLM-L6-v2`).
- Milestone M4 Phase 7 delivered:
  - Finalized M4 documentation, documentation index (`docs/README.md`), and milestone status closeout.
  - Formally closed Milestone M4.
- Milestone M5 delivered:
  - Milestone M5 Plan (`docs/plans/05-real-embedding-inference-plan.md`) reviewed and approved.
  - Milestone M5 Specification (`docs/specs/SPEC-05-real-embedding-inference.md`) reviewed and approved.
  - Bounded runtime dependencies staged in `requirements.txt` with lightweight CPU-only PyTorch (`torch==2.5.1+cpu`, `sentence-transformers==3.3.1`, `transformers==4.47.1`).
  - Real embedding client (`SentenceTransformerEmbeddingClient`) implemented in `src/rag/embeddings/client.py`:
    - In-process CPU execution under `torch.inference_mode()`.
    - Lazy model loading lifecycle (`load_model()`, `unload_model()`, `is_loaded`).
    - Offline model safety with `local_files_only=True` support.
    - Model sequence ceiling (256 tokens) with exact WordPiece token counting (`count_tokens()`).
    - Bulk ingestion truncation with structured, privacy-safe `token_truncation_warning` logs.
    - Strict query length validation rejecting oversized queries in `client.embed_text()` and `pipeline.embed_query()`.
  - Pipeline orchestrator (`EmbeddingPipeline`) updated in `src/rag/embeddings/pipeline.py` to use `count_tokens()` for query validation while keeping M3 chunking's `estimate_tokens()` heuristic untouched.
  - Real-model integration test suite (`tests/test_m5_real_inference.py`) implemented with 9 comprehensive tests (`T5-01` through `T5-09`):
    - Model lifecycle lazy loading, vector dimensionality (strictly 384-d), and mathematical $L_2$ unit normalization ($1.0 \pm 10^{-4}$).
    - Semantic relative-ranking verification: $\text{sim}_{\text{rel}} > \text{sim}_{\text{unrel}}$ ($0.5575 > 0.0392$, margin $= +0.5183$).
    - Query-document vector consistency and batching consistency.
    - Bulk truncation structured warnings with zero content/token leakage.
    - Comprehensive validation and missing-weights error handling.
    - Full-corpus benchmark across all 113 raw chunks: 2.90s total, 25.70 ms/chunk, 38.92 chunks/s, 741 MB RSS, 100% pass across all 7 hard correctness gates.
  - Zero-download offline test isolation verified with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
  - Complete repository test suite: 79/79 passing tests (13 M1 + 23 M3 + 34 M4 + 9 M5).
  - Documentation delivered: `docs/design/05-real-embedding-inference.md`, `docs/STATUS.md`, `docs/CHANGELOG.md`, `docs/README.md`.

## Not yet done
- Milestone M6: Database Storage & pgvector integration (`knowledge_base_articles.embedding vector(384)`).
- Milestone M7: Retrieval and evaluation benchmark harness.

## Next milestone action
Advance to Database Storage & pgvector integration (`knowledge_base_articles.embedding vector(384)`) under authoritative Shared Team Contract (`docs/CONTRACTS.md` §5).

## Active external dependency
- Shared Contract Synchronization: Informational note in `docs/PROJECT_CONTEXT.md` §Shared contract can be aligned with `docs/CONTRACTS.md` §5 by contract owner.

## Update this file
At the end of each meaningful work session, update completed items, current milestone, next action, blockers, and relevant commit hash (when available).

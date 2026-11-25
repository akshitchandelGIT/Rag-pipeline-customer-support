# Project Status

## Current milestone
Milestone M3 — Chunking Subsystem Implementation (COMPLETE: Phases 1–5 delivered, tested, audited, and verified against all 10 raw corpus documents. Milestone M5 Embedding & Database Integration is next).

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

## Not yet done
- Milestone M5: Embedding model benchmarking and database integration.
- Milestone M6: Retrieval and evaluation benchmark harness.

## Next milestone action
Begin Milestone M5: Benchmark candidate embedding models and implement vector database storage/migrations under Shared Team Contract v3.0 (1536-d, PostgreSQL UUID PK).

## Update this file
At the end of each meaningful work session, update completed items, current milestone, next action, blockers, and relevant commit hash (when available).

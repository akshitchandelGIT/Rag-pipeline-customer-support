# Changelog

Meaningful, reviewable changes only. Add date, milestone, summary, tests, and commit reference (if available).

## Docs: M2 Contract v3.0 Locked Reconciliation Notes
- Date: 2026-11-25
- Milestone: M2 — Chunking and Ingestion Design (Documentation Maintenance)
- Summary: Added minimal informational reconciliation notes to Section 13.2 of `docs/plans/02-chunking-plan.md` and Section 19.3 of `docs/specs/SPEC-02-chunking.md` explicitly referencing Shared Team Contract v3.0 Locked. Clarified that database PK `id` (UUID), absence of `metadata JSONB`, and 1536-d embedding dimensions are locked in Contract v3.0, while preserving the approved M2 architecture and leaving corpus expansion as the sole active external timeline dependency.
- Tests: Read-only diff verification; 23/23 M3 unit tests passing; 13/13 M1 regression unit tests passing (36 total tests passing).

## Git Hygiene: Revert Unintended M1 Scanner Comment
- Date: 2026-11-25
- Milestone: M1 — Dataset Reconnaissance (Maintenance)
- Summary: Reverted unintended single-line comment edit in `docs/experiments/m1_recon_scanner.py` to restore exact committed content against HEAD.
- Tests: `git diff docs/experiments/m1_recon_scanner.py` confirmed clean (0 diff); 13/13 M1 regression unit tests passing.

## Git Hygiene: Exclude Generated Chunk Previews from Git
- Date: 2026-11-25
- Milestone: M3 — Chunking Subsystem Implementation (Maintenance)
- Summary: Added `docs/experiments/m3_chunk_preview/` to `.gitignore` to prevent raw-derived chunk preview artifacts from being tracked or staged, preserving raw dataset protection while maintaining local inspection files.
- Tests: `git check-ignore -v docs/experiments/m3_chunk_preview/` verified; `git check-ignore -v data/raw/` verified.

## Fix: Structural FSM Parser Docstring Syntax Error
- Date: 2026-11-25
- Milestone: M3 — Chunking Subsystem Implementation (Maintenance)
- Summary: Resolved `SyntaxError` in `src/rag/chunking/parser.py` by consolidating redundant string literals into a single module docstring preceding `from __future__ import annotations`.
- Tests: `python3 -m py_compile src/rag/chunking/parser.py`; 23/23 unit tests passing in `tests/test_m3_chunking.py`; 13/13 M1 regression unit tests passing (36 total passing tests).

## Milestone M3: Chunking Subsystem — Phase 5: Stage 0 Smoke Verification
- Date: 2026-11-24
- Milestone: M3 — Chunking Subsystem Implementation (Phase 5)
- Summary: Executed empirical Stage 0 Smoke Verification of the completed chunking pipeline against all 10 Markdown documents in `data/raw/` (5 Org A, 5 Org B). Verified that: (1) 10/10 documents parsed with 0 errors; (2) 113 total valid chunks generated (56 Org A, 57 Org B); (3) token size distribution measured at min 38, max 169, mean 90.2, median 83.0 tokens (100% of chunks within the 400-token ceiling); (4) table preservation verified intact for SLA commitment tables; (5) sequential chunk indexing $0 \dots N-1$ verified across all documents; (6) 100% metadata fidelity and frontmatter inheritance confirmed; (7) 100% tenant isolation confirmed with zero cross-tenant contamination; (8) 100% determinism confirmed across repeated runs (matching UUID v5 IDs and SHA-256 hashes); (9) 0 unexpected warnings or errors logged. Milestone M3 is complete.
- Tests: 10/10 corpus documents smoke-verified; 23/23 unit tests passing in `tests/test_m3_chunking.py`; 13/13 M1 regression unit tests passing; 39/39 scratch verification tests passing (75 total passing tests).

## Milestone M3: Chunking Subsystem — Phase 4: Authoritative Test Suite
- Date: 2026-11-24
- Milestone: M3 — Chunking Subsystem Implementation (Phase 4)
- Summary: Implemented the authoritative 21-test matrix (T-01 through T-21) plus logging privacy and corpus regression smoke testing in `tests/test_m3_chunking.py` using Python standard library `unittest` exclusively. All tests utilize synthetic in-memory fixtures to guarantee total determinism, repeatability, and zero external I/O. Verifications cover: (1) frontmatter and metadata error handling (`MissingFrontmatterError`, `MalformedFrontmatterError`, empty/missing fields, invalid syntax); (2) strict raw UUID v4 validation and raw whitespace rejection without silent stripping; (3) invalid access level rejection; (4) empty document error handling; (5) ATX heading parsing, level hierarchy, and canonical `heading_path`; (6) dual code fence suppression (both backticks ``` and tildes ~~~) with fence length matching; (7) pipe table detection; (8) procedural list preservation; (9) single-section document chunking; (10) breadcrumb injection; (11) small H2 intro forward-merging (<60 tokens); (12) standalone H2 isolation (<60 tokens, 0 H3 children) with zero sibling merging; (13) table row partitioning, column header/divider re-injection, and canonical `heading_path` decoupling from presentation markers (Invariant F-01); (14) Zone 1 controlled oversize (500 tokens) emitted intact with `is_oversized=True, is_split_item=False`; (15) Zone 2 emergency sentence splitting (900 tokens) with continuation markers; (16) Zone 2 exact character-for-character reconstruction with whitespace retention across diverse delimiters, decimals, abbreviations, repeated punctuation, and Unicode text (Invariant F-03); (17) 100% metadata inheritance; (18) sequential 0-based chunk indexing; (19) deterministic UUID v5 chunk identification; (20) Org A tenant isolation; (21) interleaved multi-tenant batch isolation (alternating Org A and Org B in a single loop); (22) structured logging privacy with zero raw content / sensitive token leakage; (23) conditional corpus regression smoke testing against `data/raw/`.
- Tests: 23 unit tests passing in `tests/test_m3_chunking.py` (`python3 -m unittest tests/test_m3_chunking.py -v`); 13/13 M1 regression unit tests passing; 39/39 scratch verification tests passing (75 total passing tests across the project).

## Milestone M3: Chunking Subsystem — Phase 3: Chunking Engine & Public API
- Date: 2026-11-24
- Milestone: M3 — Chunking Subsystem Implementation (Phase 3)
- Summary: Implemented section-aware chunking engine, breadcrumb injection, and fallback protocols in `src/rag/chunking/engine.py` using Python standard library exclusively. Features include: (1) breadcrumb formatting (`format_breadcrumb`) adhering to hierarchy templates while strictly preserving canonical `DocumentChunk.heading_path` decoupling from presentation markers (Invariant F-01); (2) full constructed chunk token estimation (`estimate_tokens`); (3) H2 section topic isolation and standalone H2 isolation (<60 tokens kept independent without merging into sibling H2s); (4) small H2 intro forward-merging (<60 tokens merged into first H3 child chunk) and substantial H2 intro independence (>=60 tokens emitted independently); (5) Markdown table handling with atomic preservation, budget flush, row splitting on `\n|...|\n` boundaries, and repeated column headers/dividers on continuation chunks; (6) Two-Zone fallback protocol: Zone 1 (400-800 tokens) emitted intact (`is_oversized=True`, `is_split_item=False`); Zone 2 (>800 tokens) split on sentence boundaries (`is_oversized=True`, `is_split_item=True`) with explicit continuation markers; (7) Zone 2 whitespace retention and exact character-level reassembly (`"".join(sub_chunks) == original_source_text`, Invariant F-03); (8) deterministic UUID v5 chunk IDs, sequential 0-based chunk indices, SHA-256 content hashes, and 100% metadata inheritance; (9) non-fatal structured logging with zero raw content/PII leakage; (10) top-level orchestrator `chunk_document` and package exports in `src/rag/chunking/__init__.py`.
- Tests: 13 verification unit tests passing in Phase 3 scratch test suite, including end-to-end processing across all 10 raw corpus documents under `data/raw/` generating 113 valid chunks; 16/16 Phase 2 parser unit tests passing; 10/10 Phase 1 model unit tests passing; 13/13 M1 regression unit tests passing (52 total unit tests passing).

## Milestone M3: Chunking Subsystem — Phase 2: Structural FSM Parser
- Date: 2026-11-24
- Milestone: M3 — Chunking Subsystem Implementation (Phase 2)
- Summary: Implemented bounded structural Markdown FSM parser in `src/rag/chunking/parser.py` using Python standard library exclusively. Features include: (1) frontmatter extraction and validation (`extract_frontmatter`) enforcing non-emptiness (`EmptyDocumentError`), opening fence on line 1 (`MissingFrontmatterError`), unclosed fence detection before EOF (`MalformedFrontmatterError`), strict raw UUID v4 validation (`InvalidMetadataError`), and fixed access levels (`InvalidAccessLevelError`); (2) structural ATX heading parser building hierarchical `ParsedSection` tree with canonical `heading_path`; (3) dual code fence suppression (both backticks ``` and tildes ~~~) completely bypassing heading evaluation inside code blocks with matching fence character and length; (4) atomic table grouping on contiguous pipe rows; (5) list item and block preservation; (6) graceful EOF recovery on unclosed code blocks emitting structured log warnings without content leakage. Public exports added to `src/rag/chunking/__init__.py`.
- Tests: 16 verification unit tests passing in Phase 2 scratch test suite, including successful parsing and validation across all 10 raw corpus documents under `data/raw/`; 10/10 Phase 1 model unit tests passing; 13/13 M1 regression unit tests passing.

## Milestone M3: Chunking Subsystem — Phase 1: Models & Foundational Types
- Date: 2026-11-24
- Milestone: M3 — Chunking Subsystem Implementation (Phase 1)
- Summary: Implemented foundational data structures, custom exception hierarchy (`ChunkingError`, `EmptyDocumentError`, `MissingFrontmatterError`, `MalformedFrontmatterError`, `InvalidMetadataError`, `InvalidAccessLevelError`), strict raw RFC 4122 UUID v4 validation (without whitespace stripping or normalization), deterministic UUID v5 chunk ID generation (`generate_chunk_id`), token estimation (`estimate_tokens`), and SHA-256 content hashing (`compute_content_hash`) in `src/rag/chunking/models.py`. Implemented frozen `DocumentChunk` conforming to Shared Team Contract v3.0 (UUID PK, fixed access levels, zero role identifiers, and canonical `heading_path` decoupling from presentation markers). Package exports initialized in `src/rag/__init__.py` and `src/rag/chunking/__init__.py`.
- Tests: 10 verification unit tests passing for Phase 1 models and utilities; 13/13 M1 regression unit tests passing.

## Milestone M1: Dataset Reconnaissance Tooling & Specification
- Date: 2026-11-23
- Milestone: M1 — Dataset Reconnaissance
- Summary: Implemented standalone, standard-library-only reconnaissance scanner (`docs/experiments/m1_recon_scanner.py`) and comprehensive unit test suite (`tests/test_m1_recon_scanner.py`). Formalized inspection protocol across 13 dimensions, PDF detection boundaries, lightweight frontmatter parsing, exact and near-duplicate detection, and PII safety with zero sensitive leakage across stdout, JSON, and Markdown. Approved M1 Plan and SPEC-01 specification. Resolved finding F-01 by normalizing `scanned_directory` to prevent exposing machine-specific absolute filesystem paths.
- Tests: 13 unit tests passing in `tests/test_m1_recon_scanner.py` (`python3 -m unittest tests/test_m1_recon_scanner.py -v`).

## Initial setup
- Initialized individual RAG repository and GitHub remote.
- Added agent instructions and project documentation scaffolding.

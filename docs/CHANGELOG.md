# Changelog

Meaningful, reviewable changes only. Add date, milestone, summary, tests, and commit reference (if available).

## Milestone M1: Dataset Reconnaissance Tooling & Specification
- Date: 2026-11-23
- Milestone: M1 — Dataset Reconnaissance
- Summary: Implemented standalone, standard-library-only reconnaissance scanner (`docs/experiments/m1_recon_scanner.py`) and comprehensive unit test suite (`tests/test_m1_recon_scanner.py`). Formalized inspection protocol across 13 dimensions, PDF detection boundaries, lightweight frontmatter parsing, exact and near-duplicate detection, and PII safety with zero sensitive leakage across stdout, JSON, and Markdown. Approved M1 Plan and SPEC-01 specification. Resolved finding F-01 by normalizing `scanned_directory` to prevent exposing machine-specific absolute filesystem paths.
- Tests: 13 unit tests passing in `tests/test_m1_recon_scanner.py` (`python3 -m unittest tests/test_m1_recon_scanner.py -v`).

## Initial setup
- Initialized individual RAG repository and GitHub remote.
- Added agent instructions and project documentation scaffolding.

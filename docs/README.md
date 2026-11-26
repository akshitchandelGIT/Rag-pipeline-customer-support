# Documentation Index

## Core Project & Architectural Documentation
- `CONTRACTS.md` — authoritative Shared Team Contract (v3.0 Locked, Section 5: `embedding VECTOR(384)` with `all-MiniLM-L6-v2`).
- `PROJECT_CONTEXT.md` — project goal, scope, shared contract summary, and integration constraints.
- `ARCHITECTURE.md` — evolving architecture and component boundaries.
- `DECISIONS.md` — decision log and architectural trade-offs.
- `STATUS.md` — current milestone status, next actions, and blockers.
- `CHANGELOG.md` — chronological log of meaningful changes and test verifications.
- `DATASET_RECON.md` — dataset reconnaissance framework, scanner tooling, and empirical findings.
- `EVALUATION_PLAN.md` — retrieval quality and security evaluation approach.

## Milestone M1: Dataset Reconnaissance
- `plans/01-dataset-recon-plan.md` — Milestone M1 plan (Dataset Reconnaissance).
- `specs/SPEC-01-dataset-recon.md` — Milestone M1 technical specification (Dataset Reconnaissance).
- `experiments/m1_recon_scanner.py` — Standalone, standard-library reconnaissance scanner.
- `../tests/test_m1_recon_scanner.py` — Unit test suite for M1 reconnaissance scanner.

## Milestone M2 & M3: Chunking Subsystem
- `plans/02-chunking-plan.md` — Milestone M2 plan (Chunking & Ingestion Design).
- `specs/SPEC-02-chunking.md` — Milestone M2 technical specification (Document Chunking & Ingestion Architecture).
- `plans/03-chunking-implementation-plan.md` — Milestone M3 audited implementation plan (Chunking Subsystem).
- `../tests/test_m3_chunking.py` — Authoritative 23-test unit and regression suite for chunking pipeline.

## Milestone M4: Embeddings Subsystem
- `plans/04-embeddings-plan.md` — Milestone M4 architecture and development plan.
- `plans/04-embeddings-implementation-plan.md` — Milestone M4 detailed implementation plan (`all-MiniLM-L6-v2`, 384 dimensions).
- `contracts/RFC-01-embedding-dimension-amendment.md` — Shared Team Contract Amendment RFC (v3.0 -> v3.1: 384-d vector transition).
- `../tests/test_m4_embeddings.py` — Authoritative 34-test unit, boundary, and full-corpus smoke test suite for embeddings pipeline.

## Subdirectories
- `experiments/` — experiment configurations and results as they become meaningful.
- `design/` — detailed design documents for subsystems as they emerge.
- `contracts/` — cross-team contract amendment proposals and formal RFC documents.

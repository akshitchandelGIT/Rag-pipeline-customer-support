# Decision Log

Record decisions with date, status (proposed/accepted/rejected/superseded), context, decision, alternatives, and consequences.

## D-001 — CLI-first, UI later
- Status: Accepted
- Context: Need inspect and evaluate retrieval behavior quickly and transparently.
- Decision: Prioritize a CLI and reproducible tests before adding a demo UI.
- Consequences: Faster feedback; UI will be added only after core retrieval is dependable.

## D-002 — Dataset-driven architecture
- Status: Accepted
- Context: Dataset format, size, content, and metadata are not yet inspected.
- Decision: Do not lock loaders, chunking, embedding model, storage schema, or dependencies before dataset reconnaissance and contract review.
- Consequences: Avoid premature design and unnecessary installations.

## D-003 — Security filtering must be pre-retrieval
- Status: Accepted
- Context: Shared team contract mandates tenant and access-tier filtering in SQL before retrieval/scoring.
- Decision: Implement retrieval queries that constrain org and authorized tiers before ranking.
- Consequences: Security tests must directly validate query/filter behavior and zero unauthorized results.

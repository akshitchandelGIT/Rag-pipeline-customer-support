# Implementation Prompt (Use only after plan approval)

Read `AGENTS.md` and all documentation/code/tests relevant to the approved plan. Implement only the approved scope.

Requirements:
- Do not silently expand scope or refactor unrelated code.
- Preserve tenant and access-level security invariants and shared contracts.
- Add or update tests; run them and report exact results. Do not claim unrun tests passed.
- Inspect the final diff for accidental changes.
- Create/update the relevant Markdown docs, including `docs/STATUS.md`, `docs/CHANGELOG.md`, and `docs/README.md` links when appropriate. For new meaningful components, create a dedicated concise document under `docs/`.
- If the implementation reveals a design decision or contract mismatch, stop and report it rather than silently changing the contract.
- Do not commit, push, merge, delete branches, or perform destructive operations unless explicitly instructed.

At completion, report changed files, design choices, test commands/results, documentation changes, known limitations, and a suggested commit message.

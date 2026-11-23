# Milestone M1: Dataset Reconnaissance — Plan (Revised)

## 1. Objectives and Scope

The objective of Milestone M1 is to plan and conduct a thorough, empirical reconnaissance of the actual Knowledge Base (KB) dataset once available, establishing critical engineering baselines before any ingestion, chunking, embedding, or database schemas are developed.

Following the core operating rules in `AGENTS.md` and Decision `D-002` ("Dataset-driven architecture"):
- We do not lock loaders, chunking strategies, embedding models, storage schemas, or dependencies before empirical reconnaissance.
- Synthetic corpus generation is **strictly out of scope** for M1 (deferred to future testing milestones); M1 plans for the reconnaissance of the actual corpus when provided.
- The Kaggle Customer Support Ticket Dataset referenced in the shared contract is treated as a separate external data source requiring independent evaluation, rather than assumed to be the KB corpus or prematurely transformed.
- Document metadata convention: We recommend YAML frontmatter (`document_id`, `org_id`, `access_level`, `title`, `source`) as a design recommendation. Directory hierarchies assist human navigation but must **never** be treated as an authorization mechanism.
- Embedding dimensions: Reconciling the shared contract's 1536-dimensional requirement with the brief's suggested 384-dimensional model is explicitly deferred to a future embedding-selection milestone. M1 will not select, install, or test embedding models.
- The scope is strictly limited to dataset reconnaissance, inspection tooling, schema mapping, and documentation. No production pipeline, database setup, or third-party dependencies will be introduced.

---

## 2. Fact & Assumption Categorization

To maintain strict scientific and engineering rigor, project knowledge is partitioned into four explicit categories:

### A. Confirmed Facts
1. **Repository State:** Current branch `feature/m1-dataset-recon`, working tree clean.
2. **Corpus Absence:** The actual knowledge-base corpus is **currently unavailable** in the repository. Directories `data/raw/` and `data/processed/` do not exist.
3. **Security Invariants:** Tenant identity (`org_id`) and role must be extracted from verified JWT authentication context, never from query text, prompt content, or directory paths. SQL prefiltering by `org_id` and authorized `access_level` before similarity ranking is mandatory. Leakage target is 0%.
4. **Fixed Test Identities:**
   - Org A (`11111111-1111-1111-1111-111111111111`): Alice (`Org_Admin`), Bob (`Support_Agent`), Charlie (`End_Customer`).
   - Org B (`22222222-2222-2222-2222-222222222222`): Dave (`Org_Admin`).
5. **Database Target Contract:** Shared team contract v2.3 specifies target table `knowledge_base_articles` (`id`, `org_id`, `document_id`, `chunk_index`, `title`, `content`, `access_level`, `embedding`).

### B. Project-Brief Requirements
1. **Target Corpus Profile:** Track 3 brief specifies creating or adapting 50+ Markdown/PDF documents across three explicit access tiers for Org A and Org B:
   - `public` tier (~25 docs): Product FAQs, onboarding guides, password reset steps, shipping/subscription policies.
   - `internal_staff` tier (~15 docs): Support agent troubleshooting runbooks, bug reproduction SOPs, internal escalation macros.
   - `admin_only` tier (~10 docs): Refund approval thresholds, chargeback dispute playbooks, VIP SLA compensation rules, billing exception policies.
2. **Evaluation Standard:** RAG evaluation requires 0% unauthorized tier leakage and evaluation of retrieval recall/precision using explicit ground-truth query-evidence pairs.

### C. Design Assumptions
1. **Metadata Convention:** Documents will ideally carry structured YAML frontmatter (`document_id`, `org_id`, `access_level`, `title`, `source`).
2. **Authorization Boundary:** Authorization is enforced via verified metadata stored in database columns, not by filesystem directory structures.
3. **Kaggle Ticket Dataset Role:** The Kaggle dataset mentioned in `CONTRACTS 4.md` serves as the operational dataset for Track 1 (tickets, customers, invoices). If inspected for RAG utility in later milestones, it will require distinct evaluation rather than assuming equivalence to the KB corpus.

### D. Pending Inputs
1. **Receipt of Actual Corpus:** Physical placement of the KB documents into `data/raw/`.
2. **Kaggle Support Dataset Delivery:** Receipt of Kaggle data if team leadership requests exploratory analysis of ticket resolutions as secondary KB material.

---

## 3. Investigation Steps and 13 Analysis Dimensions

When the actual corpus is staged in `data/raw/`, the M1 investigation will examine and document the following 13 dimensions:

1. **File formats and dataset structure:** Directory hierarchy, file types (`.md`, `.pdf`, `.json`, `.txt`), file sizes, character encodings (UTF-8 validation).
2. **Number of documents and records:** Total document count, distribution by organization (`Org A` vs. `Org B`) and access level (`public`, `internal_staff`, `admin_only`).
3. **Available fields and metadata:** Inventory of frontmatter keys (`document_id`, `org_id`, `access_level`, `title`, `source`, `category`), dates, author metadata.
4. **Missing values and data quality issues:** Empty files, missing titles, untagged access tiers, corrupted encoding, broken markdown syntax.
5. **Document length and content distribution:** Min, max, mean, and median character, word, and estimated token counts; section depth and heading hierarchies.
6. **Duplicate and near-duplicate documents:** Exact duplicates (SHA-256) and semantic/lexical overlap (token set Jaccard similarity).
7. **Language distribution:** Charset verification, language identification, presence of non-English content.
8. **Potential PII or sensitive information:** Safe scanning for email addresses, phone numbers, API keys, passwords, and real customer names without logging raw sensitive data.
9. **Dataset license and provenance:** Origin of source data, usage rights, and attribution requirements.
10. **B2B Multi-tenant metadata:** Fields establishing organizational tenancy, service categorization, and tenant separation.
11. **Contract mapping:** Explicit, non-invented mapping to `knowledge_base_articles` schema (`org_id`, `document_id`, `title`, `content`, `access_level`).
12. **Ingestion and retrieval challenges:** Complex structures such as Markdown tables, code blocks, bulleted runbooks, policy boundaries, and chunk size trade-offs.
13. **Recommended next steps:** Concrete, empirical technical recommendations for M2 (Ingestion, Chunking, and Normalization).

### Separate Assessment of Kaggle Ticket Dataset (If Provided)
If the Kaggle Customer Support Ticket Dataset is supplied:
- Inspect schema, column distributions, and volume.
- Evaluate whether ticket resolution text is clean, coherent, and suitable for RAG knowledge retrieval.
- Do not transform or load tickets into the KB in M1.

---

## 4. Proposed Inspection Tools and Methods

- **Zero-Dependency Python Inspector:** Standalone script (`docs/experiments/m1_recon_scanner.py`) utilizing strictly Python 3 standard library:
  - `pathlib` for safe filesystem traversal.
  - `re` for frontmatter extraction and PII pattern auditing.
  - `hashlib` for SHA-256 deduplication.
  - `statistics` for length and token distribution metrics.
  - `unicodedata` for character set and language sanity checks.
- **CLI Utilities:** Standard Linux utilities (`wc`, `file`, `head`, `sha256sum`, `jq`).
- **Strictly No External Dependencies:** No third-party packages installed or executed during M1.

---

## 5. Files Created or Modified

| File | Action | Purpose |
|---|---|---|
| `docs/plans/01-dataset-recon-plan.md` | NEW | Revised Milestone M1 Plan (this document). |
| `docs/DATASET_RECON.md` | MODIFY | Populated with confirmed facts, project requirements, and framework ready for empirical data. |
| `docs/STATUS.md` | MODIFY | Update milestone status, pending inputs, and next actions. |
| `docs/CHANGELOG.md` | MODIFY | Record M1 planning revision and scope boundaries. |
| `docs/README.md` | MODIFY | Add documentation links for plans and specs. |
| `docs/experiments/m1_recon_scanner.py` | NEW | Reusable, standard-library reconnaissance script. |

---

## 6. Proposed Dataset Analysis Report Structure (`docs/DATASET_RECON.md`)

The final M1 report in `docs/DATASET_RECON.md` will be structured as:
- **1. Executive Summary & Corpus Provenance** (Source origin, license, total volume).
- **2. Structural Inventory & File Formats** (File extensions, directory layout, encodings).
- **3. Quantitative Distributions** (Document counts by tier/org, word and token distributions).
- **4. Metadata Schema & Frontmatter Analysis** (Available vs. missing fields, title consistency).
- **5. Data Quality, Duplication & Language** (Malformed records, SHA-256 duplicates, language breakdown).
- **6. Security, PII & Prompt Injection Audit** (Regex PII scan results, injection risk flags).
- **7. Contract Mapping (`knowledge_base_articles`)** (Direct field mapping table for `org_id`, `access_level`, `title`, `content`).
- **8. Kaggle Ticket Dataset Assessment** (Independent evaluation of ticket data utility if provided).
- **9. Initial Stage 0 Test Set Selection** (5 representative documents selected across tiers for initial testing).
- **10. Pipeline Challenges & Engineering Implications** (Chunking risks, table handling, token budget).
- **11. Recommendations for M2** (Chunking strategy, metadata enrichment, normalizer rules).

---

## 7. Risks and Privacy Considerations

- **PII Protection:** Never output raw customer names, emails, phone numbers, or credentials to markdown files or terminal logs. Output only aggregated frequency statistics.
- **Git Cleanliness:** Ensure `data/raw/` remains strictly in `.gitignore`. Never commit raw data files.
- **Prompt Injection Defense:** Treat all document content as untrusted input. Flag instructions or directive phrasing that attempts to override system prompts or access levels.
- **Directory Hierarchy Security Risk:** Never treat directory structure as an authorization mechanism. Authorization must be bound to validated document metadata.

---

## 8. Verification Steps

1. **Reconnaissance Execution:** Run `python3 docs/experiments/m1_recon_scanner.py` against `data/raw/` once documents are staged and verify clean execution.
2. **Repository Integrity:** Execute `git status` to verify no raw data files are staged or untracked.
3. **Documentation Consistency:** Verify that `docs/DATASET_RECON.md`, `docs/STATUS.md`, and `docs/CHANGELOG.md` link properly and contain no broken references.
4. **Contract Fidelity:** Review the mapping against `CONTRACTS 4.md` (v2.3 Final) to confirm 100% alignment with `knowledge_base_articles` and access tiers.

---

## 9. Acceptance Criteria for M1

- [ ] Clear documentation of dataset availability, distinguishing facts from assumptions.
- [ ] Document metadata convention documented (YAML frontmatter recommended with `document_id`, `org_id`, `access_level`, `title`, `source`).
- [ ] Directory hierarchy documented as organizational only, not an authorization mechanism.
- [ ] Embedding model selection and dimension reconciliation explicitly deferred to future milestones.
- [ ] Synthetic corpus generation documented as out of scope for M1.
- [ ] Reusable standard-library inspection tool (`m1_recon_scanner.py`) ready to execute against `data/raw/`.
- [ ] Investigation protocol for the Kaggle dataset clearly segregated from the core KB corpus.
- [ ] `docs/STATUS.md` and `docs/CHANGELOG.md` updated.
- [ ] Plan approved prior to authoring `SPEC-01-dataset-recon.md`.

---

## 10. Proposed Specification (`SPEC-01`) Contents

Upon user approval of this plan, `docs/specs/SPEC-01-dataset-recon.md` will define:
1. **Corpus Specification & Intake Schema:** Recommended frontmatter fields (`title`, `org_id`, `access_level`, `category`, `source`).
2. **Metadata Validation Rules:** Rules to validate `org_id` and `access_level` from frontmatter without relying on directory paths.
3. **Quality Validation Gates:** Rejection criteria for unparseable, malformed, or empty documents.
4. **Reconnaissance Tool CLI Specification:** CLI arguments, input path handling, and JSON/Markdown reporting format.
5. **Stage 0 Test Document Manifest:** The exact 5 document IDs, access tiers, and expected query types for initial retrieval verification.

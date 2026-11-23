# Specification SPEC-01: Dataset Reconnaissance

**Milestone:** M1 — Dataset Reconnaissance  
**Status:** Draft / Pending Review  
**Governing Plan:** [01-dataset-recon-plan.md](../plans/01-dataset-recon-plan.md)  
**Author:** Akshit Chandel (RAG & Knowledge Base Engineer)  
**Date:** 2026-11-23  

---

## 1. Objective and Scope

### 1.1 Objective
This specification formalizes the engineering procedures, inspection criteria, metadata validation standards, and reporting schema for Milestone M1 (Dataset Reconnaissance). It translates the approved plan ([01-dataset-recon-plan.md](../plans/01-dataset-recon-plan.md)) into an actionable, reproducible technical specification.

The primary goal of M1 is to establish a rigorous, empirical baseline of the Knowledge Base (KB) dataset before any document ingestion, chunking, embedding generation, or database schema implementation is designed.

### 1.2 In-Scope
- Formalizing the dataset inspection protocol across 13 analytical dimensions.
- Defining diagnostic criteria for document counts, file formats, structural layouts, length/token distributions, duplicates, languages, and potential PII.
- Evaluating tenant (`org_id`) and access-tier (`access_level`) metadata requirements against the shared team contract (`CONTRACTS 4.md` v2.3).
- Documenting the proposed YAML frontmatter metadata convention as a design recommendation.
- Defining the standalone, zero-dependency diagnostic inspection script (`docs/experiments/m1_recon_scanner.py`).
- Establishing the exact reporting structure for [DATASET_RECON.md](../DATASET_RECON.md).
- Defining selection criteria for the 5-document Stage 0 baseline test set.

### 1.3 Out-of-Scope (Strict Boundaries)
- **No synthetic corpus generation:** Fabricating or generating mock KB documents is prohibited in M1 (deferred to future testing milestones).
- **No Kaggle ticket transformation:** The Kaggle Customer Support Ticket Dataset is treated as an external dataset requiring separate investigation; it will not be assumed to be the KB corpus or transformed into KB articles in M1.
- **No embedding model selection or dimension reconciliation:** Reconciling the shared contract's 1536-dimensional requirement with the brief's suggested 384-dimensional model (`BAAI/bge-small-en-v1.5`) is deferred to a future embedding-selection milestone. M1 will not select, install, or benchmark embedding models.
- **No ingestion, parsing, or chunking implementations:** No production parsers, chunkers, or loaders will be authored in M1.
- **No database or storage implementation:** No PostgreSQL tables, pgvector extensions, or database schemas will be created.
- **No external dependency installations:** No third-party packages (e.g. `langchain`, `ragas`, `sentence-transformers`) will be installed.
- **No Git commits or pushes:** All work remains uncommitted on branch `feature/m1-dataset-recon` until explicitly approved.

---

## 2. Confirmed Repository and Dataset Availability State

### 2.1 Git and Working Tree State
- **Branch:** `feature/m1-dataset-recon` (derived from `main`).
- **Working Tree:** Clean, with milestone planning and specification documents located under `docs/plans/` and `docs/specs/`.
- **Ignore Rules:** `.gitignore` explicitly excludes `data/raw/` and `data/processed/`, preventing accidental exposure or tracking of raw data.

### 2.2 Dataset Availability Status
- **Current Status:** **CONFIRMED UNAVAILABLE.**
- Neither raw nor processed Knowledge Base documents exist in the repository root or subdirectories.
- Directory `data/raw/` does not exist.
- Directory `tests/` contains no sample data or test fixtures.
- **Rule of Scientific Integrity:** M1 will not claim or invent empirical statistics, document counts, or token distributions until the actual corpus is staged in `data/raw/`.

---

## 3. Four-Register Information Model

All information within this project is strictly partitioned into four explicit registers to prevent conflation of confirmed facts with design assumptions:

| Register | Definition | Items in this Register |
|---|---|---|
| **Confirmed Facts** | Empirically verified truths in code, repo, or signed team contract. | 1. Working tree is clean on `feature/m1-dataset-recon`.<br>2. Knowledge Base dataset is currently absent from repository.<br>3. Shared contract v2.3 mandates table `knowledge_base_articles` (`org_id`, `document_id`, `chunk_index`, `title`, `content`, `access_level`, `embedding`).<br>4. Mandatory SQL prefilter: `WHERE org_id = $1 AND access_level = ANY($2)` before vector similarity scoring.<br>5. Authorized access tiers: `public` (End_Customer), `public` + `internal_staff` (Support_Agent), `public` + `internal_staff` + `admin_only` (Org_Admin).<br>6. Fixed test identities: Org A (`11111111-1111-1111-1111-111111111111`) and Org B (`22222222-2222-2222-2222-222222222222`). |
| **Project-Brief Requirements** | Targets specified in project guidelines and integration blueprints. | 1. 50+ unstructured Markdown/PDF documents divided across Org A and Org B.<br>2. Document tier breakdown: `public` (~25 docs), `internal_staff` (~15 docs), `admin_only` (~10 docs).<br>3. Zero cross-tenant and zero unauthorized-tier retrieval leakage standard.<br>4. Progressive evaluation approach starting with Stage 0 (5 documents). |
| **Design Assumptions** | Architectural proposals requiring validation or future implementation. | 1. Documents will ideally use YAML frontmatter for metadata encapsulation.<br>2. Directory hierarchy is an organizational aid for humans, not an authorization boundary.<br>3. The Kaggle Customer Support Ticket Dataset belongs to Track 1 (relational tickets/customers/invoices); its relevance to KB is secondary and unverified. |
| **Pending Inputs** | External dependencies required before execution can proceed. | 1. Physical staging of the actual Knowledge Base corpus into `data/raw/`.<br>2. Delivery or staging of Kaggle ticket data if team leadership requests RAG ticket-resolution exploration. |

---

## 4. Dataset Inspection Criteria (The 13 Dimensions)

When the actual corpus is placed into `data/raw/`, the inspection must evaluate and record the following 13 dimensions:

### D1. File Types, Structural Formats, and PDF Inspection Boundaries
- **Supported Formats:** `.md` (Markdown), `.txt` (Plain text), `.json` (Structured articles), `.pdf` (Portable Document Format).
- **Encoding:** Validate UTF-8 compliance and detect any non-UTF-8 or byte-order mark (BOM) anomalies in text files.
- **Directory Hierarchy:** Analyze directory depth and folder hierarchy.
- **PDF Limitations (File Detection vs. Content Inspection):**
  - Python's standard library does not provide native PDF text extraction capability.
  - For `.pdf` files, the standard-library scanner performs **file detection and filesystem metadata logging only** (file existence, relative path, byte size, modification timestamp).
  - The scanner must **not** attempt or claim to extract, parse, or inspect the text content of PDF files without approved tooling.
  - All detected PDF files must be explicitly classified and reported in the report under:
    `"detected_uninspected_pdfs": [{"relative_path": "...", "size_bytes": 12345}]`.
  - Full content inspection (character/word counts, estimated tokens, frontmatter metadata, PII regex scanning) is strictly bounded to text-based files (`.md`, `.txt`, `.json`). If full text inspection of PDFs is required later, approved tools or dependencies will be evaluated and decided under a dedicated decision.

### D2. Document and Record Counts
- Total file count across the dataset.
- File counts broken down by tenant: `Org A` vs. `Org B`.
- File counts broken down by access tier: `public`, `internal_staff`, `admin_only`.
- Detection of unassigned or ambiguously assigned files.

### D3. Metadata Inventory and Frontmatter Consistency
- Detection of YAML frontmatter delimiters (`---`) in text files.
- Inventory of present keys: `document_id`, `org_id`, `access_level`, `title`, `source`, `category`, `last_updated`.
- Evaluation of key naming consistency (e.g. `access_level` vs `access_tier` vs `tier`).

### D4. Data Quality and Missing Fields
- Empty or zero-byte file detection.
- Documents with missing titles or fallback to filename-derived titles.
- Missing or malformed `org_id` (must be a valid UUID).
- Missing or invalid `access_level` (must match canonical enum: `public`, `internal_staff`, `admin_only`).
- Broken markdown syntax (unclosed code fences, malformed tables).

### D5. Document Length and Content Distribution (Text Files)
- Compute character count distribution: min, max, mean, median, standard deviation.
- Compute word count distribution: min, max, mean, median, standard deviation.
- Compute estimated token count distribution using the standard heuristic ($\text{tokens} \approx \text{words} \times 1.33$).
- Analyze structural content depth: count of top-level headers (H1), subheaders (H2, H3), lists, and tables per document.

### D6. Duplicate and Near-Duplicate Detection (Heuristic Boundaries)
- **Exact Duplicates:** Compute SHA-256 cryptographic hashes for full file content. Group and list identical documents.
- **Near-Duplicates (Lexical Overlap Heuristic):**
  - Compute normalized whitespace/lowercase word set Jaccard similarity:
    $$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
  - Flag document pairs with $J(A, B) \ge 0.85$ for manual review.
  - **Important Methodological Limitations:**
    - Jaccard token overlap is a **surface-level lexical heuristic**, not semantic duplicate or paraphrase detection. It cannot identify semantically equivalent text written with different vocabulary.
    - **Computational Scaling:** Pairwise comparison requires $\frac{N(N-1)}{2} = O(N^2)$ comparisons. While computationally acceptable for the target corpus of ~50 documents ($\approx 1,225$ comparisons), it becomes computationally prohibitive on large corpora. Semantic deduplication and scalable hashing (e.g., MinHash or vector-similarity clustering) are reserved for future pipeline milestones.

### D7. Language Distribution
- Character set verification: ratio of ASCII characters to non-ASCII characters.
- Unicode block analysis: verify presence of Latin-1, accented characters, or non-Latin scripts.
- Assess whether the corpus is exclusively English or requires multilingual tokenization considerations.

### D8. PII and Sensitive Data Audit (Strict Redaction & Safety Rules)
- Scan text content for potential Personally Identifiable Information (PII) using strict regex patterns:
  - Email addresses: `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b`
  - Phone numbers: `\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b`
  - Credit card numbers: `\b(?:\d{4}[-\s]?){3}\d{4}\b`
  - API Keys / Tokens: `\b(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|Bearer\s+[a-zA-Z0-9._-]+)\b`
  - Social Security Numbers: `\b\d{3}-\d{2}-\d{4}\b`
- **Mandatory Privacy & Redaction Rules:**
  - **Zero Payload Exposure:** The scanner, console logs, JSON reports, and markdown documentation must **never** print, log, or write detected sensitive values or snippets.
  - **Relative Path Referencing:** Full absolute filesystem paths (e.g. `/home/akshit/...`) must not be exposed. Documents must be referenced strictly by their repository/dataset-relative path (e.g. `org_a/public/faq_01.md`) or their `document_id`.
  - Reports output only aggregate counts, pattern types, and relative document identifiers.

### D9. Dataset License and Provenance
- Identify corpus source: enterprise documentation, public domain FAQ, synthetic customer service pack, or internal runbooks.
- Verify licensing constraints (Apache 2.0, MIT, Creative Commons, proprietary enterprise).
- Document attribution obligations or redistribution restrictions.

### D10. Multi-Tenant B2B Metadata
- Assess whether documents are cleanly segregated by organization.
- Check for cross-tenant references or leaking entity names (e.g. Org A document mentioning Org B confidential policies).
- Verify business domain categorization (`billing`, `technical`, `account`, `general`).

### D11. Shared Contract Schema Mapping
- Formally map raw document fields to the PostgreSQL `knowledge_base_articles` target table:
  - `org_id` $\leftarrow$ Verified organization UUID.
  - `document_id` $\leftarrow$ Document UUID.
  - `title` $\leftarrow$ Extracted title from frontmatter or first H1 header.
  - `content` $\leftarrow$ Document body text.
  - `access_level` $\leftarrow$ Validated access tier string (`public`, `internal_staff`, `admin_only`).
- Prohibit inventing or guessing missing fields; unmapped fields must be reported as gaps.

### D12. Ingestion and Retrieval Challenges
- Identify formatting that creates chunking hazards:
  - Markdown tables (risk of table row fragmentation across chunk boundaries).
  - Code blocks and configuration snippets (risk of splitting syntax).
  - Numbered procedural runbooks (risk of losing prerequisite steps if split).
  - Callout blocks (`> [!NOTE]`, `> [!WARNING]`).

### D13. Actionable Next Steps
- Synthesize empirical findings into exact chunking, metadata enrichment, and normalization requirements for the future ingestion and chunking milestone.

---

## 5. Metadata Specification: Proposed YAML Frontmatter Convention

### 5.1 Design Proposal (Not Implemented Code)
To establish clean, self-describing documents, this specification proposes adopting YAML frontmatter embedded at the start of every Markdown document.

```markdown
---
document_id: "550e8400-e29b-41d4-a716-446655440000"
org_id: "11111111-1111-1111-1111-111111111111"
access_level: "public"
title: "Standard Billing and Invoice Resolution Policy"
source: "https://kb.orga.com/billing/invoice-policy"
category: "billing"
last_updated: "2026-11-20"
---

# Standard Billing and Invoice Resolution Policy

Document body text begins here...
```

### 5.2 Field Schema Definition

| Field Name | Type | Mandatory? | Canonical Values / Validation Rules | Description |
|---|---|---|---|---|
| `document_id` | UUID string | Yes | Valid RFC 4122 UUID v4 | Unique identifier for the source article. |
| `org_id` | UUID string | Yes | Must match Org A (`11111111-1111-1111-1111-111111111111`) or Org B (`22222222-2222-2222-2222-222222222222`) | Enforces tenant ownership. |
| `access_level` | String enum | Yes | `'public'`, `'internal_staff'`, `'admin_only'` | Enforces role-based access filtering. |
| `title` | String | Yes | 1 to 255 characters; non-empty | Article title for citation and display. |
| `source` | String / URI | No | Valid URI or local reference path | Lineage and provenance tracking. |
| `category` | String enum | No | `'billing'`, `'technical'`, `'account'`, `'general'` | Canonical ticket/KB category alignment. |
| `last_updated` | ISO Date | No | `YYYY-MM-DD` format | Document currency tracking. |

### 5.3 Critical Security Rule: Directory Paths Are Not Authorization Boundaries
- While storing files in directories like `data/raw/org_a/public/` is recommended for human organization, **directory paths must never be treated as an authorization mechanism**.
- Authorization in the RAG retrieval pipeline is strictly enforced by:
  1. Reading validated document metadata during ingestion.
  2. Storing `org_id` and `access_level` in strongly-typed database columns.
  3. Pre-filtering via SQL queries based on verified JWT claims:
     ```sql
     SELECT * FROM knowledge_base_articles
     WHERE org_id = :jwt_org_id 
       AND access_level = ANY(:authorized_tiers)
     ORDER BY embedding <=> :query_embedding
     LIMIT :top_k;
     ```

---

## 6. Investigation Protocol for Kaggle Ticket Dataset

If the Kaggle Customer Support Ticket Dataset (`Customer Support Ticket Dataset.csv`) is provided for investigation:
1. **Separation of Concerns:** The Kaggle dataset will be analyzed in an isolated section of [DATASET_RECON.md](../DATASET_RECON.md), strictly separate from the core KB corpus.
2. **Analysis Dimensions:**
   - Total record count and file size.
   - Column schema audit: mapping against Section 7 of `CONTRACTS 4.md` (`Ticket Subject`, `Ticket Description`, `Resolution`, `Customer Name`, etc.).
   - Null rate in `Resolution` field: evaluate whether resolved tickets contain sufficient detail to serve as synthetic or supplemental KB troubleshooting articles.
   - PII presence in customer fields (`Customer Name`, `Customer Email`).
3. **Execution Boundary:** No tickets will be ingested or transformed into `knowledge_base_articles` during M1.

---

## 7. Stage 0 Baseline Document Selection Criteria

### 7.1 Purpose and Bounded Scope
To support the progressive evaluation strategy outlined in [EVALUATION_PLAN.md](../EVALUATION_PLAN.md), the reconnaissance process will select a minimal subset of **five representative documents** from the actual corpus once available.

> [!IMPORTANT]
> **Stage 0 Baseline vs. Comprehensive Security Coverage:**
> The five documents defined below constitute a representative **Stage 0 smoke baseline** designed to verify basic end-to-end pipeline wiring, schema field population, and basic query mechanics.
> **They do not constitute comprehensive authorization or tenant-isolation test coverage.** Comprehensive security testing—including exhaustive tenant boundary tests, role escalation prevention, adversarial injections, and zero-leakage verification—is a separate requirement detailed in [EVALUATION_PLAN.md](../EVALUATION_PLAN.md) (Stages 1 through 3).

### 7.2 Five-Document Smoke Baseline Composition

| Slot | Organization | Access Level | Selection Criteria / Baseline Purpose |
|---|---|---|---|
| **DOC-01** | Org A | `public` | Clean FAQ or onboarding document; standard happy-path customer retrieval query. |
| **DOC-02** | Org A | `internal_staff` | Technical troubleshooting SOP; accessible by Support Agent, excluded for Customer. |
| **DOC-03** | Org A | `admin_only` | High-value refund policy or escalation playbook; accessible only by Org Admin. |
| **DOC-04** | Org B | `public` | Public FAQ belonging to Org B; verifies cross-tenant separation on smoke queries. |
| **DOC-05** | Org A | `public` or `internal_staff` | Document featuring structural complexity (Markdown table or code snippet) for chunking smoke tests. |

---

## 8. Diagnostic Tooling Specification (`m1_recon_scanner.py`)

### 8.1 Technical Constraints & File Mutation Boundaries
- Must use **Python 3 standard library only** (`pathlib`, `json`, `csv`, `re`, `hashlib`, `statistics`, `unicodedata`, `sys`, `argparse`).
- **Zero external package dependencies.**
- **Strict Read-Only Enforcement on Scanned Inputs:**
  - The scanner must **never** modify, mutate, rewrite, or rename the scanned source files or directories.
  - The scanner is permitted only to write its designated diagnostic report outputs (e.g. specified via `--output-json <path>`).

### 8.2 Execution Interface
```bash
python3 docs/experiments/m1_recon_scanner.py --data-dir data/raw --output-json docs/experiments/recon_results.json
```

### 8.3 Output Schema
The script will output a structured JSON summary conforming to:
```json
{
  "scan_metadata": {
    "timestamp": "ISO-8601 string",
    "scanned_directory": "relative string path",
    "total_files": 0
  },
  "counts_by_extension": {},
  "counts_by_tenant": {},
  "counts_by_access_tier": {},
  "detected_uninspected_pdfs": [],
  "length_statistics": {
    "character_counts": {"min": 0, "max": 0, "mean": 0.0, "median": 0.0},
    "word_counts": {"min": 0, "max": 0, "mean": 0.0, "median": 0.0},
    "estimated_tokens": {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
  },
  "exact_duplicates": [],
  "near_duplicate_candidates": [],
  "pii_findings": {
    "email_count": 0,
    "phone_count": 0,
    "credit_card_count": 0,
    "api_key_count": 0,
    "ssn_count": 0,
    "affected_documents": []
  },
  "data_quality_issues": {
    "empty_files": [],
    "missing_frontmatter": [],
    "invalid_org_id": [],
    "invalid_access_tier": []
  }
}
```
*Note: `affected_documents` contains relative document paths or document IDs only. No matched sensitive data strings or full absolute filesystem paths are included.*

### 8.4 Scanner Verification via Temporary Test Fixtures
To verify the scanner implementation while the actual corpus is absent:
- Testing will execute against **small, temporary test fixtures** (2–3 minimal markdown files placed in an isolated scratch/temporary directory such as a temporary directory created via `tempfile`).
- The test fixture files must be clearly identified as temporary harness fixtures and deleted immediately after verification.
- Testing the scanner against temporary fixtures must **not** generate or be represented as a synthetic Knowledge Base corpus.

---

## 9. Deliverables and Expected Report Structure

### 9.1 Milestone Deliverables
1. **`docs/plans/01-dataset-recon-plan.md`:** Approved milestone plan.
2. **`docs/specs/SPEC-01-dataset-recon.md`:** Approved milestone specification (this document).
3. **`docs/experiments/m1_recon_scanner.py`:** Reusable standard-library inspection tool.
4. **`docs/DATASET_RECON.md`:** Completed reconnaissance report populated with empirical findings once data is available.
5. **`docs/STATUS.md` & `docs/CHANGELOG.md`:** Updated progress tracking.

### 9.2 Expected Structure of `docs/DATASET_RECON.md`
The final report must contain the following sections:
- **1. Executive Summary & Corpus Provenance:** Source origin, volume, license terms.
- **2. Structural Inventory & Format Breakdown:** File counts, extensions, directory tree, and PDF detection notices.
- **3. Quantitative Distributions:** Word, character, and token statistics for text files.
- **4. Metadata & Frontmatter Audit:** Inventory of keys, presence rates, syntax compliance.
- **5. Data Quality, Duplication & Language:** Empty files, exact duplicates (SHA-256), near-duplicates (lexical heuristic), charset analysis.
- **6. Security, PII & Prompt Injection Audit:** PII pattern frequency summary (redacted) and injection risk flags.
- **7. Contract Mapping (`knowledge_base_articles`):** Field mapping matrix and unmapped attribute audit.
- **8. Kaggle Customer Support Ticket Dataset Evaluation:** (If provided) Schema, volume, and suitability notes.
- **9. Stage 0 Test Document Selection:** Table of 5 baseline documents with IDs and expected query categories.
- **10. Challenges & Recommendations for Ingestion:** Chunking, table handling, normalizer requirements for the future ingestion and chunking milestone.

---

## 10. Acceptance Criteria and Verification Steps

### 10.1 Acceptance Criteria
- [ ] Specification `SPEC-01-dataset-recon.md` created, peer-reviewed, and aligned with `01-dataset-recon-plan.md`.
- [ ] Confirmed facts, project requirements, design assumptions, and pending inputs clearly segregated.
- [ ] Zero external dependencies required or installed.
- [ ] Working tree clean, with `.gitignore` verified to ignore raw data.
- [ ] Diagnostic scanner (`m1_recon_scanner.py`) verified against temporary fixtures without modifying source files.
- [ ] When corpus is received in `data/raw/`, `docs/DATASET_RECON.md` populated with genuine empirical findings across all 13 dimensions.
- [ ] No synthetic corpus generation or unauthorized schema creation performed.

### 10.2 Verification Commands
```bash
# 1. Verify git working tree cleanliness
git status

# 2. Verify git ignore rule for raw dataset directory
git check-ignore data/raw/*

# 3. Verify documentation links across docs directory
python3 -c "
import pathlib, re
for p in pathlib.Path('docs').rglob('*.md'):
    text = p.read_text()
    for link in re.findall(r'\[.*?\]\((.*?)\)', text):
        if not link.startswith('http') and not link.startswith('file://') and not (p.parent / link.split('#')[0]).exists():
            print(f'Broken link in {p}: {link}')
print('Link check passed.')
"
```

---

## 11. Explicit Exclusions and Risks

| Risk | Impact | Mitigation Strategy |
|---|---|---|
| **Premature Schema Locking** | Creating database tables before knowing chunk lengths or metadata shapes forces breaking migrations later. | M1 strictly prohibits schema creation or DB setup. All database work is deferred to future storage and retrieval milestones. |
| **Directory-Based Security Failure** | Relying on folder paths (`/public/`, `/internal/`) for authorization leaks documents if files are moved. | Specification explicitly establishes that authorization must be bound to validated document metadata, never folder paths. |
| **PII Leakage in Documentation** | Exposing real customer names, emails, card numbers, or credentials in Git logs or markdown reports violates privacy laws. | Scanner regex reports counts and relative document IDs only. Raw PII matches are never printed, written, or committed. |
| **Prompt Injection Hazards** | Untrusted document content containing prompt-injection phrasing (e.g. "Ignore previous instructions") could hijack the LLM orchestrator. | M1 inspection audits text for directive patterns; architecture enforces that retrieved chunks are treated as untrusted data, never instructions. |
| **Embedding Dimension Mismatch** | Schema specifies 1536 dimensions while model suggested is 384 dimensions. | Defer embedding model selection to the future embedding-selection milestone; flag dimension parity check in the M1 report as a prerequisite for schema design. |
| **PDF Inspection Overreach** | Standard-library scanner failing on or corrupting binary PDF structures. | Clearly bound scanner to detect PDFs as uninspected files. Defer deep PDF parsing to tooling decided in future milestones. |

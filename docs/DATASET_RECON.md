# Dataset Reconnaissance

**Milestone:** M1 — Dataset Reconnaissance  
**Status:** Tooling and Specification Complete — Awaiting Actual Corpus Delivery  
**Governing Specification:** [SPEC-01-dataset-recon.md](specs/SPEC-01-dataset-recon.md)  
**Governing Plan:** [01-dataset-recon-plan.md](plans/01-dataset-recon-plan.md)  
**Diagnostic Scanner:** [m1_recon_scanner.py](experiments/m1_recon_scanner.py)  
**Unit Test Suite:** `tests/test_m1_recon_scanner.py` (12/12 tests passing)  

---

## 1. Executive Summary & Corpus Status

- **Corpus Availability:** The actual Knowledge Base corpus is **currently absent from the repository** (`data/raw/` and `data/processed/` do not exist).
- **Non-Fabrication Invariant:** In accordance with `AGENTS.md` and Decision `D-002`, empirical document counts, token distributions, and quality metrics are **not** fabricated or synthesized.
- **Milestone Accomplishment:** Completed the formal specification ([SPEC-01](specs/SPEC-01-dataset-recon.md)), standalone standard-library scanner (`docs/experiments/m1_recon_scanner.py`), and comprehensive unit test harness (`tests/test_m1_recon_scanner.py`). The pipeline is fully prepared to execute immediate reconnaissance the moment the actual corpus is staged in `data/raw/`.

---

## 2. Four-Register Project Model

| Register | Status & Description |
|---|---|
| **Confirmed Facts** | 1. Working tree is clean on `feature/m1-dataset-recon`.<br>2. KB corpus is absent; raw data is ignored via `.gitignore`.<br>3. Shared contract table: `knowledge_base_articles` (`id`, `org_id`, `document_id`, `chunk_index`, `title`, `content`, `access_level`, `embedding`).<br>4. Mandatory SQL prefilter: `WHERE org_id = $1 AND access_level = ANY($2)`. Zero leakage tolerance.<br>5. Authorized tiers: `public` (End_Customer), `public` + `internal_staff` (Support_Agent), `public` + `internal_staff` + `admin_only` (Org_Admin).<br>6. Test identities: Org A (`11111111-1111-1111-1111-111111111111`) and Org B (`22222222-2222-2222-2222-222222222222`). |
| **Project-Brief Requirements** | 1. 50+ unstructured Markdown/PDF documents divided across Org A and Org B.<br>2. Tier breakdown: `public` (~25 docs), `internal_staff` (~15 docs), `admin_only` (~10 docs).<br>3. Zero cross-tenant and zero unauthorized-tier retrieval leakage standard.<br>4. Progressive evaluation approach starting with Stage 0 (5 documents). |
| **Design Assumptions** | 1. Documents will ideally use YAML frontmatter for metadata encapsulation.<br>2. Directory hierarchy assists human organization, but is **not** an authorization mechanism.<br>3. The Kaggle Customer Support Ticket Dataset belongs to Track 1; its relevance to the KB is secondary and requires separate evaluation. |
| **Pending Inputs** | 1. Physical staging of the actual Knowledge Base corpus into `data/raw/`.<br>2. Kaggle dataset delivery if team leadership requests exploratory analysis of ticket resolutions as secondary KB material. |

---

## 3. Diagnostic Tooling & Capabilities

The M1 scanner (`docs/experiments/m1_recon_scanner.py`) provides automated inspection without external dependencies:

```bash
# Basic run: outputs JSON to stdout
python3 docs/experiments/m1_recon_scanner.py --data-dir data/raw --pretty

# Write structured report to file
python3 docs/experiments/m1_recon_scanner.py --data-dir data/raw --output-json docs/experiments/recon_results.json --pretty
```

### Inspection Capabilities:
1. **File Discovery:** Recursively catalogs all files, file extensions, and file sizes.
2. **PDF Boundary Handling:** Detects `.pdf` files and logs them under `detected_uninspected_pdfs` with relative paths and byte sizes. Explicitly avoids attempting binary text extraction.
3. **Text Inspection:** Evaluates UTF-8 text formats (`.md`, `.txt`, `.json`), catching `UnicodeDecodeError` under `encoding_errors`.
4. **Lightweight Frontmatter Parsing:** Parses flat `key: value` pairs within `---` fences. Captures malformed or unsupported YAML structures in `malformed_frontmatter`.
5. **Metadata Validation:** Verifies `org_id` against UUID v4 regex and `access_level` against `{'public', 'internal_staff', 'admin_only'}`. Flags missing titles.
6. **Quantitative Distributions:** Calculates character, word, and estimated token counts ($\text{tokens} = \text{round}(\text{words} \times 1.33)$) with min, max, mean, and median metrics.
7. **Deduplication:** Groups identical files via full-content SHA-256 hashes.
8. **Near-Duplicate Lexical Overlap:** Pairwise Jaccard similarity on word token sets ($J(A, B) \ge 0.85$) with zero-division guards for empty files.
9. **PII Safety Audit:** Regex match counts for emails, phone numbers, credit cards, API keys, and SSNs. Strictly excludes matched values or content snippets from all outputs.
10. **Read-Only Guarantee:** Verified to leave source files and directory contents 100% untouched.

---

## 4. Stage 0 Smoke Baseline Selection Criteria

Once the actual corpus is staged, five representative documents will be selected to verify end-to-end pipeline wiring:

| Document ID | Target Tenant | Access Level | Selection Criteria / Purpose |
|---|---|---|---|
| **DOC-01** | Org A | `public` | Clean FAQ/onboarding guide; standard happy-path customer query verification. |
| **DOC-02** | Org A | `internal_staff` | Technical troubleshooting runbook; accessible by staff, blocked for customers. |
| **DOC-03** | Org A | `admin_only` | High-value refund policy or escalation playbook; accessible only by Org Admin. |
| **DOC-04** | Org B | `public` | Org B public FAQ; verifies cross-tenant isolation on smoke queries. |
| **DOC-05** | Org A | `public` / `internal_staff` | Structurally complex document (tables or code fences) for chunking smoke tests. |

> [!NOTE]
> Stage 0 is strictly a minimal smoke baseline to verify pipeline wiring. Comprehensive tenant and access-tier security evaluation across all role permutations is a separate requirement detailed in [EVALUATION_PLAN.md](EVALUATION_PLAN.md).

---

## 5. Kaggle Customer Support Ticket Dataset Protocol

If the Kaggle dataset is supplied for evaluation:
- The dataset will be inspected in an isolated section of this report.
- We will evaluate column consistency against `CONTRACTS 4.md` Section 7 and audit the `Resolution` field for substantive troubleshooting content.
- Tickets will **not** be automatically converted or loaded into `knowledge_base_articles` during M1.

---

## 6. Corpus Intake Runbook

When the actual corpus files are delivered:
1. Place the documents into `data/raw/` (e.g. `data/raw/org_a/...`, `data/raw/org_b/...`).
2. Verify `.gitignore` prevents git tracking:
   ```bash
   git check-ignore data/raw/*
   ```
3. Execute the reconnaissance scanner:
   ```bash
   python3 docs/experiments/m1_recon_scanner.py --data-dir data/raw --output-json docs/experiments/recon_results.json --pretty
   ```
4. Populate this document with the empirical outputs (counts, distributions, duplicates, quality findings).

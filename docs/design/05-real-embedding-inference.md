# Milestone M5: Real Embedding Inference Design & Empirical Benchmark

## 1. Overview & Architectural Purpose
Milestone M5 transitions the RAG pipeline from synthetic mock embeddings to real, in-process, semantic embedding inference powered by `sentence-transformers/all-MiniLM-L6-v2`.

This subsystem produces 384-dimensional unit-normalized float vectors strictly adhering to the Shared Team Contract (`docs/CONTRACTS.md` Section 5: `embedding VECTOR(384)`) and RFC-01.

All inference executes locally on CPU with zero network egress, fully preserving multi-tenant isolation, data privacy, and strict sequence boundaries.

---

## 2. Component Architecture & Lifecycle

### 2.1 Class Structure & Interfaces
- **`BaseEmbeddingClient`** (`src/rag/embeddings/client.py`):
  Abstract base class defining the model-independent embedding interface:
  - `embed_texts(texts: List[str]) -> List[Tuple[float, ...]]`
  - `embed_text(text: str) -> Tuple[float, ...]`
  - `dimensions: int`
  - `model_name: str`
  - `count_tokens(text: str) -> int` (default fallback heuristic `ceil(words * 1.33)` for backward-compatible test doubles)

- **`SentenceTransformerEmbeddingClient`** (`src/rag/embeddings/client.py`):
  Concrete implementation using `sentence_transformers.SentenceTransformer`:
  - **Lazy Model Loading:** Heavy weights (~90MB) are not loaded during `__init__`. Loading is deferred until the first inference call, first `count_tokens()` call, or explicit `client.load_model()`.
  - **Memory Management:** `client.unload_model()` dereferences the PyTorch model and executes `gc.collect()`, resetting `is_loaded` to `False`.
  - **CPU Enforcement:** Loads model strictly on `device="cpu"` under `torch.inference_mode()`.
  - **Offline Safety:** Supports `local_files_only=True` to guarantee zero network calls. Raises `EmbeddingInferenceError` on missing weights.

- **`EmbeddingPipeline`** (`src/rag/embeddings/pipeline.py`):
  High-level orchestrator:
  - Partitions chunks by `org_id` before batching to ensure cross-tenant data is never co-mingled in any client call.
  - Slices batches respecting `config.batch_size` (default 64).
  - Uses 5-step indexed reassembly to preserve 100% strict 1:1 positional caller order.
  - Intercepts interactive queries via `embed_query()`, validating actual token counts with `client.count_tokens()`.

---

## 3. Tokenizer Integration & Sequence Limits

### 3.1 Heuristic vs Exact Tokenizer Decoupling
- **M3 Chunking Subsystem:** Retains `estimate_tokens(text) = math.ceil(word_count * 1.33)` untouched. Used exclusively for heuristic paragraph splitting, table chunking budgets, and oversized zone fallbacks.
- **M5 Embedding Subsystem:** Integrates the model's actual BERT WordPiece tokenizer (`_model.tokenizer(text, add_special_tokens=True)`).
  - Model sequence ceiling: 256 tokens.
  - Special tokens: `[CLS]` (id 101) and `[SEP]` (id 102) consume 2 tokens.
  - Usable text capacity: $256 - 2 = 254$ WordPiece tokens.

### 3.2 Ingestion vs Query Validation Protocols
| Protocol | Scope | Behavior on Overflow (>256 tokens) | Mechanism |
|---|---|---|---|
| **Bulk Ingestion** | `embed_texts()` | Permitted truncation (`truncation=True`) with structured content-free warning. Outliers do not abort batches. | `logger.warning()` with `category="token_truncation_warning"`, `actual_tokens`, `max_seq_length`, `batch_index`. Zero raw text logged. |
| **Search Queries** | `embed_text()` / `embed_query()` | Strict rejection with `EmbeddingValidationError`. Silently truncating queries corrupts user intent. | Evaluates `count_tokens(query) > 256` before inference; raises descriptive exception. |

---

## 4. Empirical Benchmark Telemetry (Stage 0 Raw Corpus)

The full-corpus integration benchmark (`T5-09` in `tests/test_m5_real_inference.py`) was executed across all 10 raw Markdown documents in `data/raw/` (5 Org A, 5 Org B), producing 113 valid chunks via M3 chunking.

### 4.1 Benchmark Measurements

| Metric | Measured Value | Target / Requirement | Status |
|---|---|---|---|
| **Total Chunks Processed** | 113 | 113 (56 Org A, 57 Org B) | PASS |
| **Total Wall-Clock Time** | 2.9037 s | < 15.0 s (Observational) | PASS |
| **Average Per-Chunk Latency** | 25.70 ms/chunk | Observational Telemetry | RECORDED |
| **Effective Inference Throughput** | 38.92 chunks/s | Observational Telemetry | RECORDED |
| **Peak Resident Memory (RSS)** | 741.44 MB | < 1.5 GB | PASS |

### 4.2 Invariant Verification Gates (Hard Pass/Fail)

1. **Count Match:** Exactly 113 `EmbeddedChunk` instances generated (1:1 correspondence).
2. **Positional Order Preservation:** 100% strict 1:1 positional input ordering preserved across all indices ($0 \dots 112$).
3. **Metadata Fidelity:** 100% attribute fidelity (`id`, `chunk_id`, `document_id`, `org_id`, `access_level`, `chunk_index`, `title`, `content`).
4. **Tenant Batch Isolation:** Exactly 2 client batches executed (Batch 0: 56 chunks, Batch 1: 57 chunks). Batch 0 contains 100% Org A chunks; Batch 1 contains 100% Org B chunks. 0 cross-tenant batch mixing.
5. **Vector Geometry:** Strictly 384 dimensions for all 113 vectors.
6. **Numerical Validity:** 100% of vector elements are finite Python floats (no `NaN`, no `Inf`).
7. **Mathematical Normalization:** Vector $L_2$ norms satisfy $\|v\|_2 = 1.0 \pm 10^{-4}$ across all 113 vectors.

---

## 5. Semantic Relative-Ranking Telemetry

Semantic validation (`T5-04`) evaluated cosine similarity on unit-normalized vectors:
- **Query:** `"How do I request a refund for an invoice?"`
- **Related Target:** `"Customer refund and reimbursement policy guide and submission steps"`
- **Unrelated Target:** `"PostgreSQL database write-ahead log archiving and checkpoint configuration"`

### Results:
- $\text{sim}(\text{query}, \text{target}_{\text{related}}) = 0.5575$
- $\text{sim}(\text{query}, \text{target}_{\text{unrelated}}) = 0.0392$
- **Semantic Margin ($\Delta$):** $+0.5183$
- **Verification:** $\text{sim}_{\text{rel}} > \text{sim}_{\text{unrel}}$ holds by a decisive margin of $>0.50$.

---

## 6. Multi-Tenant Security & Privacy Controls

1. **Zero External Data Egress:** All model weights (`sentence-transformers/all-MiniLM-L6-v2`) reside in local cache (`~/.cache/huggingface/hub`). Inference runs in-process on CPU without network calls.
2. **Tenant-Isolated Batching:** Batches passed to `SentenceTransformer.encode()` never co-mingle distinct `org_id` values.
3. **Structured Logging Privacy:** Warnings record only sequence lengths, batch indices, and categories. Customer text, document titles, and queries are never logged.
4. **Offline Test Isolation:** Real model integration tests check local weight availability before execution and load with `local_files_only=True`, preventing automated network calls during CI or offline test runs.

---

## 7. Authoritative Test Matrix

The M5 test suite in `tests/test_m5_real_inference.py` contains 9 test cases:

| Test ID | Method | Description | Result |
|---|---|---|---|
| **T5-01** | `test_real_client_lazy_loading` | Model lifecycle, lazy loading, explicit unload, and implicit reload. | PASS |
| **T5-02** | `test_real_client_dimensions` | Vector geometry (384-d) across 1, 3, and 10 inputs. | PASS |
| **T5-03** | `test_real_client_unit_normalization` | Mathematical $L_2$ unit normalization within $10^{-4}$ tolerance. | PASS |
| **T5-04** | `test_real_semantic_relative_ranking` | Relative ranking ($\text{sim}_{\text{rel}} > \text{sim}_{\text{unrel}}$) and margin telemetry. | PASS |
| **T5-05** | `test_real_query_document_consistency` | Bitwise equality between `client.embed_text` and `pipeline.embed_query`. | PASS |
| **T5-06** | `test_real_batching_consistency` | Batch vs individual embedding consistency within float tolerance. | PASS |
| **T5-07** | `test_tokenizer_truncation_warning_and_privacy` | Bulk truncation warning emission and zero content leakage. | PASS |
| **T5-08** | `test_real_model_error_handling` | Empty/whitespace strings, invalid types, oversized queries, and missing weights. | PASS |
| **T5-09** | `test_stage0_corpus_real_inference_benchmark` | Full-corpus benchmark across 113 chunks with 7 correctness gates. | PASS |

**Total Repository Test Suite:** 79 passing tests (13 M1 + 23 M3 + 34 M4 + 9 M5) in 7.75s.

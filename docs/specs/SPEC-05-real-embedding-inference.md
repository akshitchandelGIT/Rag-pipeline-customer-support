# Specification SPEC-05: Real Embedding Inference Architecture

**Milestone:** M5 — Real Embedding Inference  
**Author:** Akshit Chandel (RAG & Knowledge Base Track)  
**Status:** Staged Specification — Aligned with Approved Plan (Implementation Gated)  
**Governing Plan:** Approved Milestone M5 Implementation Plan ([05-real-embedding-inference-plan.md](../plans/05-real-embedding-inference-plan.md))  
**Governing Team Contract:** Shared Team Contract v3.0 Locked ([CONTRACTS.md](../CONTRACTS.md) §5) & RFC-01 ([RFC-01-embedding-dimension-amendment.md](../contracts/RFC-01-embedding-dimension-amendment.md))  
**Target Model:** `sentence-transformers/all-MiniLM-L6-v2`  
**Target Module:** `src/rag/embeddings/` (introducing `SentenceTransformerEmbeddingClient` in `client.py`)  
**Target Test Suite:** `tests/test_m5_real_inference.py`  
**Date:** 2026-11-27  

---

## 1. Purpose, Scope, and System Context

### 1.1 Purpose
This technical specification formalizes the engineering requirements, data structures, model lifecycle management, tokenizer truncation protocols, dependency specifications, and testing strategies for integrating **real model inference** into the RAG Knowledge Base.

Milestone M4 established the model-independent client interface (`BaseEmbeddingClient`), pipeline orchestration with strict positional reassembly (`EmbeddingPipeline`), and a standard-library offline mock client (`DeterministicMockEmbeddingClient`). Milestone M5 replaces the synthetic mock projections with actual pre-trained neural representations produced by `sentence-transformers/all-MiniLM-L6-v2` on CPU, producing 384-dimensional unit-normalized vectors directly consumable by PostgreSQL `VECTOR(384)` storage.

### 1.2 In-Scope
1. **Concrete Client Implementation:** Specification of `SentenceTransformerEmbeddingClient` implementing `BaseEmbeddingClient` in `src/rag/embeddings/client.py`.
2. **CPU Execution Optimization:** Stateless CPU inference using `torch.inference_mode()` with `model.eval()`, standard IEEE 754 single precision (`float32`), and controlled thread allocation.
3. **Model Lifecycle & Memory Management:** Lazy model loading, process-level caching, optional explicit model unloading (`unload_model()`), and standard Hugging Face cache compliance (`$HF_HOME`).
4. **Deterministic Tokenizer Truncation & Validation Protocol:**
   - Ground-truth token count inspection using the loaded model's actual WordPiece tokenizer (`self._model.tokenizer`).
   - Permitted truncation for bulk chunk ingestion (`truncation=True`) to prevent ingestion crashes.
   - Structured, content-free logging warnings on token overflow (`actual_tokens > max_seq_length`).
   - Strict rejection of oversized runtime search queries via `EmbeddingValidationError` in `EmbeddingPipeline.embed_query()` using `client.count_tokens()`.
5. **Mathematical Precision & Normalization:** $L_2$ unit normalization ($\|v\|_2 = 1.0 \pm 10^{-4}$) and strict dimension validation (length 384, finite floats).
6. **Zero-Download Test Network Isolation:** Automated test suite isolation ensuring tests never automatically initiate network downloads or fail when offline.
7. **Relative-Ranking Semantic Validation:** Semantic verification protocol asserting that semantically related query–document pairs rank strictly higher than unrelated baseline pairs, logging numerical margins as observational telemetry.
8. **Observational Benchmark Telemetry:** Telemetry collection (latency, throughput, RSS memory) during full-corpus verification (113 chunks) without brittle latency thresholds.
9. **Bounded Runtime Dependencies:** Exact dependency bounds and CPU-only PyTorch wheel specification for Python 3.10 Linux.

### 1.3 Out-of-Scope (Strict Execution Boundaries)
- **No Implementation Code:** Writing production code in `src/` or `tests/` is deferred until this specification is approved.
- **No Dependency Installation:** No packages installed in `.venv` during specification review.
- **No Weight Downloads:** No model weights fetched from Hugging Face Hub.
- **No Database Operations:** PostgreSQL database migrations and pgvector table creation remain in Milestone M5 downstream storage / backend track (Tarun Saini).
- **No Retrieval Execution:** Cosine distance SQL queries, hybrid search, and ranking remain in Milestone M6.
- **No Breaking Changes:** Zero alterations to `BaseEmbeddingClient`, `EmbeddingPipeline`, `EmbeddedChunk`, or existing 70 unit tests.

---

## 2. Mathematical Vector Constraints and Database Contract

### 2.1 Dimensionality and Geometric Coordinates
The output embedding vector $\mathbf{v}$ for any text string $T$ must conform to:
$$\mathbf{v} \in \mathbb{R}^{384}, \quad \text{dim}(\mathbf{v}) = 384$$
Every component $v_i$ must be a finite IEEE 754 32-bit float:
$$\forall i \in \{0, 1, \dots, 383\}, \quad v_i \in \mathbb{R} \land \text{isfinite}(v_i) \land \neg\text{isnan}(v_i)$$

### 2.2 $L_2$ Unit Normalization
All generated vectors must be normalized to unit length on the 383-dimensional hypersphere $\mathbb{S}^{383}$:
$$\|\mathbf{v}\|_2 = \sqrt{\sum_{i=0}^{383} v_i^2} = 1.0 \pm 10^{-4}$$

### 2.3 Dot Product Equivalence to Cosine Similarity
Because all document vectors $\mathbf{d}$ and query vectors $\mathbf{q}$ are strictly unit-normalized, the inner product directly computes the cosine similarity:
$$\cos(\theta) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2} = \mathbf{q} \cdot \mathbf{d} = \sum_{i=0}^{383} q_i d_i$$
In PostgreSQL `pgvector`, this allows using the inner product operator (`<#>`) or cosine distance operator (`<=>`):
$$\text{cosine\_distance}(\mathbf{q}, \mathbf{d}) = 1.0 - (\mathbf{q} \cdot \mathbf{d})$$

### 2.4 Database DDL Schema Alignment
Vectors produced by `SentenceTransformerEmbeddingClient` directly align with the locked shared team contract:
```sql
-- Shared Team Contract v3.0 Locked (docs/CONTRACTS.md §5)
CREATE TABLE knowledge_base_articles (
    id UUID PRIMARY KEY,
    org_id UUID NOT NULL REFERENCES organizations(id),
    document_id UUID NOT NULL,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    access_level VARCHAR(32) NOT NULL, -- 'public' | 'internal_staff' | 'admin_only'
    embedding VECTOR(384),            -- all-MiniLM-L6-v2
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_kb_lookup ON knowledge_base_articles (org_id, access_level);
```

---

## 3. Component Specification & Interfaces

### 3.1 Class Architecture & Module Placement
The concrete client will reside in `src/rag/embeddings/client.py` and be exported through `src/rag/embeddings/__init__.py`:

```
src/rag/embeddings/
├── __init__.py           # Exports SentenceTransformerEmbeddingClient
├── models.py             # Unchanged (EmbeddingConfig, EmbeddedChunk, exceptions)
├── client.py             # Implements BaseEmbeddingClient.count_tokens() & SentenceTransformerEmbeddingClient
└── pipeline.py           # Uses client.count_tokens() for exact query sequence validation
```

### 3.2 `BaseEmbeddingClient` Protocol Extension
To provide exact token measurement without breaking existing test doubles:
```python
class BaseEmbeddingClient(ABC):
    """Abstract base class defining the model-independent embedding client interface."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[Tuple[float, ...]]:
        pass

    def embed_text(self, text: str) -> Tuple[float, ...]:
        if not isinstance(text, str):
            raise EmbeddingValidationError(f"text must be a string, got {type(text).__name__}")
        if not text.strip():
            raise EmbeddingValidationError("Input text must be a non-empty, non-whitespace string")
        return self.embed_texts([text])[0]

    @property
    @abstractmethod
    def dimensions(self) -> int:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    def count_tokens(self, text: str) -> int:
        """Count tokens in text. Subclasses with actual tokenizers override this method.
        
        Concrete default fallback ensures backward compatibility with mock test doubles.
        """
        return math.ceil(len(text.split()) * 1.33)
```

### 3.3 `SentenceTransformerEmbeddingClient` Specification
```python
class SentenceTransformerEmbeddingClient(BaseEmbeddingClient):
    """Local, in-process embedding client using sentence-transformers on CPU.
    
    Implements BaseEmbeddingClient protocol for sentence-transformers/all-MiniLM-L6-v2,
    producing 384-dimensional unit-normalized float tuples.
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        local_files_only: bool = False,
    ) -> None:
        """Initialize the client configuration without loading heavy weights.
        
        Args:
            config: Optional EmbeddingConfig. Defaults to standard EmbeddingConfig(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                expected_dimensions=384,
                batch_size=64,
                max_seq_length=256,
                device="cpu",
                normalize_embeddings=True,
            ).
            local_files_only: If True, restricts model loading to local disk cache only,
                prohibiting any network access.
                
        Raises:
            EmbeddingConfigError: If config parameters are invalid or incompatible.
        """
        ...

    def load_model(self) -> None:
        """Explicitly load model weights and tokenizer into memory if not already loaded.
        
        Raises:
            EmbeddingInferenceError: If weights cannot be loaded due to missing files
                or corrupt cache.
        """
        ...

    def unload_model(self) -> None:
        """Release in-memory model weights and trigger garbage collection."""
        ...

    @property
    def is_loaded(self) -> bool:
        """Return True if model weights are currently resident in memory."""
        ...

    @property
    def dimensions(self) -> int:
        """Output dimensionality of generated vectors (strictly 384)."""
        return 384

    @property
    def model_name(self) -> str:
        """Canonical identifier of the underlying model."""
        return self._config.model_name

    def count_tokens(self, text: str) -> int:
        """Return exact WordPiece token count including [CLS] and [SEP] special tokens.
        
        Args:
            text: Input string to tokenize.
            
        Returns:
            Integer total token count including special tokens.
        """
        self.load_model()
        encoded = self._model.tokenizer(text, truncation=False, add_special_tokens=True)
        return len(encoded["input_ids"])

    def embed_texts(self, texts: List[str]) -> List[Tuple[float, ...]]:
        """Generate embedding vectors for a batch of text strings.
        
        Execution Protocol:
        1. Validate that texts is a non-empty list of non-empty, non-whitespace strings.
        2. Ensure model is loaded (lazy initialization).
        3. Inspect actual WordPiece token counts via self._model.tokenizer(texts, truncation=False, add_special_tokens=True).
        4. For any item where len(input_ids) > self._config.max_seq_length (256):
           Emit structured warning: category="token_truncation_warning".
        5. Execute inference under torch.inference_mode() with model.eval().
        6. Encode texts with batch_size=self._config.batch_size, normalize_embeddings=True,
           convert_to_numpy=True.
        7. Validate that returned array shape is (len(texts), 384) and all values are finite.
        8. Convert rows to immutable Tuple[float, ...] instances.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            List of 384-dimensional unit-normalized float tuples matching input length and order.
            
        Raises:
            EmbeddingValidationError: If texts is malformed, contains empty strings, or output is non-finite.
            EmbeddingInferenceError: On runtime model crash or memory allocation failure.
        """
        ...

    def embed_text(self, text: str) -> Tuple[float, ...]:
        """Convenience method: generate an embedding vector for a single text string.
        
        Enforces strict sequence limit validation: if count_tokens(text) > max_seq_length,
        raises EmbeddingValidationError.
        """
        ...
```

---

## 4. Tokenizer Integration: Truncation, Limits & Communication

### 4.1 Separation: M3 Heuristic vs M5 Tokenizer
- **M3 Chunking Subsystem:** Uses `estimate_tokens(text: str) -> int` (`ceil(len(text.split()) * 1.33)`) as an upstream structural text-budgeting heuristic in `src/rag/chunking/models.py`. It is completely untouched.
- **M5 Embedding Subsystem:** Uses the actual BERT WordPiece tokenizer (`bert-base-uncased`, 30,522 vocabulary) loaded by `sentence-transformers/all-MiniLM-L6-v2`. It is used exclusively at inference time for ground-truth token counting, truncation detection, and query length validation.

### 4.2 Special Token Accounting & Model Input Limits
- `all-MiniLM-L6-v2` has an architectural maximum sequence limit of **256 tokens**.
- MiniLM's tokenizer automatically prepends `[CLS]` (token ID 101) and appends `[SEP]` (token ID 102) when `add_special_tokens=True`.
- Consequently, the usable raw text token capacity is:
  $$\text{Capacity}_{\text{text}} = \text{max\_seq\_length} - 2 = 256 - 2 = 254 \text{ WordPiece tokens}$$
- Any token counting or sequence validation must account for special tokens (`add_special_tokens=True`).

### 4.3 Bulk-Document Ingestion Truncation Policy
For bulk chunk ingestion (`embed_chunks` / `embed_texts`):
1. **Truncation Permitted:** `truncation=True` is enabled in `model.encode()` so that an outlier or oversized document does not crash an ingestion batch.
2. **Ground-Truth Tokenizer Inspection:**
   Prior to vector generation, the client checks exact token lengths using the model's actual tokenizer:
   ```python
   tokenized_batch = self._model.tokenizer(texts, truncation=False, add_special_tokens=True)
   for idx, input_ids in enumerate(tokenized_batch["input_ids"]):
       actual_tokens = len(input_ids)
       if actual_tokens > self._config.max_seq_length:
           logger.warning(
               "Input sequence exceeded model maximum sequence length; truncated by tokenizer.",
               extra={
                   "category": "token_truncation_warning",
                   "actual_tokens": int(actual_tokens),
                   "max_seq_length": self._config.max_seq_length,
                   "batch_index": idx,
               }
           )
   ```
3. **Structured Content-Free Warning Schema:**
   The warning logs strictly structured metadata (`category="token_truncation_warning"`, `actual_tokens`, `max_seq_length`, `batch_index`).
   **Security Invariant:** Zero raw chunk text, document titles, or query tokens are emitted into logs.

### 4.4 Runtime Search Query Rejection Policy
For interactive user queries (`EmbeddingPipeline.embed_query`):
1. Support queries are inherently concise; interactive queries must not be silently truncated.
2. In `EmbeddingPipeline.embed_query(query: str)`:
   - Validates that `query` is a non-empty string.
   - Evaluates exact tokenizer length via `self._client.count_tokens(stripped)`.
   - If `actual_tokens > self._config.max_seq_length` (256):
     ```python
     raise EmbeddingValidationError(
         f"Search query exceeds maximum sequence token limit of {self._config.max_seq_length} "
         f"(actual tokenizer count: {actual_tokens} tokens, including special tokens)"
     )
     ```
3. In `SentenceTransformerEmbeddingClient.embed_text(text: str)`, the same check is enforced, ensuring queries cannot be silently truncated even if invoked directly on the client.

---

## 5. Model Lifecycle, Caching & Offline Network Isolation

### 5.1 First-Run Download and Local Caching
1. When initialized with internet access, `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")` downloads the ~90MB weights snapshot to `$HF_HOME/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/<commit_hash>/`.
2. Model snapshot includes:
   - `pytorch_model.bin` (or `model.safetensors`)
   - `config.json`
   - `tokenizer.json`, `vocab.txt`, `tokenizer_config.json`
   - `modules.json`, `1_Pooling/config.json`

### 5.2 Offline Execution Enforcement
1. In production and offline environments, `local_files_only=True` or `HF_HUB_OFFLINE=1` is configured.
2. `SentenceTransformer` loads directly from the local snapshot without issuing DNS requests or HTTP socket calls.
3. If weights are not found locally under `local_files_only=True`, the client catches `(OSError, ValueError)` and raises:
   ```python
   raise EmbeddingInferenceError(
       f"Model weights for '{self._config.model_name}' are not available locally. "
       "Pre-stage weights in Hugging Face cache or run with network access to initialize."
   )
   ```

### 5.3 Automated Test Suite Isolation Guarantee
> [!CRITICAL]
> **NETWORK ISOLATION TEST INVARIANT**  
> Running `python3 -m unittest` must NEVER make external HTTP requests or trigger automatic model downloads. Tests must be deterministic, sandboxed, and safe.

In `tests/test_m5_real_inference.py`:
1. The test harness detects whether `sentence_transformers` is installed AND whether `all-MiniLM-L6-v2` is present in local cache:
   ```python
   def _is_model_cached(model_name: str) -> bool:
       try:
           from sentence_transformers import SentenceTransformer
           SentenceTransformer(model_name, local_files_only=True)
           return True
       except Exception:
           return False
   ```
2. Real-model integration tests are guarded:
   ```python
   @unittest.skipUnless(
       HAS_SENTENCE_TRANSFORMERS and IS_MODEL_CACHED,
       "sentence-transformers not installed or model weights not cached locally; "
       "skipping real-model integration tests to prevent automatic network downloads."
   )
   class TestM5RealInference(unittest.TestCase):
       ...
   ```
3. Existing unit tests in `tests/test_m4_embeddings.py` (34 tests) execute 100% offline at all times using `DeterministicMockEmbeddingClient`.

---

## 6. Dependency Bounds and CPU-Only Packaging

### 6.1 Environment Verification
- **Target OS:** Linux x86_64
- **Python Version:** 3.10.12 (Active in `.venv`)
- **Package Isolation:** Standard `pip install torch` downloads ~800MB–1.2GB of NVIDIA CUDA runtime binaries. To ensure lightweight CPU execution, CPU-specific PyTorch wheels must be targeted.

### 6.2 Bounded Dependency Specification (`requirements.txt`)
Create `requirements.txt`:
```text
# Milestone M5 Real Embedding Inference — Bounded Runtime Dependencies
sentence-transformers>=3.0.0,<3.4.0
transformers>=4.40.0,<4.48.0
torch>=2.2.0,<2.6.0
```

### 6.3 Exact Installation Command Protocol
The verified installation command sequence for `.venv`:
```bash
# Step 1: Install lightweight CPU-only PyTorch (~180MB) from official PyTorch wheel repository
.venv/bin/pip install "torch>=2.2.0,<2.6.0" --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install bounded sentence-transformers from PyPI
.venv/bin/pip install -r requirements.txt
```

---

## 7. Authoritative Test Matrix & Benchmark Protocol

### 7.1 Real-Model Integration Test Suite (`tests/test_m5_real_inference.py`)

| Test ID | Test Method | Target Component | Verification Protocol & Assertions |
|---|---|---|---|
| **T5-01** | `test_real_client_lazy_loading` | Model Lifecycle | Asserts `client.is_loaded is False` upon instantiation; asserts `client.is_loaded is True` after first `embed_text()` or explicit `load_model()`. Asserts `unload_model()` resets `is_loaded to False`. |
| **T5-02** | `test_real_client_dimensions` | Vector Geometry | Passes 1 text, 3 texts, and 10 texts; asserts every output vector has `len(vec) == 384` and `client.dimensions == 384`. |
| **T5-03** | `test_real_client_unit_normalization` | Math Validation | Computes $L_2 = \sqrt{\sum x_i^2}$ across 10 distinct texts; asserts $L_2 == 1.0 \pm 10^{-4}$ for every vector. |
| **T5-04** | `test_real_semantic_relative_ranking` | Semantic Validity | **Relative-Ranking Verification:**<br>1. Query: `"How do I request a refund for an invoice?"`<br>2. Related Target: `"Customer refund and reimbursement policy guide and submission steps"`<br>3. Unrelated Target: `"PostgreSQL database write-ahead log archiving and checkpoint configuration"`<br>Asserts: $\text{sim}(\text{query}, \text{target}_{\text{related}}) > \text{sim}(\text{query}, \text{target}_{\text{unrelated}})$.<br>Records similarity scores and margin as test telemetry. |
| **T5-05** | `test_real_query_document_consistency` | Pipeline Adherence | Verifies `client.embed_text(query)` and `pipeline.embed_query(query)` produce bitwise identical vectors. |
| **T5-06** | `test_real_batching_consistency` | Inference Consistency | Verifies encoding a batch of 5 texts simultaneously yields bitwise identical vectors to encoding each text individually. |
| **T5-07** | `test_tokenizer_truncation_warning_and_privacy` | Truncation & Privacy | Feeds a synthetic string of 350 words (>400 actual tokens); verifies `logger.warning` is emitted with `category="token_truncation_warning"`; asserts zero raw text or tokens in log records. |
| **T5-08** | `test_real_model_error_handling` | Robustness | Asserts empty string, whitespace string, non-string, and oversized query inputs raise `EmbeddingValidationError`. |
| **T5-09** | `test_stage0_corpus_real_inference_benchmark` | Full-Corpus Benchmark | Runs real inference across all 113 raw corpus chunks from `data/raw/`. Enforces correctness gates and records observational telemetry. |

### 7.2 Full-Corpus Benchmark Correctness Gates vs Observational Telemetry (`T5-09`)
- **Corpus Input:** All 10 documents from `data/raw/` (56 Org A, 57 Org B = 113 chunks).
- **Hard Correctness Invariant Gates (Pass/Fail):**
  1. Output count exactly matches input count (113 `EmbeddedChunk` instances).
  2. 100% strict positional order preservation ($0 \dots 112$).
  3. 100% metadata fidelity (`id`, `chunk_id`, `document_id`, `org_id`, `access_level`, `title`, `content`).
  4. 100% tenant batch isolation (0 batches combine Org A and Org B texts).
  5. Strictly 384 dimensions for all 113 vectors.
  6. All vector elements are finite floats.
  7. Mathematical $L_2$ unit norm ($1.0 \pm 10^{-4}$).
- **Observational Telemetry (Logged, Non-Failing):**
  - Total elapsed wall-clock time (seconds).
  - Average per-chunk latency (ms/chunk).
  - Effective throughput (chunks/second).
  - Peak resident memory (RSS MB).
  - Telemetry is recorded in `docs/design/05-real-embedding-inference.md`.

---

## 8. Multi-Tenant Security & Privacy Invariants

| Security Domain | Architectural Invariant | Implementation Mechanism |
|---|---|---|
| **Zero Data Egress** | Customer support articles and queries must never leave host memory. | 100% local in-process inference on CPU using local weights. No external API calls. |
| **Tenant Batch Isolation** | Chunks from different tenants must never be co-mingled in a single inference call. | `EmbeddingPipeline` groups chunks by `org_id` before slicing into batches. |
| **Logging Privacy** | Structured logs must never expose customer PII, raw content, or query text. | Log records capture only item counts, batch indices, token counts, and error types. |
| **Pre-Retrieval SQL Filtering** | Embedding vectors provide semantic similarity only; they must never be relied upon for tenant security. | Mandatory SQL prefilter in PostgreSQL: `WHERE org_id = $1 AND access_level = ANY($2)` before similarity calculation. |
| **Untrusted Content** | Document chunks are passive text; prompt injections must not alter system context. | Chunks are passed strictly as string data for tensor encoding; never evaluated as code or instructions. |

---

## 9. Acceptance Criteria for Milestone M5

1. **Clean Baseline Preserved:** All 70 existing unit tests pass cleanly without regression.
2. **Contract Adherence:** `SentenceTransformerEmbeddingClient` implements `BaseEmbeddingClient` with `dimensions == 384`, `model_name == "sentence-transformers/all-MiniLM-L6-v2"`, and `count_tokens()`.
3. **Pluggable Architecture:** `EmbeddingPipeline` functions identically with both `DeterministicMockEmbeddingClient` and `SentenceTransformerEmbeddingClient`.
4. **Tokenizer Truncation Protocol:** Permitted truncation for bulk ingestion; structured content-free warning emitted on overflow; strict rejection of oversized queries using exact tokenizer counts.
5. **Numerical Integrity:** Output vectors are 384-dimensional finite float tuples with $\|v\|_2 = 1.0 \pm 10^{-4}$.
6. **Relative Semantic Ranking:** Related query–target pairs rank strictly higher than unrelated pairs; margin is recorded as telemetry.
7. **Full-Corpus Correctness:** All 113 raw corpus chunks processed with 100% metadata fidelity, strict tenant batch isolation, and observational throughput logged.
8. **Test Network Isolation:** Automated test suite runs safely without initiating network downloads when offline.
9. **Documentation Delivered:** `docs/design/05-real-embedding-inference.md`, `docs/STATUS.md`, and `docs/CHANGELOG.md` updated with empirical telemetry and milestone closeout.

---

## 10. File Change Matrix

| Action | File Path | Scope & Purpose |
|---|---|---|
| **Create** | `docs/specs/SPEC-05-real-embedding-inference.md` | Authoritative specification document (This file). |
| **Create** | `requirements.txt` | Bounded runtime dependency manifest. |
| **Modify** | `src/rag/embeddings/client.py` | Implement `SentenceTransformerEmbeddingClient` with lazy loading, `count_tokens()`, and tokenizer truncation checks. |
| **Modify** | `src/rag/embeddings/__init__.py` | Export `SentenceTransformerEmbeddingClient`. |
| **Modify** | `src/rag/embeddings/pipeline.py` | Update `embed_query` to use `client.count_tokens()` for exact tokenizer sequence validation. |
| **Create** | `tests/test_m5_real_inference.py` | Real-model integration test suite (`T5-01` through `T5-09`). |
| **Create** | `docs/design/05-real-embedding-inference.md` | Design record and benchmark telemetry report. |
| **Modify** | `docs/STATUS.md` | Record M5 progress and empirical metrics. |
| **Modify** | `docs/CHANGELOG.md` | Record M5 deliverable entries. |
| **Modify** | `docs/README.md` | Update documentation index links. |

---

*Specification SPEC-05 aligned with approved plan. Awaiting review and approval before proceeding to Stage 3 (Dependency Staging) and Stage 4 (Implementation).*

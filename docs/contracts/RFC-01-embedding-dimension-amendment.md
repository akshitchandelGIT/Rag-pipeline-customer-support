# RFC-01: Embedding Model & Vector Dimensionality Amendment (v3.0 -> v3.1)

```
RFC Identifier: RFC-01-embedding-dimension-amendment
Title:          Transition from 1536-Dimensional Remote Embeddings to 384-Dimensional Local Embeddings
Author:         Akshit Chandel (RAG & Knowledge Base Track)
Target System:  Multi-Tenant B2B Customer Support AI — Knowledge Base Pipeline
Date Created:   2026-11-26
Status:         AGREED IN TEAM DECISION — PENDING FORMAL CONTRACT COMMIT
Governing Contract: Shared Team Contract v3.0 Locked
Proposed Contract:  Shared Team Contract v3.1 (Draft)
Impacted Tracks:    Backend/Database (Tarun Saini), LangGraph Orchestration (Shashank Pathania),
                    MCP Tools (Pulkit Singla), RAG Knowledge Base (Akshit Chandel)
```

> [!IMPORTANT]
> **GOVERNANCE NOTICE: AGREED IN TEAM DECISION (PENDING FORMAL CONTRACT COMMIT)**  
> Cross-functional team consensus has been communicated adopting `all-MiniLM-L6-v2` with 384 dimensions (`embedding VECTOR(384), -- all-MiniLM-L6-v2`).  
> In accordance with `AGENTS.md` ("Do not change shared contract behavior without documenting the discrepancy and asking the user"), this document records the agreement while noting that formal repository commit to the locked shared contract files (`docs/PROJECT_CONTEXT.md` / `CONTRACTS.md`) remains pending action by the contract owner. No live database migrations or package installations are executed until explicitly authorized.

---

## 1. Executive Summary

This RFC proposes amending the Shared Team Contract (advancing from v3.0 Locked to v3.1) regarding the canonical text embedding model and vector dimensionality for the Multi-Tenant B2B Customer Support AI knowledge base.

- **Current Contract v3.0 Specification:** Remote cloud API embedding model `text-embedding-3-small` producing **1536-dimensional** vectors, targeting PostgreSQL table `knowledge_base_articles` with column `embedding vector(1536)`.
- **Proposed Amendment (v3.1):** Self-hosted open-weights local embedding model `sentence-transformers/all-MiniLM-L6-v2` producing **384-dimensional** unit-normalized vectors, updating target database column to `embedding vector(384)`.
- **Key Motivating Drivers:**
  1. **Cost Elimination:** Completely eliminates recurring third-party API token costs ($0.00 ingestion and query vectorization charges).
  2. **Zero Network Latency & Data Privacy:** Embeddings are generated entirely in-process on CPU/GPU without external network calls, eliminating egress data transmission of enterprise support documents and queries.
  3. **75% Resource Reduction in pgvector:** 384-dimensional vectors reduce pgvector storage, memory footprints, and HNSW/IVFFlat index cache requirements by 75% (1.5 KB per chunk vs 6.0 KB per chunk).
- **Core Governance Requirement:** Strict alignment across database schema (`vector(384)`), ingestion pipeline (`EmbeddedChunk.dimensions == 384`), and runtime search orchestrator (`embed_query(query) -> 384-d vector`).

---

## 2. Problem Statement

Shared Team Contract v3.0 was locked with an initial architectural assumption that OpenAI's `text-embedding-3-small` (1536 dimensions) would serve as the unified embedding model across ingestion and retrieval.

However, subsequent architectural review and local development identified several trade-offs:
1. **Recurring API Expenditure & Rate Limits:** Cloud-based embedding APIs require persistent API credentials, credit consumption, and exponential backoff retry mechanisms to handle HTTP 429 rate limits during bulk document ingestion.
2. **Data Residency & Egress:** Sending internal and tenant-confidential customer support articles (including internal staff troubleshooting guides) to an external cloud provider introduces operational compliance overhead.
3. **Hardware & Index Overhead:** In pgvector, indexing 1536-dimensional vectors consumes significant RAM for HNSW graphs (`M=16, efConstruction=64`). For typical B2B support document chunks (mean size ~90 tokens), a 1536-dimensional vector represents unnecessary dimensional overhead.
4. **Schema Constraint Failure:** Milestone M4 currently generates 384-dimensional vectors in its offline test harness. If downstream Milestone M5 (Database Storage & Migrations) provisions `knowledge_base_articles.embedding` as `vector(1536)`, inserting 384-dimensional vectors will fail immediately with a fatal PostgreSQL error:
   ```
   ERROR: different vector dimensions 384 and 1536
   ```

To resolve this conflict safely and collaboratively, the project requires an explicit team decision before Milestone M5 database migrations are executed.

---

## 3. Current Contract v3.0 Specification

The currently locked Shared Team Contract v3.0 specifies:

| Dimension / Aspect        | Contract v3.0 Locked Specification                                       | Source Document                            |
| :------------------------ | :----------------------------------------------------------------------- | :----------------------------------------- |
| **Model Identifier**      | `text-embedding-3-small`                                                 | `docs/PROJECT_CONTEXT.md` §Shared contract |
| **Vector Dimensionality** | **1536 dimensions**                                                     | `docs/PROJECT_CONTEXT.md` §Shared contract |
| **Execution Environment** | Remote Cloud API (OpenAI REST endpoint)                                  | `docs/plans/04-embeddings-plan.md`         |
| **Database Table**        | `knowledge_base_articles`                                                | `docs/PROJECT_CONTEXT.md` §Shared contract |
| **Vector Column Type**    | `embedding vector(1536)`                                                | Shared Contract v3.0 DDL                   |
| **Distance Metric**       | Cosine distance (`<=>`) / Dot product (`<#>`)                            | Shared Contract v3.0                       |
| **Normalization**         | Unit length ($L_2 = 1.0$)                                                | Shared Contract v3.0                       |
| **Security Invariant**    | Pre-retrieval SQL filter: `WHERE org_id = $1 AND access_level = ANY($2)` | `AGENTS.md`, `PROJECT_CONTEXT.md`          |

---

## 4. Current M4 Implementation & Offline Mock Clarification

Milestone M4 has completed Phases 1, 2, 3, and 5 using a Python standard-library offline mock client:
- **Component:** `DeterministicMockEmbeddingClient` ([src/rag/embeddings/client.py](file:///home/akshit/Rag-pipeline-Customer-Support/src/rag/embeddings/client.py))
- **Configured Dimensionality:** 384 dimensions (matching `sentence-transformers/all-MiniLM-L6-v2`)
- **Pipeline:** `EmbeddingPipeline` ([src/rag/embeddings/pipeline.py](file:///home/akshit/Rag-pipeline-Customer-Support/src/rag/embeddings/pipeline.py))
- **Verification Status:** Processed all 10 corpus documents (113 chunks: 56 Org A, 57 Org B) with 100% tenant batch isolation and 1:1 order preservation.

> [!WARNING]
> **CRITICAL ARCHITECTURAL DISTINCTION: MOCK VECTORS ARE NON-SEMANTIC**  
> The 384-dimensional vectors produced by `DeterministicMockEmbeddingClient` are **pseudo-random Gaussian projections generated deterministically from SHA-256 digests**.  
> - They exist solely to verify structural data flow, tenant batch boundaries, memory safety, and indexed reassembly without requiring PyTorch or model weights.
> - **They possess ZERO semantic retrieval capability.** Cosine similarity between mock vectors reflects character-hash pseudo-randomness, NOT conceptual or textual relevance.
> - The mock client must **NEVER** be evaluated for retrieval precision, recall, or ranking quality.
> - Actual semantic retrieval evaluation can only take place once a real model client (e.g. `SentenceTransformerEmbeddingClient` or a remote API client) is approved, integrated, and evaluated against ground-truth queries.

---

## 5. Proposed Contract Amendment (v3.1)

If approved by the cross-functional team, Contract v3.0 will be amended to Contract v3.1 as follows:

| Field / Property            | Contract v3.0 (Current Locked) | Proposed Contract v3.1                   | Delta Rationale                                                  |
| :-------------------------- | :----------------------------- | :--------------------------------------- | :--------------------------------------------------------------- |
| **Primary Embedding Model** | `text-embedding-3-small`       | `sentence-transformers/all-MiniLM-L6-v2` | Transition to local self-hosted open-weights model.              |
| **Embedding Dimensions**    | **1536**                       | **384**                                  | Matches canonical output dimension of MiniLM-L6.                 |
| **Execution Environment**   | External HTTPS API call        | Local in-process CPU/GPU inference       | Zero network dependency; zero external data egress.              |
| **PostgreSQL Column**       | `embedding vector(1536)`       | `embedding vector(384)`                  | 75% reduction in vector storage and index memory.                |
| **Vector Normalization**    | Unit length ($L_2 = 1.0$)      | Unit length ($L_2 = 1.0$)                | Preserved; dot product equals cosine similarity.                 |
| **Max Sequence Length**     | 8191 tokens                    | 256 word pieces                          | Verified sufficient: 100% of corpus chunks are $\le 169$ tokens. |
| **Ingestion Charges**       | Variable per-token API cost    | **$0.00 (Zero)**                         | Fully open-weights Apache 2.0 license.                           |
| **Tenant Prefilter**        | Mandatory SQL prefilter        | Mandatory SQL prefilter                  | **Preserved exactly (zero security change).**                    |

---

## 6. Embedding Model & Dimension Configurability

To ensure our software architecture does not become brittle or permanently coupled to a single model, Milestone M4 was engineered with **strict model-agnostic interfaces**:

```
BaseEmbeddingClient (ABC)
   ├── embed_texts(texts: List[str]) -> List[Tuple[float, ...]]
   ├── embed_text(text: str) -> Tuple[float, ...]
   ├── @property dimensions -> int
   └── @property model_name -> str
```

- **Dynamic Configuration:** `EmbeddingConfig` parameterizes `model_name: str` and `expected_dimensions: int`.
- **Validation Decoupling:** `EmbeddingPipeline` checks vector dimensions dynamically against `self.client.dimensions`. If a 384-dimensional client is plugged in, the pipeline validates against 384; if 1536, it validates against 1536.
- **Pluggable Architecture:** Transitioning between local models (`all-MiniLM-L6-v2`, `bge-small-en-v1.5`) or remote APIs requires authoring a new client implementing `BaseEmbeddingClient` without rewriting chunking, batching, reassembly, or retrieval orchestration.
- **Architectural Boundary:** While the *code* is configurable, the *database schema* is fixed to a single dimension at any given time. Configurability in code does not eliminate the need for strict database column alignment.

---

## 7. Database & pgvector Storage Implications (Milestone M5)

The Backend/Database track (Tarun Saini) will implement Milestone M5 storage. The vector dimension directly impacts database DDL and resource sizing.

### 7.1 DDL Schema Comparison
```sql
-- CURRENT CONTRACT v3.0 LOCKED:
CREATE TABLE knowledge_base_articles (
    id UUID PRIMARY KEY,
    org_id UUID NOT NULL,
    document_id UUID NOT NULL,
    chunk_index INTEGER NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    access_level TEXT NOT NULL,
    embedding vector(1536) NOT NULL
);

-- PROPOSED CONTRACT v3.1 AMENDMENT:
CREATE TABLE knowledge_base_articles (
    id UUID PRIMARY KEY,
    org_id UUID NOT NULL,
    document_id UUID NOT NULL,
    chunk_index INTEGER NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    access_level TEXT NOT NULL,
    embedding vector(384) NOT NULL
);
```

### 7.2 Storage Footprint & Memory Analysis

| Metric                           | 1536 Dimensions (`text-embedding-3-small`)                | 384 Dimensions (`all-MiniLM-L6-v2`)                     | Impact / Savings                |
| :------------------------------- | :-------------------------------------------------------- | :------------------------------------------------------ | :------------------------------ |
| **Bytes per Vector**             | 1,536 floats $\times$ 4 bytes = **6,144 bytes** (~6.0 KB) | 384 floats $\times$ 4 bytes = **1,536 bytes** (~1.5 KB) | **75.0% reduction**             |
| **10,000 Chunks Raw Table Size** | ~61.4 MB (embeddings only)                                | ~15.4 MB (embeddings only)                              | **46.0 MB saved**               |
| **HNSW Index Size (est.)**       | ~1.5 $\times$ vector size $\approx$ 92 MB / 10k           | ~1.5 $\times$ vector size $\approx$ 23 MB / 10k         | **69 MB RAM saved**             |
| **Index Build Duration**         | Slower (higher dimensional distance calculations)         | ~4x faster                                              | Significantly faster migrations |
| **Exact Dimension Enforcement**  | Throws error if input $\ne 1536$                          | Throws error if input $\ne 384$                         | Hard pgvector type constraint   |

---

## 8. Query-Embedding Compatibility Across Systems

The Multi-Tenant B2B Customer Support AI relies on separate subsystems for ingestion and retrieval:
1. **Ingestion Pipeline (RAG Track):** Chunks raw articles, generates embeddings, writes to PostgreSQL `knowledge_base_articles`.
2. **Runtime Retrieval (LangGraph / Orchestrator Track — Shashank Pathania):** Accepts end-user customer support queries, vectorizes the query text, and queries PostgreSQL via pgvector distance operator.
3. **MCP Tooling (Pulkit Singla):** Exposes retrieval tools to the agent with structured query parameters.

### 8.1 The Compatibility Invariant
> [!CAUTION]
> **CROSS-SYSTEM MATHEMATICAL INVARIANT**  
> Document embeddings stored in PostgreSQL and query embeddings generated at runtime **MUST be produced by the identical model and identical dimensionality**.  
> - Computing cosine distance (`vector_a <=> vector_b`) between vectors from different models (e.g. an OpenAI query vector against a MiniLM document vector) is mathematically meaningless: the vector spaces have completely different geometric coordinate systems and orthogonal bases.
> - Computing distance between mismatched dimensions (e.g. 384 query against 1536 document) triggers a fatal SQL execution exception in PostgreSQL.
> - Therefore, whatever model is approved for ingestion **MUST be adopted simultaneously by the query vectorizer in the orchestrator**.

---

## 9. Model Changes & Re-Embedding Implications

A critical consideration for the team is understanding what happens if an embedding model is modified *after* documents have been ingested.

### 9.1 Immutability of Vector Spaces
- **No In-Place Conversion:** Embedding vectors cannot be converted, projected, or translated from one model to another without re-running inference on the raw text.
- **Total Invalidation on Model Switch:** If the team decides to switch from 384-d `all-MiniLM-L6-v2` to a 1536-d model (or another local model like BGE or E5) in the future:
  1. Every single article chunk in `knowledge_base_articles` must be re-embedded from `content`.
  2. The table column must be migrated (`ALTER TABLE knowledge_base_articles ALTER COLUMN embedding TYPE vector(new_dim)`).
  3. All existing vector indexes (`HNSW` / `IVFFlat`) must be dropped and rebuilt.
  4. The query embedding client in the LangGraph orchestrator must be updated concurrently.

### 9.2 Recommendation for Milestone Planning
Adopting `sentence-transformers/all-MiniLM-L6-v2` *before* Milestone M5 database creation avoids performing a painful database re-embedding migration later. The schema will be clean on Day 1.

---

## 10. Tenant Isolation and Security Invariants

This proposed amendment has **zero negative impact on tenant isolation and security boundaries**:

1. **SQL Prefiltering Remains Sovereign:**  
   As mandated by `AGENTS.md` and Contract v3.0, retrieval queries **must always prefilter** by tenant and authorized access level in SQL before vector similarity ranking:
   ```sql
   SELECT id, title, content, 1 - (embedding <=> $query_vector) AS similarity
   FROM knowledge_base_articles
   WHERE org_id = $auth_org_id 
     AND access_level = ANY($auth_allowed_tiers)
   ORDER BY embedding <=> $query_vector
   LIMIT $limit;
   ```
2. **Access Tiers Intact:**  
   The three access tiers (`public`, `internal_staff`, `admin_only`) and role permissions (End Customer, Support Agent, Org Admin) remain completely unchanged.
3. **Batch Isolation Preserved:**  
   The ingestion pipeline's tenant batch boundary enforcement (verified in M4 Phase 5) guarantees that chunks from different organizations are never co-mingled during vector generation.

---

## 11. Alternatives Considered

| Alternative                                               | Description                                               | Pros                                                                                                  | Cons / Reason Rejected                                                                                                      |
| :-------------------------------------------------------- | :-------------------------------------------------------- | :---------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------- |
| **Alt 1: Status Quo (`text-embedding-3-small`, 1536-d)** | Stick to remote OpenAI API.                               | High semantic performance; already contracted.                                                        | Requires paid API credits; external network egress; higher pgvector memory overhead; requires retry machinery for HTTP 429. |
| **Alt 2: Local 384-d (`all-MiniLM-L6-v2`) [RECOMMENDED]** | Self-hosted SentenceTransformer on CPU/GPU.               | Zero cost; zero egress; fast local inference; 75% smaller pgvector storage; well-tested B2B baseline. | Lower theoretical ceiling on complex multi-hop semantics than massive 1536-d models; 256-token context window.             |
| **Alt 3: Local 768-d (`bge-base-en-v1.5`)**               | Mid-size local model producing 768-d vectors.             | Higher MTEB score; larger 512 context window.                                                         | ~3x larger model weights (~438 MB vs 90 MB); slower CPU inference; intermediate memory footprint.                           |
| **Alt 4: Dual Storage / Dynamic Dimensioning**            | Store multiple embeddings or unconstrained vector column. | Flexible.                                                                                             | Severely violates simplicity; breaks Shared Contract; doubles storage; complicates query routing.                           |

---

## 12. Formal Recommendation

The RAG Knowledge Base track formally recommends:

1. **Approve RFC-01 and advance Shared Team Contract to v3.1.**
2. Adopt `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) as the primary local embedding model.
3. Target `embedding vector(384)` in Milestone M5 PostgreSQL DDL.
4. Provide the LangGraph orchestrator team with a standardized, lightweight query vectorizer adapter sharing `BaseEmbeddingClient.embed_text`.
5. Benchmark retrieval quality in Milestone M7 against synthetic support queries; if retrieval quality falls below acceptable thresholds, evaluate upgrading to a 768-d or 1536-d model prior to production release.

---

## 13. Contract Statements Requiring Amendment

If this RFC is approved, the following specific passages in Shared Team Contract v3.0 will be modified:

### 13.1 `docs/PROJECT_CONTEXT.md`
- **Current Text (§Shared contract):**  
  > "The contract document specifies 1536-dimensional embeddings. Confirm this requirement before model selection or schema creation."
- **Proposed Amendment (v3.1):**  
  > "Under Shared Team Contract v3.1, the canonical embedding model is `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional unit-normalized vectors. Target database table `knowledge_base_articles` defines column `embedding vector(384)`."

### 13.2 Target Database DDL Contract
- **Current DDL Column:** `embedding vector(1536) NOT NULL`
- **Proposed DDL Column:** `embedding vector(384) NOT NULL`

---

## 14. Team Approval Checklist

Track leads review and record status:

- [x] **Cross-Team Consensus Communicated:**  
  *Team decision received agreeing to `all-MiniLM-L6-v2` with 384 dimensions (`embedding VECTOR(384), -- all-MiniLM-L6-v2`).*  
  *Status:* `AGREED IN PRINCIPLE BY TEAM`  
  *Date:* `2026-11-26`

- [ ] **Contract Owner Repository Commit (Tarun Saini / Lead):**  
  *Formal commit to shared contract files (`docs/PROJECT_CONTEXT.md` / `CONTRACTS.md`) reflecting Contract v3.1.*  
  *Status:* `PENDING CONTRACT COMMIT`  
  *Date:* `________`

- [x] **RAG Knowledge Base Lead (Akshit Chandel):**  
  *Author and implementation owner for M4 local embeddings subsystem (384-d pipeline verified).*  
  *Status:* `RECONCILED & COMPLIANT`  
  *Date:* `2026-11-26`

- [ ] **Backend / Database Lead (Tarun Saini):**  
  *Approval of `vector(384)` column definition for `knowledge_base_articles` in Milestone M5 migrations.*  
  *Status:* `AGREED VIA TEAM DECISION — M5 STAGING PENDING`  
  *Date:* `2026-11-26`

- [ ] **LangGraph Orchestrator Lead (Shashank Pathania):**  
  *Approval of 384-dimensional query vectorization interface (`embed_query`) for runtime retrieval.*  
  *Status:* `AGREED VIA TEAM DECISION — M6 INTEGRATION PENDING`  
  *Date:* `2026-11-26`

- [ ] **MCP Tools Lead (Pulkit Singla):**  
  *Acknowledgment of 384-dimensional vector retrieval pipeline for tool invocation.*  
  *Status:* `AGREED VIA TEAM DECISION`  
  *Date:* `2026-11-26`

---

## 15. Concrete Questions for Cross-Track Decision

The team must formally resolve the following questions:

1. **Inference Environment:** Does the deployment environment (Docker / server / CI) have sufficient CPU/memory to run local PyTorch/sentence-transformers inference (~90MB model, ~200MB RAM), or is there an environment constraint that necessitates remote HTTP inference?
2. **Dependency Approval:** Does the team approve adding `sentence-transformers` (and its underlying PyTorch runtime) to `requirements.txt` for local inference in Phase 4?
3. **Database Migration Timing:** Does Tarun Saini accept creating `knowledge_base_articles` with `vector(384)` directly in Milestone M5, or does the backend require a configurable migration script?
4. **Context Window Sufficiency:** Given that 100% of our current 113 raw chunks are $\le 169$ tokens (comfortably below MiniLM's 256 ceiling), does the team foresee future documents requiring chunks $> 256$ tokens that would necessitate a 512-token model?
5. **Quality Fallback Strategy:** If Milestone M7 retrieval evaluations show that MiniLM-L6-v2 fails complex semantic retrieval queries, what is our agreed escalation path (e.g. upgrade to local `bge-base-en-v1.5` 768-d vs fallback to cloud `text-embedding-3-small` 1536-d)?

---

*End of RFC-01 Document. Status: AGREED IN TEAM DECISION — PENDING FORMAL CONTRACT COMMIT.*

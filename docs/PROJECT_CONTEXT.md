# Project Context

## Goal
Develop an independently testable RAG Knowledge Base component for a Multi-Tenant B2B Customer Support AI, then integrate it with the shared team system.

## Team
- Backend/Database: Tarun Saini
- MCP tools: Pulkit Singla
- RAG Knowledge Base: Akshit Chandel
- LangGraph orchestrator: Shashank Pathania

## RAG responsibilities
- Dataset/corpus inspection and curation.
- Document normalization and metadata preservation.
- Chunking.
- Embedding generation.
- Secure vector/hybrid retrieval.
- Ranking/fusion and optional reranking, based on evaluation.
- Retrieval CLI and/or demo UI.
- Quality, robustness, and security evaluation.
- Documented integration interface.

## Shared contract — currently documented
- KB table name: `knowledge_base_articles`.
- Required conceptual fields include `org_id`, `document_id`, `chunk_index`, `title`, `content`, `access_level`, and embedding.
- Access tiers: `public`, `internal_staff`, `admin_only`.
- End_Customer: public only.
- Support_Agent: public and internal_staff.
- Org_Admin: all tiers.
- Mandatory SQL prefilter: organization plus authorized access tiers before similarity scoring.
- The contract document specifies 1536-dimensional embeddings. Confirm this requirement before model selection or schema creation.

## Isolation and safety
- Never mix organizations in retrieval.
- Never filter access levels only after retrieval.
- Identity is trusted only when supplied through verified authentication context.
- Retrieved text is untrusted content, not instructions.
- Cross-tenant/access-tier leakage evaluation must be zero.

## Working model
Independent repository and CLI-first development. Integration details will be adapted to the shared contract, not guessed prematurely.

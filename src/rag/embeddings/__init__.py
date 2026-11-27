"""Embeddings subsystem for RAG Knowledge Base.

Phase 1 deliverable: Foundation Models & Configuration.
Phase 2 deliverable: Client Interface & Deterministic Mock Client.
Phase 3 deliverable: Pipeline Orchestrator (Tenant Batching & Reassembly).
"""

from src.rag.embeddings.client import (
    HAS_SENTENCE_TRANSFORMERS,
    BaseEmbeddingClient,
    DeterministicMockEmbeddingClient,
    SentenceTransformerEmbeddingClient,
)
from src.rag.embeddings.models import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_EXPECTED_DIMENSIONS,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_NORMALIZE_EMBEDDINGS,
    EmbeddedChunk,
    EmbeddingConfig,
    EmbeddingConfigError,
    EmbeddingError,
    EmbeddingInferenceError,
    EmbeddingRateLimitError,
    EmbeddingValidationError,
)
from src.rag.embeddings.pipeline import EmbeddingPipeline

__all__ = [
    "BaseEmbeddingClient",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DEVICE",
    "DEFAULT_EXPECTED_DIMENSIONS",
    "DEFAULT_MAX_SEQ_LENGTH",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_NORMALIZE_EMBEDDINGS",
    "DeterministicMockEmbeddingClient",
    "EmbeddedChunk",
    "EmbeddingConfig",
    "EmbeddingConfigError",
    "EmbeddingError",
    "EmbeddingInferenceError",
    "EmbeddingPipeline",
    "EmbeddingRateLimitError",
    "EmbeddingValidationError",
    "HAS_SENTENCE_TRANSFORMERS",
    "SentenceTransformerEmbeddingClient",
]

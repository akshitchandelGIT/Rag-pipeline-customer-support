"""Data models, configuration, and exception hierarchy for embeddings subsystem.

Phase 1 deliverable grounded in:
- docs/plans/04-embeddings-plan.md
- docs/plans/04-embeddings-implementation-plan.md
- Shared Team Contract v3.0 Locked & RFC-01 (sentence-transformers/all-MiniLM-L6-v2, 384 dimensions)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Tuple

from src.rag.chunking.models import DocumentChunk

# -----------------------------------------------------------------------------
# Constants & Defaults
# -----------------------------------------------------------------------------

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EXPECTED_DIMENSIONS = 384
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_SEQ_LENGTH = 256
DEFAULT_DEVICE = "cpu"
DEFAULT_NORMALIZE_EMBEDDINGS = True


# -----------------------------------------------------------------------------
# Custom Exception Hierarchy
# -----------------------------------------------------------------------------


class EmbeddingError(Exception):
    """Base exception for all embedding subsystem errors."""


class EmbeddingConfigError(EmbeddingError):
    """Raised when configuration parameters or environment variables are invalid."""


class EmbeddingValidationError(EmbeddingError):
    """Raised when input text or output vector fails schema, length, or dimensional constraints."""


class EmbeddingInferenceError(EmbeddingError):
    """Raised when local or remote model inference fails or produces unrecoverable errors."""


class EmbeddingRateLimitError(EmbeddingError):
    """Raised when rate limits are encountered and retries are exhausted (remote client compatibility)."""


# -----------------------------------------------------------------------------
# Configuration Dataclass
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingConfig:
    """Immutable configuration container for embedding generation.

    Attributes:
        model_name: Canonical model identifier or path.
        expected_dimensions: Expected length of generated embedding vector (default: 384).
        batch_size: Maximum number of chunks/texts processed per batch.
        max_seq_length: Maximum sequence length (in word pieces) supported by model.
        device: Target execution device ('cpu' or 'cuda').
        normalize_embeddings: Whether output vectors must be normalized to unit length.
    """

    model_name: str = DEFAULT_MODEL_NAME
    expected_dimensions: int = DEFAULT_EXPECTED_DIMENSIONS
    batch_size: int = DEFAULT_BATCH_SIZE
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH
    device: str = DEFAULT_DEVICE
    normalize_embeddings: bool = DEFAULT_NORMALIZE_EMBEDDINGS

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise EmbeddingConfigError("model_name must be a non-empty string")
        if not isinstance(self.expected_dimensions, int) or self.expected_dimensions <= 0:
            raise EmbeddingConfigError(
                f"expected_dimensions must be a positive integer, got {self.expected_dimensions}"
            )
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise EmbeddingConfigError(
                f"batch_size must be a positive integer, got {self.batch_size}"
            )
        if not isinstance(self.max_seq_length, int) or self.max_seq_length <= 0:
            raise EmbeddingConfigError(
                f"max_seq_length must be a positive integer, got {self.max_seq_length}"
            )
        if not isinstance(self.device, str) or not self.device.strip():
            raise EmbeddingConfigError("device must be a non-empty string")
        if not isinstance(self.normalize_embeddings, bool):
            raise EmbeddingConfigError("normalize_embeddings must be a boolean")


# -----------------------------------------------------------------------------
# Immutable Embedded Chunk Container
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddedChunk:
    """Immutable container pairing a DocumentChunk with its vector embedding.

    Guarantees deep immutability by storing the numerical vector as an
    immutable Tuple of finite floats. Exposes direct property aliases matching
    the target PostgreSQL knowledge_base_articles schema.
    """

    chunk: DocumentChunk
    embedding: Tuple[float, ...]
    model: str = DEFAULT_MODEL_NAME

    def __post_init__(self) -> None:
        # Validate parent chunk type
        if not isinstance(self.chunk, DocumentChunk):
            raise EmbeddingValidationError(
                f"chunk must be an instance of DocumentChunk, got {type(self.chunk).__name__}"
            )

        # Enforce tuple conversion if a list was passed
        if isinstance(self.embedding, list):
            object.__setattr__(self, "embedding", tuple(self.embedding))

        # Validate sequence type
        if not isinstance(self.embedding, tuple):
            raise EmbeddingValidationError(
                f"embedding must be a tuple of floats, got {type(self.embedding).__name__}"
            )

        # Validate vector non-emptiness
        if len(self.embedding) == 0:
            raise EmbeddingValidationError("embedding vector cannot be empty")

        # Validate finite numeric values
        for idx, val in enumerate(self.embedding):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise EmbeddingValidationError(
                    f"embedding element at index {idx} must be a float, got {type(val).__name__}"
                )
            if not math.isfinite(val):
                raise EmbeddingValidationError(
                    f"embedding element at index {idx} must be a finite float (got {val})"
                )

        # Validate model string
        if not isinstance(self.model, str) or not self.model.strip():
            raise EmbeddingValidationError("model must be a non-empty string")

    # -------------------------------------------------------------------------
    # Target Database Schema Property Aliases (knowledge_base_articles)
    # -------------------------------------------------------------------------

    @property
    def id(self) -> str:
        """Database primary key (UUID v5), aliasing chunk.chunk_id."""
        return self.chunk.chunk_id

    @property
    def chunk_id(self) -> str:
        """Original RFC 4122 UUID v5 chunk identifier."""
        return self.chunk.chunk_id

    @property
    def document_id(self) -> str:
        """Parent document UUID v4."""
        return self.chunk.document_id

    @property
    def org_id(self) -> str:
        """Tenant UUID v4."""
        return self.chunk.org_id

    @property
    def access_level(self) -> str:
        """Document access tier ('public', 'internal_staff', 'admin_only')."""
        return self.chunk.access_level

    @property
    def chunk_index(self) -> int:
        """0-based sequential index within parent document."""
        return self.chunk.chunk_index

    @property
    def title(self) -> str:
        """Parent document title."""
        return self.chunk.title

    @property
    def content(self) -> str:
        """Complete chunk text payload (including contextual breadcrumb)."""
        return self.chunk.content

    @property
    def dimensions(self) -> int:
        """Dimensionality of the vector representation."""
        return len(self.embedding)

    def as_list(self) -> List[float]:
        """Convert immutable tuple embedding to a standard list for database adapters."""
        return list(self.embedding)

"""Embedding client interface and offline deterministic mock client.

Phase 2 deliverable grounded in:
- docs/plans/04-embeddings-plan.md
- docs/plans/04-embeddings-implementation-plan.md
- Shared Team Contract v3.0 Locked & RFC-01 (sentence-transformers/all-MiniLM-L6-v2, 384 dimensions)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import math
import random
from typing import List, Tuple

from src.rag.embeddings.models import (
    DEFAULT_EXPECTED_DIMENSIONS,
    EmbeddingConfigError,
    EmbeddingValidationError,
)

# -----------------------------------------------------------------------------
# Base Embedding Client Protocol
# -----------------------------------------------------------------------------


class BaseEmbeddingClient(ABC):
    """Abstract base class defining the model-independent embedding client interface.

    Supports both batch embedding (ingestion) and single-text embedding (retrieval)
    while exposing model metadata and dimensionality properties.
    """

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[Tuple[float, ...]]:
        """Generate embedding vectors for a batch of text strings.

        Args:
            texts: List of non-empty text strings to embed.

        Returns:
            List of float tuples matching the length and order of the input texts.

        Raises:
            EmbeddingValidationError: If any text string is empty or invalid.
            EmbeddingError: On underlying inference or transport failure.
        """
        pass

    def embed_text(self, text: str) -> Tuple[float, ...]:
        """Convenience method: generate an embedding vector for a single text string.

        Args:
            text: Single non-empty text string to embed.

        Returns:
            Immutable tuple of floats normalized to unit length.

        Raises:
            EmbeddingValidationError: If text is empty, non-string, or whitespace-only.
            EmbeddingError: On underlying inference or transport failure.
        """
        if not isinstance(text, str):
            raise EmbeddingValidationError(f"text must be a string, got {type(text).__name__}")
        if not text.strip():
            raise EmbeddingValidationError("Input text must be a non-empty, non-whitespace string")
        return self.embed_texts([text])[0]

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Expected dimensionality of output embedding vectors."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Canonical identifier of the underlying embedding model."""
        pass


# -----------------------------------------------------------------------------
# Offline Deterministic Mock Client
# -----------------------------------------------------------------------------


class DeterministicMockEmbeddingClient(BaseEmbeddingClient):
    """Offline, deterministic mock embedding client for testing and verification.

    Generates reproducible, unit-normalized float tuples using Python standard
    library hashlib and random.Random seeded from text content digests.
    Requires zero model weights, zero third-party dependencies, and zero network access.
    """

    def __init__(
        self,
        dimensions: int = DEFAULT_EXPECTED_DIMENSIONS,
        model_name: str = "mock-miniLM-384",
    ) -> None:
        """Initialize mock client with specified dimensions and model identifier.

        Args:
            dimensions: Output vector length (default: 384).
            model_name: Descriptive model identifier string.

        Raises:
            EmbeddingConfigError: If dimensions <= 0 or model_name is invalid.
        """
        if not isinstance(dimensions, int) or dimensions <= 0:
            raise EmbeddingConfigError(
                f"dimensions must be a positive integer, got {dimensions}"
            )
        if not isinstance(model_name, str) or not model_name.strip():
            raise EmbeddingConfigError("model_name must be a non-empty string")

        self._dimensions = dimensions
        self._model_name = model_name

    @property
    def dimensions(self) -> int:
        """Output dimensionality of generated vectors."""
        return self._dimensions

    @property
    def model_name(self) -> str:
        """Model identifier string."""
        return self._model_name

    def embed_texts(self, texts: List[str]) -> List[Tuple[float, ...]]:
        """Generate deterministic embedding vectors for a batch of text strings.

        Args:
            texts: List of text strings.

        Returns:
            List of unit-normalized float tuples of length self.dimensions.

        Raises:
            EmbeddingValidationError: If texts is not a list, contains non-strings,
                or contains empty/whitespace-only strings.
        """
        if not isinstance(texts, list):
            raise EmbeddingValidationError(f"texts must be a list, got {type(texts).__name__}")
        if not texts:
            return []

        results: List[Tuple[float, ...]] = []
        for idx, text in enumerate(texts):
            if not isinstance(text, str):
                raise EmbeddingValidationError(
                    f"Text item at index {idx} must be a string, got {type(text).__name__}"
                )
            if not text.strip():
                raise EmbeddingValidationError(
                    f"Text item at index {idx} must be a non-empty, non-whitespace string"
                )

            # Extract deterministic 64-bit integer seed from SHA-256 digest
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed_int = int.from_bytes(digest[:8], byteorder="big")
            rng = random.Random(seed_int)

            # Sample Gaussian distribution N(0, 1) and ensure explicit float type conversion
            raw_floats = [float(rng.gauss(0.0, 1.0)) for _ in range(self._dimensions)]
            norm = math.sqrt(sum(x * x for x in raw_floats))

            # Defensive guard against zero norm
            if norm == 0.0 or not math.isfinite(norm):
                raise EmbeddingValidationError("Generated vector norm is zero or non-finite")

            unit_vector = tuple(float(x / norm) for x in raw_floats)
            results.append(unit_vector)

        return results

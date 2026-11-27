"""Embedding client interface, offline deterministic mock client, and real SentenceTransformer client.

Deliverables grounded in:
- docs/plans/05-real-embedding-inference-plan.md
- docs/specs/SPEC-05-real-embedding-inference.md
- Shared Team Contract v3.0 Locked & RFC-01 (sentence-transformers/all-MiniLM-L6-v2, 384 dimensions)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import gc
import hashlib
import logging
import math
import random
from typing import List, Optional, Tuple

try:
    import torch
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    torch = None
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False

from src.rag.embeddings.models import (
    DEFAULT_EXPECTED_DIMENSIONS,
    EmbeddingConfig,
    EmbeddingConfigError,
    EmbeddingInferenceError,
    EmbeddingValidationError,
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Base Embedding Client Protocol
# -----------------------------------------------------------------------------


class BaseEmbeddingClient(ABC):
    """Abstract base class defining the model-independent embedding client interface.

    Supports both batch embedding (ingestion) and single-text embedding (retrieval)
    while exposing model metadata, dimensionality properties, and token counting.
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

    def count_tokens(self, text: str) -> int:
        """Count tokens in text for sequence validation.

        Default implementation falls back to word-heuristic estimation
        to preserve backward compatibility with mock test clients and test doubles.
        Subclasses with actual tokenizers override this method.
        """
        words = len(text.split())
        return math.ceil(words * 1.33)


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


# -----------------------------------------------------------------------------
# Real SentenceTransformer Embedding Client (all-MiniLM-L6-v2)
# -----------------------------------------------------------------------------


class SentenceTransformerEmbeddingClient(BaseEmbeddingClient):
    """Local, in-process embedding client using sentence-transformers on CPU.

    Implements BaseEmbeddingClient for sentence-transformers/all-MiniLM-L6-v2,
    producing 384-dimensional unit-normalized float tuples.
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        local_files_only: bool = False,
    ) -> None:
        """Initialize client configuration without loading heavy weights.

        Args:
            config: Optional EmbeddingConfig. Defaults to standard EmbeddingConfig.
            local_files_only: If True, restricts model loading to local disk cache only,
                prohibiting any network access.

        Raises:
            EmbeddingConfigError: If config parameters are invalid or incompatible.
        """
        if config is None:
            config = EmbeddingConfig()
        elif not isinstance(config, EmbeddingConfig):
            raise EmbeddingConfigError(
                f"config must be an instance of EmbeddingConfig, got {type(config).__name__}"
            )

        if not isinstance(local_files_only, bool):
            raise EmbeddingConfigError("local_files_only must be a boolean")

        self._config = config
        self._local_files_only = local_files_only
        self._dimensions = config.expected_dimensions
        self._model = None

    def load_model(self) -> None:
        """Explicitly load model weights and tokenizer into memory if not already loaded.

        Raises:
            EmbeddingInferenceError: If sentence-transformers is missing or weights cannot be loaded.
        """
        if self._model is not None:
            return

        if not HAS_SENTENCE_TRANSFORMERS:
            raise EmbeddingInferenceError(
                "sentence-transformers is not installed in the environment."
            )

        try:
            model = SentenceTransformer(
                self._config.model_name,
                device=self._config.device,
                local_files_only=self._local_files_only,
            )
            model.max_seq_length = self._config.max_seq_length
            model.eval()
            self._model = model
        except (OSError, ValueError) as exc:
            raise EmbeddingInferenceError(
                f"Model weights for '{self._config.model_name}' could not be loaded "
                f"(local_files_only={self._local_files_only}): {exc}"
            ) from exc
        except Exception as exc:
            raise EmbeddingInferenceError(
                f"Unexpected error loading model '{self._config.model_name}': {exc}"
            ) from exc

    def unload_model(self) -> None:
        """Release in-memory model weights and trigger garbage collection."""
        self._model = None
        gc.collect()

    @property
    def is_loaded(self) -> bool:
        """Return True if model weights are currently resident in memory."""
        return self._model is not None

    @property
    def dimensions(self) -> int:
        """Output dimensionality of generated vectors (strictly 384)."""
        return self._dimensions

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

        Raises:
            EmbeddingValidationError: If text is not a string.
            EmbeddingInferenceError: If model cannot be loaded.
        """
        if not isinstance(text, str):
            raise EmbeddingValidationError(f"text must be a string, got {type(text).__name__}")
        self.load_model()
        encoded = self._model.tokenizer(text, truncation=False, add_special_tokens=True, verbose=False)
        return len(encoded["input_ids"])

    def embed_texts(self, texts: List[str]) -> List[Tuple[float, ...]]:
        """Generate embedding vectors for a batch of text strings (bulk ingestion).

        Permits truncation (truncation=True) for bulk ingestion so outlier documents
        do not crash ingestion batches. Emits structured content-free warnings when
        any input sequence exceeds max_seq_length.

        Args:
            texts: List of non-empty text strings to embed.

        Returns:
            List of 384-dimensional unit-normalized float tuples matching input length and order.

        Raises:
            EmbeddingValidationError: If texts is malformed, contains empty strings, or output is non-finite.
            EmbeddingInferenceError: On runtime model crash or memory allocation failure.
        """
        if not isinstance(texts, list):
            raise EmbeddingValidationError(f"texts must be a list, got {type(texts).__name__}")
        if not texts:
            return []

        for idx, text in enumerate(texts):
            if not isinstance(text, str):
                raise EmbeddingValidationError(
                    f"Text item at index {idx} must be a string, got {type(text).__name__}"
                )
            if not text.strip():
                raise EmbeddingValidationError(
                    f"Text item at index {idx} must be a non-empty, non-whitespace string"
                )

        self.load_model()

        # Tokenizer ground-truth inspection for truncation warning
        tokenized_batch = self._model.tokenizer(texts, truncation=False, add_special_tokens=True, verbose=False)
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
                    },
                )

        # Execute inference
        try:
            with torch.inference_mode():
                raw_embeddings = self._model.encode(
                    texts,
                    batch_size=self._config.batch_size,
                    normalize_embeddings=self._config.normalize_embeddings,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
        except Exception as exc:
            raise EmbeddingInferenceError(f"Model inference failed: {exc}") from exc

        # Validate shape and finite floats
        expected_shape = (len(texts), self.dimensions)
        if raw_embeddings.shape != expected_shape:
            raise EmbeddingValidationError(
                f"Output embedding shape mismatch: expected {expected_shape}, "
                f"got {raw_embeddings.shape}"
            )

        results: List[Tuple[float, ...]] = []
        for r_idx, row in enumerate(raw_embeddings):
            float_tuple = tuple(float(x) for x in row)
            for v_idx, val in enumerate(float_tuple):
                if not math.isfinite(val):
                    raise EmbeddingValidationError(
                        f"Non-finite float detected at row {r_idx}, column {v_idx}: {val}"
                    )
            results.append(float_tuple)

        return results

    def embed_text(self, text: str) -> Tuple[float, ...]:
        """Convenience method: generate an embedding vector for a single text string.

        Enforces strict sequence limit validation: queries exceeding max_seq_length
        are rejected with EmbeddingValidationError rather than silently truncated.

        Args:
            text: Single text string to embed.

        Returns:
            384-dimensional unit-normalized float tuple.

        Raises:
            EmbeddingValidationError: If text is empty, whitespace-only, or exceeds max_seq_length.
            EmbeddingInferenceError: On model loading or inference failure.
        """
        if not isinstance(text, str):
            raise EmbeddingValidationError(f"text must be a string, got {type(text).__name__}")
        stripped = text.strip()
        if not stripped:
            raise EmbeddingValidationError("Input text must be a non-empty, non-whitespace string")

        token_count = self.count_tokens(stripped)
        if token_count > self._config.max_seq_length:
            raise EmbeddingValidationError(
                f"Search query exceeds maximum sequence token limit of {self._config.max_seq_length} "
                f"(actual tokenizer count: {token_count} tokens, including special tokens)"
            )

        return self.embed_texts([stripped])[0]

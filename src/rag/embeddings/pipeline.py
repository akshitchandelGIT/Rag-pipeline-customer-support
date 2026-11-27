"""Embedding pipeline orchestrator: tenant batching, indexed reassembly, and query vectorization.

Phase 3 deliverable grounded in:
- docs/plans/04-embeddings-plan.md
- docs/plans/04-embeddings-implementation-plan.md
- Shared Team Contract v3.0 Locked & RFC-01 (sentence-transformers/all-MiniLM-L6-v2, 384 dimensions)
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Dict, List, Optional, Tuple

from src.rag.chunking.models import DocumentChunk
from src.rag.embeddings.client import BaseEmbeddingClient
from src.rag.embeddings.models import (
    EmbeddedChunk,
    EmbeddingConfig,
    EmbeddingConfigError,
    EmbeddingValidationError,
)


class EmbeddingPipeline:
    """Orchestrates document chunk embedding and search query vectorization.

    Responsibilities:
    1. Tenant Partitioning: Groups chunks by org_id before batching so that chunks
       from different tenants are never co-mingled in a single embedding batch.
    2. Batch Slicing: Slices per-tenant chunks into batches respecting config.batch_size.
    3. Strict Indexed Reassembly: Tags input chunks with original indices (0..N-1)
       and sorts assembled EmbeddedChunk instances back to exact caller input order.
    4. Dimensionality & Finite Validation: Validates returned vector dimensions against
       client.dimensions and ensures all numerical elements are finite floats.
    5. Query Vectorization: Embeds runtime search queries with token ceiling validation
       matching document vectorization models.
    """

    def __init__(
        self,
        client: BaseEmbeddingClient,
        config: Optional[EmbeddingConfig] = None,
    ) -> None:
        """Initialize the embedding pipeline with an embedding client and configuration.

        Args:
            client: Concrete implementation of BaseEmbeddingClient.
            config: Optional EmbeddingConfig. Defaults to default EmbeddingConfig().

        Raises:
            EmbeddingConfigError: If client is not a BaseEmbeddingClient instance or
                config is not an EmbeddingConfig instance.
        """
        if not isinstance(client, BaseEmbeddingClient):
            raise EmbeddingConfigError(
                f"client must be an instance of BaseEmbeddingClient, got {type(client).__name__}"
            )

        if config is None:
            config = EmbeddingConfig()
        elif not isinstance(config, EmbeddingConfig):
            raise EmbeddingConfigError(
                f"config must be an instance of EmbeddingConfig, got {type(config).__name__}"
            )

        self._client = client
        self._config = config

    @property
    def client(self) -> BaseEmbeddingClient:
        """Configured embedding client instance."""
        return self._client

    @property
    def config(self) -> EmbeddingConfig:
        """Configured embedding parameters."""
        return self._config

    def embed_query(self, query: str) -> Tuple[float, ...]:
        """Generate a validated embedding vector for a runtime search query.

        Args:
            query: Raw search query text string.

        Returns:
            Immutable tuple of normalized floats matching client.dimensions.

        Raises:
            EmbeddingValidationError: If query is empty, whitespace-only, non-string,
                or exceeds the configured maximum sequence length.
            EmbeddingError: On underlying model inference failure.
        """
        if not isinstance(query, str):
            raise EmbeddingValidationError(
                f"query must be a string, got {type(query).__name__}"
            )
        stripped = query.strip()
        if not stripped:
            raise EmbeddingValidationError(
                "Search query must be a non-empty, non-whitespace string"
            )

        actual_tokens = self._client.count_tokens(stripped)
        if actual_tokens > self._config.max_seq_length:
            raise EmbeddingValidationError(
                f"Search query exceeds maximum sequence token limit of "
                f"{self._config.max_seq_length} (actual tokenizer count: {actual_tokens} tokens, including special tokens)"
            )

        vector = self._client.embed_text(stripped)

        # Output validation
        if not isinstance(vector, (tuple, list)):
            raise EmbeddingValidationError(
                f"Client returned non-sequence vector type: {type(vector).__name__}"
            )

        if len(vector) != self._client.dimensions:
            raise EmbeddingValidationError(
                f"Query vector dimension mismatch: expected {self._client.dimensions}, "
                f"got {len(vector)}"
            )

        for idx, val in enumerate(vector):
            if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val):
                raise EmbeddingValidationError(
                    f"Query vector element at index {idx} contains non-finite value: {val}"
                )

        return tuple(float(x) for x in vector)

    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[EmbeddedChunk]:
        """Embed a list of DocumentChunks with tenant-isolated batching and indexed reassembly.

        Guarantees:
        1. 1:1 positional correspondence: output[i] corresponds to chunks[i] for all 0 <= i < len(chunks).
        2. Strict tenant batch boundaries: no client call combines chunks from multiple tenants.
        3. Configured batch size respected: batches never exceed config.batch_size.
        4. Atomic execution: if any client call fails, the exception propagates and no partial
           result is emitted.

        Args:
            chunks: List of DocumentChunk instances to embed.

        Returns:
            List of EmbeddedChunk instances in identical order to input chunks.

        Raises:
            EmbeddingValidationError: If input is not a list, contains non-DocumentChunk objects,
                contains empty/whitespace content, or client returns invalid/mismatched vectors.
            EmbeddingError: On underlying model inference failure.
        """
        if not isinstance(chunks, list):
            raise EmbeddingValidationError(
                f"chunks must be a list, got {type(chunks).__name__}"
            )

        if not chunks:
            return []

        # Validate all items prior to batch processing
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, DocumentChunk):
                raise EmbeddingValidationError(
                    f"Chunk at index {idx} must be an instance of DocumentChunk, got {type(chunk).__name__}"
                )
            if not isinstance(chunk.content, str) or not chunk.content.strip():
                raise EmbeddingValidationError(
                    f"Chunk at index {idx} ({chunk.chunk_id}) has empty or whitespace-only content"
                )

        # Step 1: Tag with Original Input Index (0..N-1)
        indexed_chunks: List[Tuple[int, DocumentChunk]] = list(enumerate(chunks))

        # Step 2: Tenant Partitioning (Group by org_id)
        tenant_groups: Dict[str, List[Tuple[int, DocumentChunk]]] = defaultdict(list)
        for orig_idx, chunk in indexed_chunks:
            tenant_groups[chunk.org_id].append((orig_idx, chunk))

        # Step 3 & 4: Batch Slicing per Tenant, Vector Generation & Packaging
        assembled_results: List[Tuple[int, EmbeddedChunk]] = []

        for org_id, items in tenant_groups.items():
            for i in range(0, len(items), self._config.batch_size):
                batch_items = items[i : i + self._config.batch_size]
                batch_texts = [item_chunk.content for _, item_chunk in batch_items]

                # Dispatch single-tenant batch to client
                vectors = self._client.embed_texts(batch_texts)

                # Validate vector batch count
                if not isinstance(vectors, list):
                    raise EmbeddingValidationError(
                        f"Client returned non-list result: {type(vectors).__name__}"
                    )
                if len(vectors) != len(batch_items):
                    raise EmbeddingValidationError(
                        f"Client returned {len(vectors)} vectors for batch of size {len(batch_items)}"
                    )

                # Validate each vector and assemble
                for (orig_idx, item_chunk), vec in zip(batch_items, vectors):
                    if not isinstance(vec, (tuple, list)):
                        raise EmbeddingValidationError(
                            f"Vector for chunk index {orig_idx} is not a sequence, got {type(vec).__name__}"
                        )
                    if len(vec) != self._client.dimensions:
                        raise EmbeddingValidationError(
                            f"Vector dimension mismatch for chunk {orig_idx} ({item_chunk.chunk_id}): "
                            f"expected {self._client.dimensions}, got {len(vec)}"
                        )

                    for v_idx, val in enumerate(vec):
                        if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val):
                            raise EmbeddingValidationError(
                                f"Vector for chunk {orig_idx} element at {v_idx} is non-finite: {val}"
                            )

                    float_tuple = tuple(float(x) for x in vec)
                    embedded = EmbeddedChunk(
                        chunk=item_chunk,
                        embedding=float_tuple,
                        model=self._client.model_name,
                    )
                    assembled_results.append((orig_idx, embedded))

        # Step 5: Strict Positional Reassembly by Original Input Index
        assembled_results.sort(key=lambda pair: pair[0])
        return [embedded for _, embedded in assembled_results]

"""Unit test suite for M4 Embeddings Subsystem.

Phase 1 test harness verifying:
- Foundation models and dataclasses (EmbeddedChunk, EmbeddingConfig)
- Custom exception hierarchy
- Deep immutability of vectors
- Target database schema property aliases
- Strict input validation

Phase 2 test harness verifying:
- Client interface abstraction (BaseEmbeddingClient)
- Offline deterministic mock client (DeterministicMockEmbeddingClient)
- Vector dimensionality (384-d default and custom dimensions)
- Mathematical unit normalization (L2 = 1.0)
- Bitwise determinism across repeated runs and separate instances
- Float type consistency and finite float integrity
- Single-text embedding interface (embed_text)
- Empty and batch processing
- Input and configuration boundary validations

Phase 3 test harness verifying:
- Pipeline orchestration (EmbeddingPipeline)
- Tenant batch isolation (zero cross-tenant batch mixing)
- Strict 1:1 original positional index reassembly
- Batch sizing and boundary handling (exact multiples and partial remainders)
- Empty input and single/multiple chunk processing
- Query embedding interface with token ceiling validation
- Output dimensionality and finite float validation
- Client error propagation without partial success
"""

import dataclasses
import glob
import math
from pathlib import Path
import time
from typing import List, Tuple
import unittest

from src.rag.chunking.engine import chunk_document
from src.rag.chunking.models import (
    DocumentChunk,
    estimate_tokens,
    generate_chunk_id,
)
from src.rag.embeddings.client import (
    BaseEmbeddingClient,
    DeterministicMockEmbeddingClient,
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


class TestM4Phase1Models(unittest.TestCase):
    """Unit tests for Phase 1 foundational data structures and configuration."""

    def setUp(self) -> None:
        """Create standard synthetic DocumentChunk fixture."""
        self.doc_id = "11111111-1111-4111-8111-111111111111"
        self.org_id = "22222222-2222-4222-8222-222222222222"
        self.chunk_index = 0
        self.chunk_id = generate_chunk_id(self.doc_id, self.chunk_index)
        self.title = "Password Reset Policy"
        self.content = "[Document: Password Reset Policy > Overview]\n\nInstructions to reset."
        self.chunk = DocumentChunk(
            chunk_id=self.chunk_id,
            document_id=self.doc_id,
            org_id=self.org_id,
            access_level="public",
            chunk_index=self.chunk_index,
            title=self.title,
            content=self.content,
            heading_path=["Password Reset Policy", "Overview"],
            token_count_est=15,
            content_hash="mockhash123",
        )
        # Synthetic 384-dimensional unit vector
        self.mock_vector = tuple(0.1 for _ in range(384))

    def test_embedded_chunk_creation(self) -> None:
        """T4-01: Verifies construction, property delegation, and dimension calculation."""
        embedded = EmbeddedChunk(
            chunk=self.chunk,
            embedding=self.mock_vector,
        )

        # Core attribute assertions
        self.assertEqual(embedded.chunk, self.chunk)
        self.assertEqual(embedded.embedding, self.mock_vector)
        self.assertEqual(embedded.model, DEFAULT_MODEL_NAME)
        self.assertEqual(embedded.dimensions, 384)

        # Target database contract property alias assertions (knowledge_base_articles)
        self.assertEqual(embedded.id, self.chunk_id)
        self.assertEqual(embedded.chunk_id, self.chunk_id)
        self.assertEqual(embedded.document_id, self.doc_id)
        self.assertEqual(embedded.org_id, self.org_id)
        self.assertEqual(embedded.access_level, "public")
        self.assertEqual(embedded.chunk_index, 0)
        self.assertEqual(embedded.title, self.title)
        self.assertEqual(embedded.content, self.content)

    def test_embedded_chunk_deep_immutability(self) -> None:
        """T4-02: Verifies vector is stored as tuple, cannot be mutated in place, and frozen dataclass rejects writes."""
        embedded = EmbeddedChunk(
            chunk=self.chunk,
            embedding=list(self.mock_vector),  # Pass mutable list
        )

        # Enforces conversion to immutable tuple
        self.assertIsInstance(embedded.embedding, tuple)

        # Attribute assignment raises FrozenInstanceError
        with self.assertRaises(dataclasses.FrozenInstanceError):
            embedded.model = "different-model"  # type: ignore

        with self.assertRaises(dataclasses.FrozenInstanceError):
            embedded.chunk = self.chunk  # type: ignore

    def test_embedded_chunk_as_list(self) -> None:
        """T4-03: Verifies .as_list() returns a standard mutable list matching tuple values."""
        embedded = EmbeddedChunk(
            chunk=self.chunk,
            embedding=self.mock_vector,
        )
        as_list = embedded.as_list()

        self.assertIsInstance(as_list, list)
        self.assertEqual(len(as_list), 384)
        self.assertEqual(as_list, list(self.mock_vector))

        # Mutating the returned list must not mutate the internal tuple
        as_list[0] = 999.0
        self.assertEqual(embedded.embedding[0], 0.1)

    def test_embedding_config_defaults(self) -> None:
        """T4-04: Verifies default configuration parameters and immutability."""
        config = EmbeddingConfig()

        self.assertEqual(config.model_name, "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(config.expected_dimensions, 384)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.max_seq_length, 256)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.normalize_embeddings, True)

        # Immutability
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.batch_size = 32  # type: ignore

    def test_embedded_chunk_validation_errors(self) -> None:
        """Verifies strict schema and value validation on EmbeddedChunk construction."""
        # Non-DocumentChunk object
        with self.assertRaises(EmbeddingValidationError):
            EmbeddedChunk(chunk="not-a-chunk", embedding=self.mock_vector)  # type: ignore

        # Empty vector
        with self.assertRaises(EmbeddingValidationError):
            EmbeddedChunk(chunk=self.chunk, embedding=())

        # Non-numeric element in vector
        with self.assertRaises(EmbeddingValidationError):
            EmbeddedChunk(chunk=self.chunk, embedding=(0.1, "bad", 0.3))  # type: ignore

        # Boolean in vector (bool is subclass of int in Python)
        with self.assertRaises(EmbeddingValidationError):
            EmbeddedChunk(chunk=self.chunk, embedding=(0.1, True, 0.3))  # type: ignore

        # NaN in vector
        with self.assertRaises(EmbeddingValidationError):
            EmbeddedChunk(chunk=self.chunk, embedding=(0.1, float("nan"), 0.3))

        # Infinity in vector
        with self.assertRaises(EmbeddingValidationError):
            EmbeddedChunk(chunk=self.chunk, embedding=(0.1, float("inf"), 0.3))

        # Empty model string
        with self.assertRaises(EmbeddingValidationError):
            EmbeddedChunk(chunk=self.chunk, embedding=self.mock_vector, model="  ")

    def test_embedding_config_validation_errors(self) -> None:
        """Verifies strict boundary validation on EmbeddingConfig parameters."""
        # Non-positive expected_dimensions
        with self.assertRaises(EmbeddingConfigError):
            EmbeddingConfig(expected_dimensions=0)
        with self.assertRaises(EmbeddingConfigError):
            EmbeddingConfig(expected_dimensions=-10)

        # Non-positive batch_size
        with self.assertRaises(EmbeddingConfigError):
            EmbeddingConfig(batch_size=0)

        # Non-positive max_seq_length
        with self.assertRaises(EmbeddingConfigError):
            EmbeddingConfig(max_seq_length=0)

        # Empty model_name
        with self.assertRaises(EmbeddingConfigError):
            EmbeddingConfig(model_name=" ")

        # Empty device
        with self.assertRaises(EmbeddingConfigError):
            EmbeddingConfig(device="")

        # Non-bool normalize_embeddings
        with self.assertRaises(EmbeddingConfigError):
            EmbeddingConfig(normalize_embeddings="yes")  # type: ignore

    def test_exception_hierarchy(self) -> None:
        """Verifies inheritance relationships in custom exception hierarchy."""
        self.assertTrue(issubclass(EmbeddingConfigError, EmbeddingError))
        self.assertTrue(issubclass(EmbeddingValidationError, EmbeddingError))
        self.assertTrue(issubclass(EmbeddingInferenceError, EmbeddingError))
        self.assertTrue(issubclass(EmbeddingRateLimitError, EmbeddingError))
        self.assertTrue(issubclass(EmbeddingError, Exception))


class TestM4Phase2Client(unittest.TestCase):
    """Unit tests for Phase 2 client interface and deterministic mock client."""

    def setUp(self) -> None:
        """Instantiate mock client instances for testing."""
        self.client = DeterministicMockEmbeddingClient()
        self.sample_text = "[Document: SLA Commitments > Overview]\n\nStandard response times."

    def test_mock_client_dimensions(self) -> None:
        """T4-05: Asserts output vectors have exactly 384 dimensions by default and respect custom dimensions."""
        # Default dimensions (384)
        vectors = self.client.embed_texts([self.sample_text])
        self.assertEqual(len(vectors), 1)
        self.assertEqual(len(vectors[0]), 384)
        self.assertEqual(self.client.dimensions, 384)
        self.assertEqual(self.client.model_name, "mock-miniLM-384")

        # Custom dimensions (e.g. 128, 512, 1536)
        custom_client = DeterministicMockEmbeddingClient(dimensions=128, model_name="custom-128")
        custom_vectors = custom_client.embed_texts([self.sample_text])
        self.assertEqual(len(custom_vectors[0]), 128)
        self.assertEqual(custom_client.dimensions, 128)
        self.assertEqual(custom_client.model_name, "custom-128")

    def test_mock_client_unit_norm(self) -> None:
        """T4-06: Asserts vector L2 norm equals 1.0 +/- 10^-5 across multiple distinct texts."""
        texts = [
            "Short text",
            "A slightly longer sentence about customer support billing policy.",
            "[Document: Architecture > Overview]\n\nTechnical details with code: `SELECT 1;`.",
        ]
        vectors = self.client.embed_texts(texts)
        self.assertEqual(len(vectors), len(texts))

        for idx, vec in enumerate(vectors):
            l2_norm = math.sqrt(sum(x * x for x in vec))
            self.assertAlmostEqual(
                l2_norm,
                1.0,
                places=5,
                msg=f"Vector at index {idx} failed L2 unit norm constraint (got {l2_norm})",
            )

    def test_mock_client_determinism(self) -> None:
        """T4-07: Asserts identical input strings produce bitwise identical vectors across instances and repeated calls."""
        client_a = DeterministicMockEmbeddingClient()
        client_b = DeterministicMockEmbeddingClient()

        # Repeated calls on the same client
        vec1 = self.client.embed_text(self.sample_text)
        vec2 = self.client.embed_text(self.sample_text)
        self.assertEqual(vec1, vec2)

        # Separate client instances
        vec_a = client_a.embed_text(self.sample_text)
        vec_b = client_b.embed_text(self.sample_text)
        self.assertEqual(vec_a, vec_b)

        # Batch vs single-text
        batch_vec = client_a.embed_texts([self.sample_text])[0]
        self.assertEqual(batch_vec, vec_a)

    def test_mock_client_distinct_texts(self) -> None:
        """T4-08: Asserts different text strings produce non-identical embedding vectors."""
        text_a = "Account security guidelines and multi-factor authentication setup."
        text_b = "Refund request procedure and return window deadlines."

        vec_a = self.client.embed_text(text_a)
        vec_b = self.client.embed_text(text_b)

        self.assertNotEqual(vec_a, vec_b)

    def test_single_text_embedding_interface(self) -> None:
        """T4-09: Asserts embed_text dispatches a single string, returns 384-d tuple, and rejects empty strings."""
        vector = self.client.embed_text(self.sample_text)

        self.assertIsInstance(vector, tuple)
        self.assertEqual(len(vector), 384)
        self.assertTrue(all(isinstance(x, float) for x in vector))

        # Empty text raises EmbeddingValidationError
        with self.assertRaises(EmbeddingValidationError):
            self.client.embed_text("")

        # Whitespace-only text raises EmbeddingValidationError
        with self.assertRaises(EmbeddingValidationError):
            self.client.embed_text("   \n\t  ")

        # Non-string raises EmbeddingValidationError
        with self.assertRaises(EmbeddingValidationError):
            self.client.embed_text(12345)  # type: ignore

    def test_mock_client_float_type_consistency(self) -> None:
        """Verifies that all elements in generated vectors are strictly finite Python float instances."""
        vector = self.client.embed_text(self.sample_text)

        for idx, val in enumerate(vector):
            # Assert exact float type (not int, not bool)
            self.assertIs(
                type(val),
                float,
                msg=f"Element at index {idx} is of type {type(val).__name__}, expected float",
            )
            self.assertTrue(
                math.isfinite(val),
                msg=f"Element at index {idx} is non-finite: {val}",
            )

    def test_mock_client_empty_and_batch_inputs(self) -> None:
        """Verifies handling of empty lists and multi-item batch lists."""
        # Empty list returns empty list
        self.assertEqual(self.client.embed_texts([]), [])

        # Multi-item batch preserves length and positions
        texts = [f"Text item number {i}" for i in range(10)]
        vectors = self.client.embed_texts(texts)
        self.assertEqual(len(vectors), 10)

        # Each vector in batch matches individual embed_text
        for i, text in enumerate(texts):
            individual_vec = self.client.embed_text(text)
            self.assertEqual(vectors[i], individual_vec)

    def test_mock_client_validation_errors(self) -> None:
        """Verifies validation boundaries on inputs and configuration."""
        # Non-list passed to embed_texts
        with self.assertRaises(EmbeddingValidationError):
            self.client.embed_texts("single-string-not-in-list")  # type: ignore

        with self.assertRaises(EmbeddingValidationError):
            self.client.embed_texts(None)  # type: ignore

        # List containing non-string
        with self.assertRaises(EmbeddingValidationError):
            self.client.embed_texts(["valid text", 42, "another valid"])  # type: ignore

        # List containing empty string
        with self.assertRaises(EmbeddingValidationError):
            self.client.embed_texts(["valid text", ""])

        # List containing whitespace string
        with self.assertRaises(EmbeddingValidationError):
            self.client.embed_texts(["valid text", "   \t\n  "])

        # Invalid dimensions in constructor
        with self.assertRaises(EmbeddingConfigError):
            DeterministicMockEmbeddingClient(dimensions=0)

        with self.assertRaises(EmbeddingConfigError):
            DeterministicMockEmbeddingClient(dimensions=-384)

        with self.assertRaises(EmbeddingConfigError):
            DeterministicMockEmbeddingClient(dimensions="384")  # type: ignore

        # Invalid model_name in constructor
        with self.assertRaises(EmbeddingConfigError):
            DeterministicMockEmbeddingClient(model_name="")

        with self.assertRaises(EmbeddingConfigError):
            DeterministicMockEmbeddingClient(model_name="   ")

    def test_mock_client_pairs_with_embedded_chunk(self) -> None:
        """Verifies seamless integration between DeterministicMockEmbeddingClient and Phase 1 EmbeddedChunk."""
        doc_id = "33333333-3333-4333-8333-333333333333"
        org_id = "44444444-4444-4444-8444-444444444444"
        chunk_id = generate_chunk_id(doc_id, 0)
        chunk = DocumentChunk(
            chunk_id=chunk_id,
            document_id=doc_id,
            org_id=org_id,
            access_level="internal_staff",
            chunk_index=0,
            title="Internal Diagnostics",
            content="[Document: Diagnostics > Step 1]\n\nRun health check command.",
            heading_path=["Diagnostics", "Step 1"],
            token_count_est=20,
            content_hash="hash987",
        )

        vector = self.client.embed_text(chunk.content)
        embedded = EmbeddedChunk(
            chunk=chunk,
            embedding=vector,
            model=self.client.model_name,
        )

        self.assertEqual(embedded.id, chunk_id)
        self.assertEqual(embedded.dimensions, 384)
        self.assertEqual(embedded.model, "mock-miniLM-384")
        self.assertEqual(embedded.access_level, "internal_staff")
        self.assertEqual(embedded.as_list(), list(vector))


# -----------------------------------------------------------------------------
# Phase 3 Test Doubles & Spy Clients
# -----------------------------------------------------------------------------


class SpyMockEmbeddingClient(DeterministicMockEmbeddingClient):
    """Spy mock client that records all batch calls and batch contents."""

    def __init__(
        self,
        dimensions: int = DEFAULT_EXPECTED_DIMENSIONS,
        model_name: str = "mock-miniLM-384",
    ) -> None:
        super().__init__(dimensions=dimensions, model_name=model_name)
        self.call_history: List[List[str]] = []

    def embed_texts(self, texts: List[str]) -> List[Tuple[float, ...]]:
        self.call_history.append(list(texts))
        return super().embed_texts(texts)


class FailingEmbeddingClient(BaseEmbeddingClient):
    """Client that simulates model inference runtime failures."""

    def __init__(self, dimensions: int = 384, model_name: str = "failing-client") -> None:
        self._dimensions = dimensions
        self._model_name = model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_texts(self, texts: List[str]) -> List[Tuple[float, ...]]:
        raise EmbeddingInferenceError("Simulated inference engine crash")


class DimensionMismatchEmbeddingClient(BaseEmbeddingClient):
    """Client that returns vectors of length mismatched to reported dimensions."""

    def __init__(self, dimensions: int = 384, return_dim: int = 128) -> None:
        self._dimensions = dimensions
        self._return_dim = return_dim

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return "mismatch-client"

    def embed_texts(self, texts: List[str]) -> List[Tuple[float, ...]]:
        return [tuple(0.1 for _ in range(self._return_dim)) for _ in texts]


class NonFiniteEmbeddingClient(BaseEmbeddingClient):
    """Client that returns non-finite values (NaN/Inf) in output vectors."""

    def __init__(self, dimensions: int = 384, inject_nan: bool = True) -> None:
        self._dimensions = dimensions
        self._inject_nan = inject_nan

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return "non-finite-client"

    def embed_texts(self, texts: List[str]) -> List[Tuple[float, ...]]:
        val = float("nan") if self._inject_nan else float("inf")
        return [tuple(val if i == 0 else 0.1 for i in range(self._dimensions)) for _ in texts]


# -----------------------------------------------------------------------------
# Phase 3 Pipeline Test Suite
# -----------------------------------------------------------------------------


class TestM4Phase3Pipeline(unittest.TestCase):
    """Unit tests for Phase 3 pipeline orchestrator."""

    def setUp(self) -> None:
        """Create standard client, pipeline, and synthetic document fixtures."""
        self.client = DeterministicMockEmbeddingClient()
        self.pipeline = EmbeddingPipeline(client=self.client)
        self.org_a = "11111111-1111-4111-8111-111111111111"
        self.org_b = "22222222-2222-4222-8222-222222222222"
        self.doc_a = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
        self.doc_b = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"

    def _create_chunk(
        self,
        doc_id: str,
        org_id: str,
        chunk_index: int,
        content: str = "",
        access_level: str = "public",
    ) -> DocumentChunk:
        """Helper to create synthetic DocumentChunk instances."""
        chunk_id = generate_chunk_id(doc_id, chunk_index)
        if not content:
            content = (
                f"[Document: Doc {doc_id[:8]} > Section {chunk_index}]\n\n"
                f"Sample content for chunk {chunk_index} of org {org_id[:8]}."
            )
        return DocumentChunk(
            chunk_id=chunk_id,
            document_id=doc_id,
            org_id=org_id,
            access_level=access_level,
            chunk_index=chunk_index,
            title=f"Doc {doc_id[:8]}",
            content=content,
            heading_path=[f"Doc {doc_id[:8]}", f"Section {chunk_index}"],
            token_count_est=25,
            content_hash="mockhash123",
        )

    def test_pipeline_initialization_and_properties(self) -> None:
        """Verifies pipeline construction, property delegation, and configuration validation."""
        # Default config
        pipeline = EmbeddingPipeline(client=self.client)
        self.assertIs(pipeline.client, self.client)
        self.assertEqual(pipeline.config.model_name, DEFAULT_MODEL_NAME)
        self.assertEqual(pipeline.config.batch_size, 64)

        # Custom config
        custom_config = EmbeddingConfig(batch_size=16)
        custom_pipeline = EmbeddingPipeline(client=self.client, config=custom_config)
        self.assertEqual(custom_pipeline.config.batch_size, 16)

        # Invalid client
        with self.assertRaises(EmbeddingConfigError):
            EmbeddingPipeline(client="not-a-client")  # type: ignore

        with self.assertRaises(EmbeddingConfigError):
            EmbeddingPipeline(client=None)  # type: ignore

        # Invalid config
        with self.assertRaises(EmbeddingConfigError):
            EmbeddingPipeline(client=self.client, config={"batch_size": 16})  # type: ignore

    def test_pipeline_empty_input(self) -> None:
        """Verifies passing empty chunk list returns empty list immediately with zero client calls."""
        spy_client = SpyMockEmbeddingClient()
        pipeline = EmbeddingPipeline(client=spy_client)

        results = pipeline.embed_chunks([])
        self.assertEqual(results, [])
        self.assertEqual(len(spy_client.call_history), 0)

    def test_single_chunk_embedding(self) -> None:
        """Verifies embedding a single DocumentChunk returns a valid EmbeddedChunk with correct attributes."""
        chunk = self._create_chunk(self.doc_a, self.org_a, 0)
        results = self.pipeline.embed_chunks([chunk])

        self.assertEqual(len(results), 1)
        embedded = results[0]
        self.assertIsInstance(embedded, EmbeddedChunk)
        self.assertEqual(embedded.chunk, chunk)
        self.assertEqual(embedded.id, chunk.chunk_id)
        self.assertEqual(embedded.dimensions, 384)
        self.assertEqual(embedded.model, self.client.model_name)

    def test_multiple_chunks_same_tenant(self) -> None:
        """Verifies embedding multiple chunks from a single tenant preserves sequential order."""
        chunks = [self._create_chunk(self.doc_a, self.org_a, i) for i in range(5)]
        results = self.pipeline.embed_chunks(chunks)

        self.assertEqual(len(results), 5)
        for i in range(5):
            self.assertEqual(results[i].chunk, chunks[i])
            self.assertEqual(results[i].chunk_index, i)

    def test_batching_exact_multiple(self) -> None:
        """T4-14: Tests 128 chunks with batch_size=64; asserts exactly 2 client calls of 64 items."""
        spy_client = SpyMockEmbeddingClient()
        pipeline = EmbeddingPipeline(client=spy_client, config=EmbeddingConfig(batch_size=64))

        chunks = [self._create_chunk(self.doc_a, self.org_a, i) for i in range(128)]
        results = pipeline.embed_chunks(chunks)

        self.assertEqual(len(results), 128)
        self.assertEqual(len(spy_client.call_history), 2)
        self.assertEqual(len(spy_client.call_history[0]), 64)
        self.assertEqual(len(spy_client.call_history[1]), 64)

    def test_batching_partial_remainder(self) -> None:
        """T4-15: Tests 70 chunks with batch_size=64; asserts 2 client calls of 64 and 6 items."""
        spy_client = SpyMockEmbeddingClient()
        pipeline = EmbeddingPipeline(client=spy_client, config=EmbeddingConfig(batch_size=64))

        chunks = [self._create_chunk(self.doc_a, self.org_a, i) for i in range(70)]
        results = pipeline.embed_chunks(chunks)

        self.assertEqual(len(results), 70)
        self.assertEqual(len(spy_client.call_history), 2)
        self.assertEqual(len(spy_client.call_history[0]), 64)
        self.assertEqual(len(spy_client.call_history[1]), 6)

    def test_strict_order_preservation_interleaved(self) -> None:
        """T4-16: Passes interleaved chunks [A0, B0, A1, B1, A2, B2]; asserts 100% strict 1:1 input order match."""
        c_a0 = self._create_chunk(self.doc_a, self.org_a, 0)
        c_b0 = self._create_chunk(self.doc_b, self.org_b, 0)
        c_a1 = self._create_chunk(self.doc_a, self.org_a, 1)
        c_b1 = self._create_chunk(self.doc_b, self.org_b, 1)
        c_a2 = self._create_chunk(self.doc_a, self.org_a, 2)
        c_b2 = self._create_chunk(self.doc_b, self.org_b, 2)

        interleaved = [c_a0, c_b0, c_a1, c_b1, c_a2, c_b2]
        results = self.pipeline.embed_chunks(interleaved)

        self.assertEqual(len(results), 6)
        for idx in range(6):
            self.assertEqual(
                results[idx].chunk.chunk_id,
                interleaved[idx].chunk_id,
                msg=f"Positional order mismatch at index {idx}",
            )
            self.assertIs(results[idx].chunk, interleaved[idx])

    def test_tenant_isolation_batching(self) -> None:
        """T4-17: Inspects client call history; asserts 0 batches contain mixed Org A and Org B texts."""
        spy_client = SpyMockEmbeddingClient()
        pipeline = EmbeddingPipeline(client=spy_client, config=EmbeddingConfig(batch_size=4))

        # 5 chunks of Org A, 5 chunks of Org B in alternating sequence
        chunks: List[DocumentChunk] = []
        for i in range(5):
            chunks.append(self._create_chunk(self.doc_a, self.org_a, i, content=f"ORGA content item {i}"))
            chunks.append(self._create_chunk(self.doc_b, self.org_b, i, content=f"ORGB content item {i}"))

        results = pipeline.embed_chunks(chunks)
        self.assertEqual(len(results), 10)

        # Verify call history
        self.assertTrue(len(spy_client.call_history) >= 2)
        for batch_idx, batch in enumerate(spy_client.call_history):
            has_org_a = any("ORGA" in text for text in batch)
            has_org_b = any("ORGB" in text for text in batch)
            self.assertFalse(
                has_org_a and has_org_b,
                msg=f"Batch at index {batch_idx} violates tenant isolation: contains mixed Org A and Org B texts!",
            )

    def test_metadata_preservation(self) -> None:
        """Verifies full fidelity of metadata propagation across EmbeddedChunk database contract properties."""
        chunk = self._create_chunk(
            doc_id="99999999-9999-4999-8999-999999999999",
            org_id="88888888-8888-4888-8888-888888888888",
            chunk_index=3,
            content="[Document: SLA Policy > Response Times]\n\nGuaranteed within 1 hour.",
            access_level="internal_staff",
        )
        results = self.pipeline.embed_chunks([chunk])
        self.assertEqual(len(results), 1)
        res = results[0]

        self.assertEqual(res.id, chunk.chunk_id)
        self.assertEqual(res.chunk_id, chunk.chunk_id)
        self.assertEqual(res.document_id, "99999999-9999-4999-8999-999999999999")
        self.assertEqual(res.org_id, "88888888-8888-4888-8888-888888888888")
        self.assertEqual(res.access_level, "internal_staff")
        self.assertEqual(res.chunk_index, 3)
        self.assertEqual(res.title, "Doc 99999999")
        self.assertEqual(res.content, "[Document: SLA Policy > Response Times]\n\nGuaranteed within 1 hour.")

    def test_query_embedding_pipeline(self) -> None:
        """T4-10: Verifies embed_query validates input, calls client, and returns 384-d tuple matching direct client."""
        query = "How do I configure OAuth2 single sign-on?"
        vector = self.pipeline.embed_query(query)

        self.assertIsInstance(vector, tuple)
        self.assertEqual(len(vector), 384)
        self.assertTrue(all(isinstance(x, float) and math.isfinite(x) for x in vector))

        # Check unit norm
        l2_norm = math.sqrt(sum(x * x for x in vector))
        self.assertAlmostEqual(l2_norm, 1.0, places=5)

        # Matches direct client call
        direct_vector = self.client.embed_text(query)
        self.assertEqual(vector, direct_vector)

    def test_input_validation_empty_and_whitespace(self) -> None:
        """T4-11: Verifies empty/whitespace chunks or queries raise EmbeddingValidationError."""
        # Empty and whitespace queries
        with self.assertRaises(EmbeddingValidationError):
            self.pipeline.embed_query("")

        with self.assertRaises(EmbeddingValidationError):
            self.pipeline.embed_query("   \n\t  ")

        with self.assertRaises(EmbeddingValidationError):
            self.pipeline.embed_query(12345)  # type: ignore

        # Chunk with empty or whitespace-only content
        empty_chunk = self._create_chunk(self.doc_a, self.org_a, 0, content="")
        # Force empty content via object.__setattr__ since DocumentChunk is frozen
        object.__setattr__(empty_chunk, "content", "")
        with self.assertRaises(EmbeddingValidationError):
            self.pipeline.embed_chunks([empty_chunk])

        whitespace_chunk = self._create_chunk(self.doc_a, self.org_a, 1, content="   \t\n  ")
        object.__setattr__(whitespace_chunk, "content", "   \t\n  ")
        with self.assertRaises(EmbeddingValidationError):
            self.pipeline.embed_chunks([whitespace_chunk])

    def test_input_validation_non_list_and_non_chunk(self) -> None:
        """Verifies passing non-list or list with non-DocumentChunk items raises EmbeddingValidationError."""
        with self.assertRaises(EmbeddingValidationError):
            self.pipeline.embed_chunks("not-a-list")  # type: ignore

        with self.assertRaises(EmbeddingValidationError):
            self.pipeline.embed_chunks(None)  # type: ignore

        valid_chunk = self._create_chunk(self.doc_a, self.org_a, 0)
        with self.assertRaises(EmbeddingValidationError):
            self.pipeline.embed_chunks([valid_chunk, "not-a-chunk"])  # type: ignore

    def test_output_validation_wrong_dim(self) -> None:
        """T4-12: Injects mock client returning != 384 dimensions; asserts EmbeddingValidationError."""
        mismatch_client = DimensionMismatchEmbeddingClient(dimensions=384, return_dim=128)
        pipeline = EmbeddingPipeline(client=mismatch_client)

        chunk = self._create_chunk(self.doc_a, self.org_a, 0)
        with self.assertRaises(EmbeddingValidationError):
            pipeline.embed_chunks([chunk])

        with self.assertRaises(EmbeddingValidationError):
            pipeline.embed_query("Sample query")

    def test_output_validation_nan_inf(self) -> None:
        """T4-13: Injects vector containing float('nan') or float('inf'); asserts EmbeddingValidationError."""
        nan_client = NonFiniteEmbeddingClient(dimensions=384, inject_nan=True)
        nan_pipeline = EmbeddingPipeline(client=nan_client)

        chunk = self._create_chunk(self.doc_a, self.org_a, 0)
        with self.assertRaises(EmbeddingValidationError):
            nan_pipeline.embed_chunks([chunk])

        with self.assertRaises(EmbeddingValidationError):
            nan_pipeline.embed_query("Sample query")

        inf_client = NonFiniteEmbeddingClient(dimensions=384, inject_nan=False)
        inf_pipeline = EmbeddingPipeline(client=inf_client)

        with self.assertRaises(EmbeddingValidationError):
            inf_pipeline.embed_chunks([chunk])

        with self.assertRaises(EmbeddingValidationError):
            inf_pipeline.embed_query("Sample query")

    def test_client_failure_propagation(self) -> None:
        """Verifies client inference failure propagates directly without returning partial results."""
        failing_client = FailingEmbeddingClient()
        pipeline = EmbeddingPipeline(client=failing_client)

        chunk = self._create_chunk(self.doc_a, self.org_a, 0)
        with self.assertRaises(EmbeddingInferenceError):
            pipeline.embed_chunks([chunk])

        with self.assertRaises(EmbeddingInferenceError):
            pipeline.embed_query("Sample query")

    def test_query_oversized_token_ceiling(self) -> None:
        """Verifies search query exceeding maximum sequence length raises EmbeddingValidationError."""
        # Max sequence length is 256; create a query with ~300 words (>350 tokens)
        oversized_query = "word " * 300
        with self.assertRaises(EmbeddingValidationError) as ctx:
            self.pipeline.embed_query(oversized_query)
        self.assertIn("exceeds maximum sequence token limit", str(ctx.exception))


# -----------------------------------------------------------------------------
# Phase 5 Full-Corpus Smoke Verification Suite
# -----------------------------------------------------------------------------


class TestM4Phase5CorpusSmoke(unittest.TestCase):
    """Full-corpus smoke verification suite for Milestone M4 Phase 5."""

    def test_stage0_raw_corpus_smoke_verification(self) -> None:
        """Loads all 10 raw documents, chunks them (113 chunks), and passes through EmbeddingPipeline.

        Verifies:
        1. Discovers 10 documents under data/raw/ (5 in Org A, 5 in Org B).
        2. Generates exactly 113 valid chunks (56 Org A, 57 Org B).
        3. Generates exactly 113 EmbeddedChunk objects (1:1 count correspondence).
        4. Strict 1:1 original positional order preservation.
        5. Zero cross-tenant batch mixing (100% tenant batch isolation).
        6. Exactly 384 dimensions for all generated vectors.
        7. 100% finite float elements (no NaN or Inf).
        8. L2 unit normalization within 10^-5 tolerance.
        9. 100% metadata fidelity to source DocumentChunk.
        10. 100% bitwise determinism across repeated executions.
        11. Total execution latency < 0.50s.
        """
        raw_files = sorted(glob.glob("data/raw/**/*.md", recursive=True))
        if not raw_files:
            self.skipTest("data/raw/ directory not available or empty; skipping corpus smoke test.")

        # Document count assertion
        self.assertEqual(len(raw_files), 10)
        org_a_files = [f for f in raw_files if "/org_a/" in f]
        org_b_files = [f for f in raw_files if "/org_b/" in f]
        self.assertEqual(len(org_a_files), 5)
        self.assertEqual(len(org_b_files), 5)

        # Chunk all documents
        org_a_chunks: List[DocumentChunk] = []
        for p in org_a_files:
            content = Path(p).read_text(encoding="utf-8")
            org_a_chunks.extend(chunk_document(content))

        org_b_chunks: List[DocumentChunk] = []
        for p in org_b_files:
            content = Path(p).read_text(encoding="utf-8")
            org_b_chunks.extend(chunk_document(content))

        self.assertEqual(len(org_a_chunks), 56)
        self.assertEqual(len(org_b_chunks), 57)
        all_chunks = org_a_chunks + org_b_chunks
        self.assertEqual(len(all_chunks), 113)

        # Configure pipeline with SpyClient
        spy_client = SpyMockEmbeddingClient(dimensions=384)
        pipeline = EmbeddingPipeline(client=spy_client, config=EmbeddingConfig(batch_size=64))

        start_time = time.perf_counter()
        embedded_chunks = pipeline.embed_chunks(all_chunks)
        duration = time.perf_counter() - start_time

        # 1. Output count match
        self.assertEqual(len(embedded_chunks), 113)

        # 2. Positional order preservation
        for idx in range(113):
            self.assertIs(embedded_chunks[idx].chunk, all_chunks[idx])
            self.assertEqual(embedded_chunks[idx].id, all_chunks[idx].chunk_id)

        # 3. Tenant batch separation
        self.assertEqual(len(spy_client.call_history), 2)  # 56 in batch 0, 57 in batch 1
        org_a_texts = {c.content for c in org_a_chunks}
        org_b_texts = {c.content for c in org_b_chunks}

        batch_0_set = set(spy_client.call_history[0])
        batch_1_set = set(spy_client.call_history[1])

        self.assertTrue(batch_0_set.issubset(org_a_texts))
        self.assertTrue(batch_1_set.issubset(org_b_texts))

        # 4. Vector validation (384-d, finite, unit norm)
        for ec in embedded_chunks:
            self.assertEqual(len(ec.embedding), 384)
            self.assertTrue(all(isinstance(x, float) and math.isfinite(x) for x in ec.embedding))
            l2_norm = math.sqrt(sum(x * x for x in ec.embedding))
            self.assertAlmostEqual(l2_norm, 1.0, places=5)

        # 5. Metadata fidelity
        for ec, c in zip(embedded_chunks, all_chunks):
            self.assertEqual(ec.chunk_id, c.chunk_id)
            self.assertEqual(ec.document_id, c.document_id)
            self.assertEqual(ec.org_id, c.org_id)
            self.assertEqual(ec.access_level, c.access_level)
            self.assertEqual(ec.chunk_index, c.chunk_index)
            self.assertEqual(ec.title, c.title)
            self.assertEqual(ec.content, c.content)

        # 6. Determinism across repeated runs
        re_embedded = pipeline.embed_chunks(all_chunks)
        for ec1, ec2 in zip(embedded_chunks, re_embedded):
            self.assertEqual(ec1.embedding, ec2.embedding)

        # 7. Performance latency (< 0.50s)
        self.assertLess(duration, 0.50, f"Corpus embedding took {duration:.4f}s, expected < 0.50s")

    def test_interleaved_corpus_tenant_isolation_and_order(self) -> None:
        """Verifies interleaved corpus chunks (A0, B0, A1, B1...) preserve order and strict batch isolation."""
        raw_files = sorted(glob.glob("data/raw/**/*.md", recursive=True))
        if not raw_files:
            self.skipTest("data/raw/ directory not available or empty; skipping corpus smoke test.")

        org_a_files = [f for f in raw_files if "/org_a/" in f]
        org_b_files = [f for f in raw_files if "/org_b/" in f]

        org_a_chunks: List[DocumentChunk] = []
        for p in org_a_files:
            org_a_chunks.extend(chunk_document(Path(p).read_text(encoding="utf-8")))

        org_b_chunks: List[DocumentChunk] = []
        for p in org_b_files:
            org_b_chunks.extend(chunk_document(Path(p).read_text(encoding="utf-8")))

        interleaved: List[DocumentChunk] = []
        max_len = max(len(org_a_chunks), len(org_b_chunks))
        for i in range(max_len):
            if i < len(org_a_chunks):
                interleaved.append(org_a_chunks[i])
            if i < len(org_b_chunks):
                interleaved.append(org_b_chunks[i])

        self.assertEqual(len(interleaved), 113)

        spy_client = SpyMockEmbeddingClient(dimensions=384)
        pipeline = EmbeddingPipeline(client=spy_client, config=EmbeddingConfig(batch_size=16))

        results = pipeline.embed_chunks(interleaved)

        # 1. Output count match
        self.assertEqual(len(results), 113)

        # 2. Strict 1:1 order preservation
        for idx in range(113):
            self.assertIs(results[idx].chunk, interleaved[idx])

        # 3. Batch tenant isolation across 8 batches
        self.assertEqual(len(spy_client.call_history), 8)
        org_a_texts = {c.content for c in org_a_chunks}
        org_b_texts = {c.content for c in org_b_chunks}

        for b_idx, batch in enumerate(spy_client.call_history):
            is_org_a = all(t in org_a_texts for t in batch)
            is_org_b = all(t in org_b_texts for t in batch)
            self.assertTrue(
                is_org_a or is_org_b,
                f"Batch {b_idx} contains mixed tenant texts!",
            )


if __name__ == "__main__":
    unittest.main()

"""Milestone M5: Real Embedding Inference Integration Test Suite.

Grounded in:
- docs/plans/05-real-embedding-inference-plan.md
- docs/specs/SPEC-05-real-embedding-inference.md
- Shared Team Contract v3.0 Locked & RFC-01 (sentence-transformers/all-MiniLM-L6-v2, 384 dimensions)
"""

from __future__ import annotations

import glob
import logging
import math
from pathlib import Path
import resource
import time
from typing import List, Tuple
import unittest

from src.rag.chunking import chunk_document
from src.rag.chunking.models import DocumentChunk
from src.rag.embeddings.client import (
    HAS_SENTENCE_TRANSFORMERS,
    SentenceTransformerEmbeddingClient,
)
from src.rag.embeddings.models import (
    DEFAULT_MODEL_NAME,
    EmbeddedChunk,
    EmbeddingConfig,
    EmbeddingConfigError,
    EmbeddingInferenceError,
    EmbeddingValidationError,
)
from src.rag.embeddings.pipeline import EmbeddingPipeline

logger = logging.getLogger(__name__)


def _is_model_cached(model_name: str) -> bool:
    """Check whether model weights are already cached locally without initiating network downloads."""
    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(model_name, local_files_only=True)
        return True
    except Exception:
        return False


IS_MODEL_CACHED = _is_model_cached(DEFAULT_MODEL_NAME) if HAS_SENTENCE_TRANSFORMERS else False


class TrackingRealEmbeddingClient(SentenceTransformerEmbeddingClient):
    """Test spy subclass recording batch calls for tenant isolation verification."""

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        local_files_only: bool = True,
    ) -> None:
        super().__init__(config=config, local_files_only=local_files_only)
        self.batch_calls: List[List[str]] = []

    def embed_texts(self, texts: List[str]) -> List[Tuple[float, ...]]:
        self.batch_calls.append(list(texts))
        return super().embed_texts(texts)


@unittest.skipUnless(
    HAS_SENTENCE_TRANSFORMERS and IS_MODEL_CACHED,
    "sentence-transformers not installed or model weights not cached locally; "
    "skipping real-model integration tests to prevent automatic network downloads.",
)
class TestM5RealInference(unittest.TestCase):
    """Integration test suite for Milestone M5 Real Embedding Inference."""

    def setUp(self) -> None:
        """Initialize real embedding client with local-only enforcement."""
        self.client = SentenceTransformerEmbeddingClient(local_files_only=True)
        self.pipeline = EmbeddingPipeline(client=self.client)

    def test_real_client_lazy_loading(self) -> None:
        """T5-01: Verifies lazy model loading lifecycle and explicit unload."""
        client = SentenceTransformerEmbeddingClient(local_files_only=True)
        self.assertFalse(client.is_loaded, "Model should not be loaded upon instantiation.")
        self.assertEqual(client.dimensions, 384)
        self.assertEqual(client.model_name, DEFAULT_MODEL_NAME)

        # Explicit load
        client.load_model()
        self.assertTrue(client.is_loaded, "Model should be loaded after explicit load_model().")

        # Explicit unload
        client.unload_model()
        self.assertFalse(client.is_loaded, "Model should not be loaded after unload_model().")

        # Implicit lazy load on inference
        vec = client.embed_text("Lazy loading integration verification text.")
        self.assertTrue(client.is_loaded, "Model should be loaded after embed_text() call.")
        self.assertEqual(len(vec), 384)

    def test_real_client_dimensions(self) -> None:
        """T5-02: Verifies output vector geometry across 1, 3, and 10 inputs."""
        test_cases = [
            ["Single text vector dimension check."],
            [
                "First sentence for dimension verification.",
                "Second sentence checking vector length.",
                "Third distinct sentence in test batch.",
            ],
            [f"Numbered test phrase {i} for dimensional validation." for i in range(10)],
        ]

        for batch in test_cases:
            embeddings = self.client.embed_texts(batch)
            self.assertEqual(len(embeddings), len(batch))
            for vec in embeddings:
                self.assertIsInstance(vec, tuple)
                self.assertEqual(len(vec), 384)
                self.assertEqual(self.client.dimensions, 384)

    def test_real_client_unit_normalization(self) -> None:
        """T5-03: Verifies mathematical L2 unit normalization (|v|_2 = 1.0 +/- 10^-4)."""
        texts = [f"Mathematical normalization verification sentence index {i}." for i in range(10)]
        embeddings = self.client.embed_texts(texts)

        for idx, vec in enumerate(embeddings):
            self.assertTrue(
                all(isinstance(x, float) and math.isfinite(x) for x in vec),
                f"Vector at index {idx} contains non-finite or non-float values.",
            )
            l2_norm = math.sqrt(sum(x * x for x in vec))
            self.assertAlmostEqual(
                l2_norm,
                1.0,
                places=4,
                msg=f"Vector {idx} L2 norm {l2_norm:.6f} deviated beyond 10^-4 tolerance.",
            )

    def test_real_semantic_relative_ranking(self) -> None:
        """T5-04: Verifies semantic relative-ranking (sim_rel > sim_unrel) with telemetry logging."""
        query = "How do I request a refund for an invoice?"
        related_target = "Customer refund and reimbursement policy guide and submission steps"
        unrelated_target = (
            "PostgreSQL database write-ahead log archiving and checkpoint configuration"
        )

        vq = self.client.embed_text(query)
        v_rel = self.client.embed_text(related_target)
        v_unrel = self.client.embed_text(unrelated_target)

        # Dot product equals cosine similarity because vectors are L2-normalized
        sim_rel = sum(x * y for x, y in zip(vq, v_rel))
        sim_unrel = sum(x * y for x, y in zip(vq, v_unrel))
        margin = sim_rel - sim_unrel

        # Telemetry logging
        print(
            f"\n[Telemetry T5-04] Semantic Similarity: "
            f"sim_rel={sim_rel:.4f}, sim_unrel={sim_unrel:.4f}, margin={margin:.4f}"
        )

        # Relative ranking correctness invariant
        self.assertGreater(
            sim_rel,
            sim_unrel,
            f"Expected sim_rel ({sim_rel:.4f}) > sim_unrel ({sim_unrel:.4f})",
        )

    def test_real_query_document_consistency(self) -> None:
        """T5-05: Verifies client.embed_text and pipeline.embed_query yield identical vectors."""
        query = "How do I configure Single Sign-On (SSO) with SAML 2.0 identity providers?"
        v_client = self.client.embed_text(query)
        v_pipeline = self.pipeline.embed_query(query)

        self.assertEqual(
            v_client,
            v_pipeline,
            "Direct client embedding and pipeline query embedding must produce bitwise identical vectors.",
        )

    def test_real_batching_consistency(self) -> None:
        """T5-06: Verifies encoding a batch of 5 texts yields equivalent vectors to individual calls."""
        texts = [
            "Billing account settings and payment methods.",
            "SAML SSO configuration for enterprise organizations.",
            "Role-based access control and user permission management.",
            "Data export and retention schedule procedures.",
            "API rate limits and authentication headers.",
        ]

        batch_embeddings = self.client.embed_texts(texts)
        individual_embeddings = [self.client.embed_text(t) for t in texts]

        self.assertEqual(len(batch_embeddings), len(texts))
        for t_idx, (b_vec, ind_vec) in enumerate(zip(batch_embeddings, individual_embeddings)):
            self.assertEqual(len(b_vec), 384)
            self.assertEqual(len(ind_vec), 384)
            for d_idx, (b_val, ind_val) in enumerate(zip(b_vec, ind_vec)):
                self.assertAlmostEqual(
                    b_val,
                    ind_val,
                    places=5,
                    msg=f"Batch vs individual discrepancy at text {t_idx}, dimension {d_idx}: "
                    f"{b_val} != {ind_val}",
                )

    def test_tokenizer_truncation_warning_and_privacy(self) -> None:
        """T5-07: Verifies structured content-free warning emitted on oversized bulk ingestion."""
        # Create oversized text (~350 words, >350 WordPiece tokens, model limit 256)
        oversized_text = "word " * 350

        captured_records: List[logging.LogRecord] = []

        class CapturingHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured_records.append(record)

        client_logger = logging.getLogger("src.rag.embeddings.client")
        original_level = client_logger.level
        handler = CapturingHandler()
        client_logger.setLevel(logging.WARNING)
        client_logger.addHandler(handler)

        try:
            results = self.client.embed_texts([oversized_text])
            self.assertEqual(len(results), 1)
            self.assertEqual(len(results[0]), 384)

            # Assert warning was captured
            truncation_records = [
                r for r in captured_records if getattr(r, "category", None) == "token_truncation_warning"
            ]
            self.assertEqual(
                len(truncation_records),
                1,
                "Expected exactly one structured token_truncation_warning log record.",
            )

            record = truncation_records[0]
            self.assertGreater(
                record.actual_tokens,
                256,
                "Warning actual_tokens must reflect token count exceeding max_seq_length.",
            )
            self.assertEqual(record.max_seq_length, 256)
            self.assertEqual(record.batch_index, 0)

            # Privacy audit: Ensure raw document content does not appear in log message or record
            self.assertNotIn("word", record.getMessage())
            self.assertFalse(hasattr(record, "text"))
            self.assertFalse(hasattr(record, "content"))
        finally:
            client_logger.removeHandler(handler)
            client_logger.setLevel(original_level)

    def test_real_model_error_handling(self) -> None:
        """T5-08: Verifies validation and inference error handling on real client."""
        # Empty and whitespace inputs
        with self.assertRaises(EmbeddingValidationError):
            self.client.embed_text("")
        with self.assertRaises(EmbeddingValidationError):
            self.client.embed_text("   \n\t  ")
        with self.assertRaises(EmbeddingValidationError):
            self.client.embed_text(123)  # type: ignore[arg-type]

        # Oversized search query rejection in both client and pipeline
        oversized_query = "oversized " * 300
        with self.assertRaises(EmbeddingValidationError) as ctx_client:
            self.client.embed_text(oversized_query)
        self.assertIn("exceeds maximum sequence token limit", str(ctx_client.exception))

        with self.assertRaises(EmbeddingValidationError) as ctx_pipe:
            self.pipeline.embed_query(oversized_query)
        self.assertIn("exceeds maximum sequence token limit", str(ctx_pipe.exception))

        # Malformed batch lists
        with self.assertRaises(EmbeddingValidationError):
            self.client.embed_texts("not-a-list")  # type: ignore[arg-type]
        with self.assertRaises(EmbeddingValidationError):
            self.client.embed_texts(["valid text", 12345])  # type: ignore[list-item]
        with self.assertRaises(EmbeddingValidationError):
            self.client.embed_texts(["valid text", "   "])

        # Token count input validation
        with self.assertRaises(EmbeddingValidationError):
            self.client.count_tokens(123)  # type: ignore[arg-type]

        # Config validation
        with self.assertRaises(EmbeddingConfigError):
            SentenceTransformerEmbeddingClient(config="invalid-config")  # type: ignore[arg-type]
        with self.assertRaises(EmbeddingConfigError):
            SentenceTransformerEmbeddingClient(local_files_only="yes")  # type: ignore[arg-type]

        # Missing model weights in local cache
        missing_client = SentenceTransformerEmbeddingClient(
            config=EmbeddingConfig(model_name="/tmp/nonexistent_model_dir_xyz"),
            local_files_only=True,
        )
        with self.assertRaises(EmbeddingInferenceError):
            missing_client.load_model()

    def test_stage0_corpus_real_inference_benchmark(self) -> None:
        """T5-09: Full-corpus real inference benchmark across all 113 raw chunks.

        Enforces hard correctness invariant gates and records observational telemetry.
        """
        raw_files = sorted(glob.glob("data/raw/**/*.md", recursive=True))
        if not raw_files:
            self.skipTest("data/raw/ directory not available or empty; skipping benchmark.")

        # Document count verification
        self.assertEqual(len(raw_files), 10)
        org_a_files = [f for f in raw_files if "/org_a/" in f]
        org_b_files = [f for f in raw_files if "/org_b/" in f]
        self.assertEqual(len(org_a_files), 5)
        self.assertEqual(len(org_b_files), 5)

        # Chunk all documents
        org_a_chunks: List[DocumentChunk] = []
        for p in org_a_files:
            org_a_chunks.extend(chunk_document(Path(p).read_text(encoding="utf-8")))

        org_b_chunks: List[DocumentChunk] = []
        for p in org_b_files:
            org_b_chunks.extend(chunk_document(Path(p).read_text(encoding="utf-8")))

        self.assertEqual(len(org_a_chunks), 56)
        self.assertEqual(len(org_b_chunks), 57)
        all_chunks = org_a_chunks + org_b_chunks
        self.assertEqual(len(all_chunks), 113)

        # Configure pipeline with TrackingRealEmbeddingClient
        tracking_client = TrackingRealEmbeddingClient(
            config=EmbeddingConfig(batch_size=64),
            local_files_only=True,
        )
        benchmark_pipeline = EmbeddingPipeline(
            client=tracking_client,
            config=EmbeddingConfig(batch_size=64),
        )

        # Pre-warm model to isolate inference time from first-time weight loading
        tracking_client.load_model()

        # Capture resource usage and wall-clock execution time
        start_time = time.perf_counter()
        embedded_chunks: List[EmbeddedChunk] = benchmark_pipeline.embed_chunks(all_chunks)
        duration = time.perf_counter() - start_time

        # Memory usage via resource (ru_maxrss is KB on Linux)
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        peak_rss_mb = rusage.ru_maxrss / 1024.0

        # Observational telemetry
        ms_per_chunk = (duration / len(all_chunks)) * 1000.0
        chunks_per_sec = len(all_chunks) / duration if duration > 0 else 0.0

        print(f"\n{'=' * 65}")
        print("=== Milestone M5 Real Inference Benchmark Telemetry (T5-09) ===")
        print(f"Total Chunks Processed: {len(all_chunks)}")
        print(f"Total Wall-Clock Time:  {duration:.4f} s")
        print(f"Average Latency:        {ms_per_chunk:.2f} ms/chunk")
        print(f"Effective Throughput:   {chunks_per_sec:.2f} chunks/s")
        print(f"Peak Resident Memory:   {peak_rss_mb:.2f} MB RSS")
        print(f"{'=' * 65}")

        # --- Hard Correctness Invariant Gates (Pass/Fail) ---
        # 1. Output count match
        self.assertEqual(len(embedded_chunks), 113)

        # 2. Strict 1:1 positional order preservation
        for idx in range(113):
            self.assertIs(embedded_chunks[idx].chunk, all_chunks[idx])
            self.assertEqual(embedded_chunks[idx].id, all_chunks[idx].chunk_id)

        # 3. Metadata fidelity
        for ec, c in zip(embedded_chunks, all_chunks):
            self.assertEqual(ec.chunk_id, c.chunk_id)
            self.assertEqual(ec.document_id, c.document_id)
            self.assertEqual(ec.org_id, c.org_id)
            self.assertEqual(ec.access_level, c.access_level)
            self.assertEqual(ec.chunk_index, c.chunk_index)
            self.assertEqual(ec.title, c.title)
            self.assertEqual(ec.content, c.content)

        # 4. Strict tenant batch isolation
        self.assertEqual(len(tracking_client.batch_calls), 2)
        org_a_texts = {c.content for c in org_a_chunks}
        org_b_texts = {c.content for c in org_b_chunks}

        batch_0_texts = set(tracking_client.batch_calls[0])
        batch_1_texts = set(tracking_client.batch_calls[1])

        self.assertTrue(
            batch_0_texts.issubset(org_a_texts),
            "Batch 0 contains texts outside Org A.",
        )
        self.assertTrue(
            batch_1_texts.issubset(org_b_texts),
            "Batch 1 contains texts outside Org B.",
        )

        # 5. Output dimensionality strictly 384
        for ec in embedded_chunks:
            self.assertEqual(len(ec.embedding), 384)

        # 6. All vector elements are finite Python floats
        for ec in embedded_chunks:
            self.assertTrue(all(isinstance(x, float) and math.isfinite(x) for x in ec.embedding))

        # 7. Mathematical L2 unit normalization (|v|_2 = 1.0 +/- 10^-4)
        for ec in embedded_chunks:
            l2_norm = math.sqrt(sum(x * x for x in ec.embedding))
            self.assertAlmostEqual(l2_norm, 1.0, places=4)

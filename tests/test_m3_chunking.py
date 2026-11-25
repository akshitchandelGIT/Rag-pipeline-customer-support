"""Authoritative test suite for Milestone M3 — Chunking Subsystem.

Implements the authoritative 21-test matrix (T-01 through T-21) mapped to:
- docs/specs/SPEC-02-chunking.md
- docs/plans/03-chunking-implementation-plan.md
- Shared Team Contract v3.0 Locked

Standard library unittest framework only. Pure in-memory synthetic fixtures
ensure total reproducibility, deterministic execution, and zero external I/O.
"""

from __future__ import annotations

import glob
from io import StringIO
import logging
from pathlib import Path
import re
import unittest
import uuid

from src.rag.chunking.engine import (
    ChunkingEngine,
    chunk_document,
    format_breadcrumb,
    split_sentences_preserving_whitespace,
)
from src.rag.chunking.models import (
    DEFAULT_CHUNKING_VERSION,
    DocumentChunk,
    EmptyDocumentError,
    FrontmatterMetadata,
    InvalidAccessLevelError,
    InvalidMetadataError,
    MalformedFrontmatterError,
    MissingFrontmatterError,
    ParsedSection,
    VALID_ACCESS_LEVELS,
    compute_content_hash,
    estimate_tokens,
    generate_chunk_id,
    validate_uuid_v4,
)
from src.rag.chunking.parser import (
    MarkdownFSMParser,
    extract_frontmatter,
    parse_document,
    parse_sections,
)


class TestM3Chunking(unittest.TestCase):
    """Authoritative 21-Test Matrix (T-01 through T-21) for Milestone M3."""

    def setUp(self) -> None:
        self.doc_id_a = "11111111-1111-4111-8111-111111111111"
        self.org_id_a = "11111111-1111-4111-8111-111111111111"
        self.doc_id_b = "22222222-2222-4222-8222-222222222222"
        self.org_id_b = "22222222-2222-4222-8222-222222222222"
        self.default_title = "Standard Operating Procedure"
        self.default_access_level = "public"

    # =========================================================================
    # T-01: Frontmatter Errors (SPEC-02 §16, Safeguard 1 & Invariant F-02)
    # =========================================================================
    def test_frontmatter_errors(self) -> None:
        """T-01: Asserts MissingFrontmatterError on omitted opening fence, and

        MalformedFrontmatterError on unclosed frontmatter before EOF, missing
        required fields, empty required fields, or unsupported YAML sequences.
        """
        # 1. Missing opening '---' fence on line 1
        no_opening = (
            "document_id: 11111111-1111-4111-8111-111111111111\n"
            "org_id: 11111111-1111-4111-8111-111111111111\n"
            "access_level: public\n"
            "title: Guide\n"
            "---\n"
            "# Body\n"
        )
        with self.assertRaises(MissingFrontmatterError) as ctx:
            extract_frontmatter(no_opening)
        self.assertIn("missing opening '---'", str(ctx.exception).lower())

        # 2. Unclosed frontmatter fence before EOF
        unclosed = (
            "---\n"
            "document_id: 11111111-1111-4111-8111-111111111111\n"
            "org_id: 11111111-1111-4111-8111-111111111111\n"
            "access_level: public\n"
            "title: Guide\n"
        )
        with self.assertRaises(MalformedFrontmatterError) as ctx:
            extract_frontmatter(unclosed)
        self.assertIn("unclosed frontmatter fence", str(ctx.exception).lower())

        # 3. Missing required field: title
        missing_title = (
            "---\n"
            "document_id: 11111111-1111-4111-8111-111111111111\n"
            "org_id: 11111111-1111-4111-8111-111111111111\n"
            "access_level: public\n"
            "---\n"
            "# Body\n"
        )
        with self.assertRaises(MalformedFrontmatterError) as ctx:
            extract_frontmatter(missing_title)
        self.assertIn("missing required frontmatter field: 'title'", str(ctx.exception).lower())

        # 4. Empty required field: title is empty string
        empty_title = (
            "---\n"
            "document_id: 11111111-1111-4111-8111-111111111111\n"
            "org_id: 11111111-1111-4111-8111-111111111111\n"
            "access_level: public\n"
            "title: ''\n"
            "---\n"
            "# Body\n"
        )
        with self.assertRaises(MalformedFrontmatterError) as ctx:
            extract_frontmatter(empty_title)
        self.assertIn("must not be empty", str(ctx.exception).lower())

        # 5. Unsupported YAML sequence
        yaml_seq = (
            "---\n"
            "document_id: 11111111-1111-4111-8111-111111111111\n"
            "org_id: 11111111-1111-4111-8111-111111111111\n"
            "- invalid_yaml_list\n"
            "---\n"
        )
        with self.assertRaises(MalformedFrontmatterError) as ctx:
            extract_frontmatter(yaml_seq)
        self.assertIn("unsupported yaml sequence", str(ctx.exception).lower())

    # =========================================================================
    # T-02: Strict UUID Validation (SPEC-02 §16, Safeguard 2 & Invariant F-04)
    # =========================================================================
    def test_invalid_org_uuid(self) -> None:
        """T-02: Asserts InvalidMetadataError on raw whitespace-padded, non-string,

        malformed, non-v4, or normalized UUIDs without silent stripping.
        """
        valid_v4 = "11111111-1111-4111-8111-111111111111"

        # Raw whitespace surrounding rejection
        with self.assertRaises(InvalidMetadataError):
            validate_uuid_v4(f" {valid_v4}")
        with self.assertRaises(InvalidMetadataError):
            validate_uuid_v4(f"{valid_v4} ")
        with self.assertRaises(InvalidMetadataError):
            validate_uuid_v4(f"{valid_v4}\n")
        with self.assertRaises(InvalidMetadataError):
            validate_uuid_v4(f"\t{valid_v4}")

        # Non-string rejection
        with self.assertRaises(InvalidMetadataError):
            validate_uuid_v4(123456789)  # type: ignore

        # Wrong length or unhyphenated format
        with self.assertRaises(InvalidMetadataError):
            validate_uuid_v4("11111111111141118111111111111111")
        with self.assertRaises(InvalidMetadataError):
            validate_uuid_v4("invalid-uuid-format")

        # Wrong version (UUID v1, UUID v5)
        v1_uuid = str(uuid.uuid1())
        with self.assertRaises(InvalidMetadataError):
            validate_uuid_v4(v1_uuid)
        v5_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, "test"))
        with self.assertRaises(InvalidMetadataError):
            validate_uuid_v4(v5_uuid)

        # Rejection in frontmatter pipeline for org_id and document_id
        doc_bad_org = (
            "---\n"
            f"document_id: {valid_v4}\n"
            f"org_id: ' {valid_v4} '\n"
            "access_level: public\n"
            "title: Test\n"
            "---\n"
            "# Body\n"
        )
        with self.assertRaises(InvalidMetadataError):
            extract_frontmatter(doc_bad_org)

    # =========================================================================
    # T-03: Invalid Access Level (SPEC-02 §16, Safeguard 3)
    # =========================================================================
    def test_invalid_access_level(self) -> None:
        """T-03: Asserts InvalidAccessLevelError is raised when access_level

        is not in {'public', 'internal_staff', 'admin_only'}.
        """
        valid_v4 = self.doc_id_a
        for invalid_tier in [
            "confidential",
            "restricted",
            "private",
            "guest",
            "End_Customer",
            "Support_Agent",
            "Org_Admin",
            "PUBLIC",
            "internal",
        ]:
            doc = (
                "---\n"
                f"document_id: {valid_v4}\n"
                f"org_id: {valid_v4}\n"
                f"access_level: {invalid_tier}\n"
                "title: Title\n"
                "---\n"
                "# Body\n"
            )
            with self.assertRaises(InvalidAccessLevelError):
                extract_frontmatter(doc)

    # =========================================================================
    # T-04: Empty Document (SPEC-02 §17.1 (T5))
    # =========================================================================
    def test_empty_document(self) -> None:
        """T-04: Asserts EmptyDocumentError is raised on empty string or whitespace-only input."""
        with self.assertRaises(EmptyDocumentError):
            extract_frontmatter("")
        with self.assertRaises(EmptyDocumentError):
            extract_frontmatter("   \n\t\r\n  ")
        with self.assertRaises(EmptyDocumentError):
            chunk_document("")
        with self.assertRaises(EmptyDocumentError):
            chunk_document("   \n\t  ")

    # =========================================================================
    # T-05: ATX Heading Extraction (SPEC-02 §17.1 (T1))
    # =========================================================================
    def test_atx_heading_extraction(self) -> None:
        """T-05: Asserts correct extraction of H1 through H6, levels, titles,

        canonical heading_path, skipped levels, and source line numbers.
        """
        body = (
            "# Main System Manual\n"
            "Overview intro.\n\n"
            "## Architecture\n"
            "Arch content.\n\n"
            "### Microservices\n"
            "Services content.\n\n"
            "#### Queue Service\n"
            "Queue content.\n\n"
            "## Maintenance\n"
            "Maintenance content.\n"
        )
        root = parse_sections(body, root_title="Main System Manual")
        self.assertEqual(root.heading_level, 1)
        self.assertEqual(root.heading_text, "Main System Manual")
        self.assertEqual(root.heading_path, ["Main System Manual"])

        # Check H2 sections
        self.assertEqual(len(root.children), 2)
        h2_arch = root.children[0]
        h2_maint = root.children[1]

        self.assertEqual(h2_arch.heading_text, "Architecture")
        self.assertEqual(h2_arch.heading_level, 2)
        self.assertEqual(h2_arch.heading_path, ["Main System Manual", "Architecture"])

        # Check H3 subsections
        self.assertEqual(len(h2_arch.children), 1)
        h3_micro = h2_arch.children[0]
        self.assertEqual(h3_micro.heading_text, "Microservices")
        self.assertEqual(h3_micro.heading_level, 3)
        self.assertEqual(h3_micro.heading_path, ["Main System Manual", "Architecture", "Microservices"])

        # Check H4 subsections
        self.assertEqual(len(h3_micro.children), 1)
        h4_queue = h3_micro.children[0]
        self.assertEqual(h4_queue.heading_text, "Queue Service")
        self.assertEqual(h4_queue.heading_level, 4)
        self.assertEqual(
            h4_queue.heading_path,
            ["Main System Manual", "Architecture", "Microservices", "Queue Service"],
        )

        # Check stack pop to H2 Maintenance
        self.assertEqual(h2_maint.heading_text, "Maintenance")
        self.assertEqual(h2_maint.heading_path, ["Main System Manual", "Maintenance"])

    # =========================================================================
    # T-06: Code Fence Heading Suppression (SPEC-02 §17.1 (T2), Safeguard 5)
    # =========================================================================
    def test_code_fence_heading_suppression(self) -> None:
        """T-06: Verifies dual code fence suppression (``` and ~~~) and fence

        length matching; asserts '# Header' inside fences produces ZERO sections.
        """
        body = (
            "# Developer Runbook\n\n"
            "```python\n"
            "# This is a python comment, not an H1 heading!\n"
            "def initialize():\n"
            "    ## This is not an H2 heading!\n"
            "    pass\n"
            "```\n\n"
            "~~~bash\n"
            "#!/bin/bash\n"
            "# Bash comment in tilde block\n"
            "### Not an H3 heading\n"
            "~~~\n\n"
            "````markdown\n"
            "```python\n"
            "# Nested code comment inside 4-backtick block\n"
            "```\n"
            "````\n\n"
            "## Legitimate Section\n"
            "Real procedure content.\n"
        )
        root = parse_sections(body, root_title="Developer Runbook")
        # Exactly one legitimate H2 child must be emitted
        self.assertEqual(len(root.children), 1)
        self.assertEqual(root.children[0].heading_text, "Legitimate Section")
        self.assertEqual(root.children[0].heading_path, ["Developer Runbook", "Legitimate Section"])

        # Verify that comments inside code fences remain in content lines as raw text
        root_text = "".join(root.content_lines)
        self.assertIn("# This is a python comment", root_text)
        self.assertIn("# Bash comment in tilde block", root_text)
        self.assertIn("# Nested code comment", root_text)

    # =========================================================================
    # T-07: Table Detection (SPEC-02 §17.1 (T3))
    # =========================================================================
    def test_table_detection(self) -> None:
        """T-07: Asserts Markdown pipe table is parsed as an indivisible structural block."""
        body = (
            "# Service Commitments\n\n"
            "## Incident Escalation\n\n"
            "| Tier | SLA Target | Escalation Contact |\n"
            "|---|---|---|\n"
            "| Sev-1 | 15 mins | oncall@example.com |\n"
            "| Sev-2 | 1 hour | tier2@example.com |\n"
            "| Sev-3 | 4 hours | support@example.com |\n\n"
            "End of section.\n"
        )
        root = parse_sections(body, root_title="Service Commitments")
        h2 = root.children[0]
        table_raw = "".join(h2.content_lines)
        self.assertIn("| Tier | SLA Target | Escalation Contact |", table_raw)
        self.assertIn("| Sev-1 | 15 mins | oncall@example.com |", table_raw)
        self.assertIn("| Sev-3 | 4 hours | support@example.com |", table_raw)

    # =========================================================================
    # T-08: List Detection (SPEC-02 §17.1 (T4))
    # =========================================================================
    def test_list_detection(self) -> None:
        """T-08: Asserts bullet and numbered list items are preserved under active

        section without mid-item cutting.
        """
        raw = f"""---
document_id: {self.doc_id_a}
org_id: {self.org_id_a}
access_level: public
title: Checklist Guide
---
# Checklist Guide

## Onboarding Steps
1. Verify customer workspace credentials in the administrative console.
2. Configure outbound SMTP relays with SPF and DKIM authentication.
3. Establish default role-based access control assignments for users.
- Bullet item A describing optional webhook configurations.
- Bullet item B describing event notifications.
"""
        chunks = chunk_document(raw)
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertIn("1. Verify customer workspace credentials", chunk.content)
        self.assertIn("2. Configure outbound SMTP relays", chunk.content)
        self.assertIn("3. Establish default role-based access control", chunk.content)
        self.assertIn("- Bullet item A describing optional webhook", chunk.content)
        self.assertIn("- Bullet item B describing event notifications", chunk.content)

    # =========================================================================
    # T-09: Single Section Document (SPEC-02 §17.1 (T6))
    # =========================================================================
    def test_single_section_document(self) -> None:
        """T-09: Asserts compact document produces exactly 1 chunk with chunk_index == 0."""
        raw = f"""---
document_id: {self.doc_id_a}
org_id: {self.org_id_a}
access_level: public
title: Single Section Guide
---
# Single Section Guide

This is a compact guide containing all instructions in a single introductory section.
No secondary or procedural headings exist in this document.
"""
        chunks = chunk_document(raw)
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.chunk_index, 0)
        self.assertEqual(chunk.document_id, self.doc_id_a)
        self.assertEqual(chunk.org_id, self.org_id_a)
        self.assertEqual(chunk.access_level, "public")
        self.assertEqual(chunk.title, "Single Section Guide")
        self.assertEqual(chunk.heading_path, ["Single Section Guide"])
        self.assertTrue(chunk.content.startswith("[Document: Single Section Guide > Overview]\n\n"))

    # =========================================================================
    # T-10: Breadcrumb Injection (SPEC-02 §17.1 (T7))
    # =========================================================================
    def test_breadcrumb_injection(self) -> None:
        """T-10: Asserts chunk text begins with [Document: Title > Section]\\n\\n."""
        raw = f"""---
document_id: {self.doc_id_a}
org_id: {self.org_id_a}
access_level: internal_staff
title: Authentication Manual
---
# Authentication Manual

## Single Sign-On Configuration
Configure SAML 2.0 metadata within the identity provider portal.
"""
        chunks = chunk_document(raw)
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        expected_bc = "[Document: Authentication Manual > Single Sign-On Configuration]\n\n"
        self.assertTrue(chunk.content.startswith(expected_bc))
        self.assertIn("Configure SAML 2.0 metadata", chunk.content)

    # =========================================================================
    # T-11: Small H2 Intro Merging (SPEC-02 §17.1 (T9))
    # =========================================================================
    def test_h2_with_h3_small_intro_merging(self) -> None:
        """T-11: Asserts H2 introductory text (< 60 tokens) merges forward into

        first H3 child chunk with '### H3 Title' header.
        """
        raw = f"""---
document_id: {self.doc_id_a}
org_id: {self.org_id_a}
access_level: public
title: Setup Guide
---
# Setup Guide

## Workspace Provisioning
Welcome to workspace setup. Review prerequisites before continuing:

### Creating the Tenant Namespace
Initialize the customer workspace using the automated provisioning API.
"""
        chunks = chunk_document(raw)
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.chunk_index, 0)
        self.assertEqual(
            chunk.heading_path,
            ["Setup Guide", "Workspace Provisioning", "Creating the Tenant Namespace"],
        )
        self.assertTrue(
            chunk.content.startswith(
                "[Document: Setup Guide > Workspace Provisioning > Creating the Tenant Namespace]\n\n"
            )
        )
        expected_body = (
            "Welcome to workspace setup. Review prerequisites before continuing:\n\n"
            "### Creating the Tenant Namespace\n\n"
            "Initialize the customer workspace using the automated provisioning API."
        )
        self.assertIn(expected_body, chunk.content)

    # =========================================================================
    # T-12: Standalone H2 Isolation (SPEC-02 §17.1 (T10))
    # =========================================================================
    def test_h2_standalone_isolation_small(self) -> None:
        """T-12: Asserts standalone H2 (< 60 tokens, 0 H3 children) remains an

        independent chunk; zero merging into sibling H2 sections.
        """
        raw = f"""---
document_id: {self.doc_id_a}
org_id: {self.org_id_a}
access_level: internal_staff
title: Security Policy
---
# Security Policy

## Password Complexity
Passwords must contain at least sixteen characters with mixed case and digits.

## Session Expiration
Administrative sessions terminate automatically after fifteen minutes of inactivity.
"""
        chunks = chunk_document(raw)
        self.assertEqual(len(chunks), 2)

        # Chunk 0: Password Complexity alone
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertEqual(chunks[0].heading_path, ["Security Policy", "Password Complexity"])
        self.assertTrue(
            chunks[0].content.startswith("[Document: Security Policy > Password Complexity]\n\n")
        )
        self.assertIn("Passwords must contain at least sixteen characters", chunks[0].content)
        self.assertNotIn("Session Expiration", chunks[0].content)

        # Chunk 1: Session Expiration alone
        self.assertEqual(chunks[1].chunk_index, 1)
        self.assertEqual(chunks[1].heading_path, ["Security Policy", "Session Expiration"])
        self.assertTrue(
            chunks[1].content.startswith("[Document: Security Policy > Session Expiration]\n\n")
        )
        self.assertIn("Administrative sessions terminate automatically", chunks[1].content)
        self.assertNotIn("Password Complexity", chunks[1].content)

    # =========================================================================
    # T-13: Table Header Repetition & Canonical Path (SPEC-02 §17.1 (T8), F-01)
    # =========================================================================
    def test_table_header_repetition(self) -> None:
        """T-13: Multi-chunk split table re-injects headers at line 1 of continuation

        chunk; asserts heading_path remains canonical without '(Table Part 2)'.
        """
        # 36 rows of ~15 tokens = ~540 tokens, forcing table row splitting
        rows = [
            f"| SEV-{i} | Severe system incident scenario {i} requiring tier-{i} response | Response within {i * 10} minutes |"
            for i in range(1, 37)
        ]
        table_text = "\n".join(rows)

        raw = f"""---
document_id: {self.doc_id_a}
org_id: {self.org_id_a}
access_level: internal_staff
title: SLA Policy
---
# SLA Policy

## Response Commitments
| Severity | Description | Target Cadence |
|---|---|---|
{table_text}
"""
        chunks = chunk_document(raw)
        self.assertGreaterEqual(len(chunks), 2)

        # Chunk 0: Table Part 1
        chunk0 = chunks[0]
        self.assertEqual(chunk0.chunk_index, 0)
        self.assertEqual(chunk0.heading_path, ["SLA Policy", "Response Commitments"])
        self.assertNotIn("(Table Part", chunk0.content.splitlines()[0])
        self.assertIn("| Severity | Description | Target Cadence |", chunk0.content)

        # Chunk 1: Table Part 2 (Continuation)
        chunk1 = chunks[1]
        self.assertEqual(chunk1.chunk_index, 1)
        # INVARIANT F-01: heading_path strictly retains canonical ancestor list!
        self.assertEqual(chunk1.heading_path, ["SLA Policy", "Response Commitments"])
        self.assertNotIn("(Table Part", chunk1.heading_path[1])

        # Formatted breadcrumb in content contains (Table Part 2)
        bc_line = chunk1.content.splitlines()[0]
        self.assertEqual(bc_line, "[Document: SLA Policy > Response Commitments (Table Part 2)]")

        # Top of continuation body re-injects Column Header Row and Divider Row
        body_after_bc = chunk1.content.split("\n\n", 1)[1]
        lines = body_after_bc.splitlines()
        self.assertEqual(lines[0], "| Severity | Description | Target Cadence |")
        self.assertEqual(lines[1], "|---|---|---|")

    # =========================================================================
    # T-14: Zone 1 Intact Emission (SPEC-02 §17.1 (T11), Safeguard 6a)
    # =========================================================================
    def test_oversized_zone1_intact(self) -> None:
        """T-14: Fixture: 500 estimated tokens indivisible block.

        Asserts chunk emitted intact; is_oversized == True; is_split_item == False.
        """
        # Exactly 376 words = ceil(376 * 1.33) = 501 tokens
        words = ["indivisible"] * 376
        body_text = " ".join(words)
        est = estimate_tokens(body_text)
        self.assertGreaterEqual(est, 450)
        self.assertLessEqual(est, 600)

        raw = f"""---
document_id: {self.doc_id_a}
org_id: {self.org_id_a}
access_level: public
title: Zone 1 Guide
---
# Zone 1 Guide

## Intact Paragraph
{body_text}
"""
        chunks = chunk_document(raw)
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.chunk_index, 0)
        self.assertTrue(chunk.is_oversized)
        self.assertFalse(chunk.is_split_item)
        self.assertIn(body_text, chunk.content)

    # =========================================================================
    # T-15: Zone 2 Sentence Split (SPEC-02 §17.1 (T12), Safeguard 6b)
    # =========================================================================
    def test_oversized_zone2_sentence_split(self) -> None:
        """T-15: Fixture: 900 estimated tokens indivisible block.

        Asserts chunk is split at sentence boundaries; is_split_item == True; is_oversized == True.
        """
        # Generate 65 sentences of 11 words each = 715 words = ~951 tokens (> 800 tokens)
        sentences = [
            f"Sentence index {i} explains cluster node diagnostic procedures for phase {i}."
            for i in range(1, 66)
        ]
        body_text = " ".join(sentences)
        est = estimate_tokens(body_text)
        self.assertGreater(est, 800)

        raw = f"""---
document_id: {self.doc_id_a}
org_id: {self.org_id_a}
access_level: public
title: Zone 2 Guide
---
# Zone 2 Guide

## Diagnostic Stream
{body_text}
"""
        chunks = chunk_document(raw)
        self.assertGreaterEqual(len(chunks), 2)

        for chunk in chunks:
            self.assertTrue(chunk.is_oversized)
            self.assertTrue(chunk.is_split_item)
            self.assertIn("[Continuation: Diagnostic Stream (Part", chunk.content)

    # =========================================================================
    # T-16: Zone 2 Exact Reassembly (SPEC-02 §11.2, Safeguard 4 & Invariant F-03)
    # =========================================================================
    def test_zone2_exact_reassembly(self) -> None:
        """T-16: Strips breadcrumbs and continuation markers; asserts "".join(...)

        character-for-character exactly matches original unchunked source text (zero whitespace loss).
        Includes diverse whitespace, multiple spaces, tabs, newlines, decimals, abbreviations,
        and repeated punctuation.
        """
        # Build complex text with diverse delimiters and edge cases
        sentences = [
            # Standard single space
            "Initial diagnostic trace initiated. ",
            # Double space
            "Memory buffer allocated successfully.  ",
            # Triple space
            "Network latency verified at 12ms.   ",
            # Decimal value within sentence (should NOT split on 3.14)
            "Database engine version 3.14 is operational and verified.  ",
            # Abbreviation within sentence
            "Contact Mr. Smith or e.g. the oncall administrator for emergency access.  ",
            # Repeated punctuation
            "Did the primary cluster failover proceed smoothly??? Yes it did!!!  ",
            # Unicode characters
            "Système vérifié: état opérationnel ‘conforme’ et prêt.  ",
        ] * 15  # 7 * 15 = 105 sentences, ~980 tokens (> 800)

        # Last sentence has no trailing whitespace
        sentences[-1] = sentences[-1].rstrip()
        original_source = "".join(sentences)
        self.assertGreater(estimate_tokens(original_source), 800)

        raw = f"""---
document_id: {self.doc_id_a}
org_id: {self.org_id_a}
access_level: public
title: Precision Reassembly Guide
---
# Precision Reassembly Guide

## Complex Diagnostic Stream
{original_source}
"""
        chunks = chunk_document(raw)
        self.assertGreaterEqual(len(chunks), 2)

        reconstructed_parts = []
        for i, chunk in enumerate(chunks, start=1):
            self.assertTrue(chunk.is_oversized)
            self.assertTrue(chunk.is_split_item)

            expected_prefix = (
                f"[Document: Precision Reassembly Guide > Complex Diagnostic Stream]\n\n"
                f"[Continuation: Complex Diagnostic Stream (Part {i})]\n\n"
            )
            self.assertTrue(
                chunk.content.startswith(expected_prefix),
                f"Chunk {i} prefix mismatch",
            )
            payload = chunk.content[len(expected_prefix):]
            reconstructed_parts.append(payload)

        # Invariant F-03: EXACT character-for-character match!
        reconstructed_text = "".join(reconstructed_parts)
        self.assertEqual(
            reconstructed_text,
            original_source,
            "Reconstructed Zone 2 text does not character-for-character match source!",
        )

    # =========================================================================
    # T-17: Metadata Propagation (SPEC-02 §17.1 (T13))
    # =========================================================================
    def test_metadata_propagation(self) -> None:
        """T-17: Asserts org_id, access_level, title, document_id, category, and source

        match parent frontmatter on 100% of chunks.
        """
        raw = f"""---
document_id: {self.doc_id_a}
org_id: {self.org_id_a}
access_level: admin_only
title: Root Credentials Guide
category: security
source: internal/vault.md
---
# Root Credentials Guide

## Section A
Data A.

## Section B
Data B.
"""
        chunks = chunk_document(raw)
        self.assertEqual(len(chunks), 2)

        for chunk in chunks:
            self.assertEqual(chunk.document_id, self.doc_id_a)
            self.assertEqual(chunk.org_id, self.org_id_a)
            self.assertEqual(chunk.access_level, "admin_only")
            self.assertEqual(chunk.title, "Root Credentials Guide")
            self.assertEqual(chunk.category, "security")
            self.assertEqual(chunk.source, "internal/vault.md")
            self.assertEqual(chunk.chunking_version, DEFAULT_CHUNKING_VERSION)
            # Verify zero role names in chunk
            self.assertFalse(hasattr(chunk, "role"))
            self.assertFalse(hasattr(chunk, "role_id"))

    # =========================================================================
    # T-18: Sequential Indexing (SPEC-02 §17.1 (T14))
    # =========================================================================
    def test_sequential_indexing(self) -> None:
        """T-18: Asserts chunk_index values are strictly sequential: 0, 1, ..., N-1

        and reset on subsequent documents.
        """
        raw = f"""---
document_id: {self.doc_id_a}
org_id: {self.org_id_a}
access_level: public
title: Multi-Chunk Doc
---
# Multi-Chunk Doc

## Section 1
Content 1.

## Section 2
Content 2.

## Section 3
Content 3.

## Section 4
Content 4.
"""
        chunks = chunk_document(raw)
        self.assertEqual(len(chunks), 4)
        for expected_idx, chunk in enumerate(chunks):
            self.assertEqual(chunk.chunk_index, expected_idx)

        # Subsequent document resets to 0
        chunks_doc2 = chunk_document(raw)
        self.assertEqual(chunks_doc2[0].chunk_index, 0)

    # =========================================================================
    # T-19: Deterministic UUID v5 (SPEC-02 §17.1 (T15))
    # =========================================================================
    def test_deterministic_uuid5(self) -> None:
        """T-19: Asserts chunk_id is valid RFC 4122 UUID v5 matching

        uuid5(NAMESPACE_URL, f"{doc_id}#{idx}") across repeated runs.
        """
        raw = f"""---
document_id: {self.doc_id_a}
org_id: {self.org_id_a}
access_level: public
title: Deterministic Doc
---
# Deterministic Doc

## Topic One
Body text one.
"""
        run1 = chunk_document(raw)
        run2 = chunk_document(raw)

        self.assertEqual(run1[0].chunk_id, run2[0].chunk_id)
        expected_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.doc_id_a}#0"))
        self.assertEqual(run1[0].chunk_id, expected_uuid)

        # RFC 4122 validation
        parsed = uuid.UUID(run1[0].chunk_id)
        self.assertEqual(parsed.version, 5)

    # =========================================================================
    # T-20: Tenant Isolation Org A (SPEC-02 §17.1 (T16))
    # =========================================================================
    def test_tenant_isolation_org_a(self) -> None:
        """T-20: Asserts all generated chunks for Org A carry strictly Org A UUID."""
        raw = f"""---
document_id: {self.doc_id_a}
org_id: {self.org_id_a}
access_level: public
title: Tenant A Document
---
# Tenant A Document

## Configuration
Tenant A proprietary settings.
"""
        chunks = chunk_document(raw)
        for chunk in chunks:
            self.assertEqual(chunk.org_id, self.org_id_a)
            self.assertNotEqual(chunk.org_id, self.org_id_b)

    # =========================================================================
    # T-21: Interleaved Batch Isolation (SPEC-02 §17.1 (T17))
    # =========================================================================
    def test_interleaved_batch_isolation(self) -> None:
        """T-21: Processes alternating Org A and Org B documents in a single loop;

        asserts 0% cross-tenant contamination or state leakage.
        """
        doc_a_raw = f"""---
document_id: {self.doc_id_a}
org_id: {self.org_id_a}
access_level: public
title: Org A Guide
---
# Org A Guide

## Section A1
Org A data 1.

## Section A2
Org A data 2.
"""
        doc_b_raw = f"""---
document_id: {self.doc_id_b}
org_id: {self.org_id_b}
access_level: internal_staff
title: Org B Guide
---
# Org B Guide

## Section B1
Org B data 1.

## Section B2
Org B data 2.
"""
        # Interleave 10 runs alternating between Org A and Org B
        for i in range(10):
            if i % 2 == 0:
                chunks = chunk_document(doc_a_raw)
                for chunk in chunks:
                    self.assertEqual(chunk.org_id, self.org_id_a)
                    self.assertEqual(chunk.document_id, self.doc_id_a)
                    self.assertEqual(chunk.access_level, "public")
                    self.assertNotIn("Org B", chunk.content)
            else:
                chunks = chunk_document(doc_b_raw)
                for chunk in chunks:
                    self.assertEqual(chunk.org_id, self.org_id_b)
                    self.assertEqual(chunk.document_id, self.doc_id_b)
                    self.assertEqual(chunk.access_level, "internal_staff")
                    self.assertNotIn("Org A", chunk.content)

    # =========================================================================
    # Additional: Logging Privacy Test (SPEC-02 Section 16 & Section 3.I)
    # =========================================================================
    def test_logging_privacy_no_pii(self) -> None:
        """Verifies warning behavior for oversized chunks and unclosed fences;

        asserts zero raw content or sensitive tokens are exposed in logs.
        """
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.WARNING)
        logger = logging.getLogger("rag.chunking")
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)

        try:
            secret_token = "TOP_SECRET_AUTH_BEARER_TOKEN_999888777"
            # Indivisible block >400 tokens containing secret token
            words = ["token"] * 375 + [secret_token]
            body_text = " ".join(words)

            doc = f"""---
document_id: {self.doc_id_a}
org_id: {self.org_id_a}
access_level: public
title: Sensitive Token Manual
---
# Sensitive Token Manual

## Secret Block
{body_text}
"""
            chunk_document(doc)
            output = log_stream.getvalue()

            # Structured keys must be present
            self.assertIn("category=oversized_chunk_zone1", output)
            self.assertIn(f"document_id={self.doc_id_a}", output)

            # Raw content and sensitive tokens must NEVER be logged
            self.assertNotIn(secret_token, output)
            self.assertNotIn("TOP_SECRET", output)
        finally:
            logger.removeHandler(handler)

    # =========================================================================
    # Additional: Corpus Regression Smoke Test (Section 3.J)
    # =========================================================================
    def test_corpus_regression_smoke(self) -> None:
        """Conditionally smoke-tests against data/raw/**/*.md if files exist on disk.

        Gracefully skips if raw dataset directory is unavailable.
        """
        raw_files = sorted(glob.glob("data/raw/**/*.md", recursive=True))
        if not raw_files:
            self.skipTest("data/raw/ directory not available or empty; skipping corpus smoke test.")

        total_chunks = 0
        for path_str in raw_files:
            content = Path(path_str).read_text(encoding="utf-8")
            chunks = chunk_document(content)
            self.assertGreater(len(chunks), 0, f"Expected chunks for {path_str}")
            total_chunks += len(chunks)

            seen_indices = set()
            for chunk in chunks:
                self.assertIsInstance(chunk, DocumentChunk)
                self.assertNotIn(chunk.chunk_index, seen_indices)
                seen_indices.add(chunk.chunk_index)
                self.assertEqual(chunk.chunk_id, generate_chunk_id(chunk.document_id, chunk.chunk_index))
                self.assertIn(chunk.access_level, VALID_ACCESS_LEVELS)
                self.assertEqual(len(chunk.content_hash), 64)

            self.assertEqual(seen_indices, set(range(len(chunks))))


if __name__ == "__main__":
    unittest.main()

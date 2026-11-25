"""Data structures, exceptions, validation, and ID generation for chunking subsystem.

Phase 1 deliverable grounded in:
- docs/specs/SPEC-02-chunking.md
- docs/plans/03-chunking-implementation-plan.md
- Shared Team Contract v3.0 Locked
"""

#Provides the data structures, validations, custom exceptions, and utility functions
# required for the document chunking pipeline.
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import re
from typing import List, Optional
import uuid

# -----------------------------------------------------------------------------
# Constants & Contracts
# -----------------------------------------------------------------------------

VALID_ACCESS_LEVELS = frozenset({"public", "internal_staff", "admin_only"})
DEFAULT_CHUNKING_VERSION = "v1.0-section-aware"

# Note: \Z ensures strictly no trailing newline or whitespace matches.
UUID_V4_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z",
    re.IGNORECASE,
)


# -----------------------------------------------------------------------------
# Custom Exception Hierarchy
# -----------------------------------------------------------------------------


class ChunkingError(Exception):
    """Base exception for all chunking pipeline errors."""


class EmptyDocumentError(ChunkingError):
    """Raised when input document is empty or whitespace-only."""


class MissingFrontmatterError(ChunkingError):
    """Raised when opening '---' frontmatter fence is absent on line 1."""


class MalformedFrontmatterError(ChunkingError):
    """Raised when frontmatter is unclosed before EOF or structurally unparseable."""


class InvalidMetadataError(ChunkingError):
    """Raised when org_id or document_id fails strict UUID v4 validation."""


class InvalidAccessLevelError(ChunkingError):
    """Raised when access_level is not in {'public', 'internal_staff', 'admin_only'}."""


# -----------------------------------------------------------------------------
# Validation Utilities
# -----------------------------------------------------------------------------


def validate_uuid_v4(value: str) -> None:
    """Strictly validate that value is a canonical RFC 4122 UUID v4 string.

    Operates strictly on the raw input string without stripping or normalization.
    Rejects non-string types, surrounding whitespace, non-canonical formatting,
    wrong version bits, or normalized inputs by raising InvalidMetadataError.
    """
    if not isinstance(value, str):
        raise InvalidMetadataError(f"UUID must be a string, got {type(value).__name__}")
    if len(value) != 36 or not UUID_V4_PATTERN.match(value):
        raise InvalidMetadataError(f"String '{value}' is not a canonical RFC 4122 UUID v4 format")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise InvalidMetadataError(f"Malformed UUID value '{value}': {exc}") from exc
    if parsed.version != 4:
        raise InvalidMetadataError(f"UUID '{value}' is version {parsed.version}, expected version 4")
    if str(parsed).lower() != value.lower():
        raise InvalidMetadataError(f"UUID '{value}' was normalized from non-canonical representation")


# -----------------------------------------------------------------------------
# Dataclasses :How the system structures its data
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FrontmatterMetadata:
    """Validated frontmatter metadata container extracted from Markdown header."""

    document_id: str
    org_id: str
    access_level: str
    title: str
    category: Optional[str] = None
    source: Optional[str] = None
#The system extracts the values and stores them in a FrontmatterMetadata object.

@dataclass
class ParsedSection:
    """Hierarchical structural node representing an ATX heading and its content lines."""

    heading_level: int
    heading_text: str
    heading_path: List[str]
    content_lines: List[str] = field(default_factory=list)
    children: List[ParsedSection] = field(default_factory=list)
    start_line: int = 0
#This parsedsection class represents a section of a document.


#After a large document is parsed and divided into smaller pieces, each chunk is represented as a DocumentChunk object.
@dataclass(frozen=True)
class DocumentChunk:
    """Immutable final chunk payload adhering strictly to database contract and audit metadata.

    Mapped database columns:
        chunk_id: maps to knowledge_base_articles.id (typed as UUID)
        document_id: maps to knowledge_base_articles.document_id
        org_id: maps to knowledge_base_articles.org_id
        access_level: maps to knowledge_base_articles.access_level
        chunk_index: maps to knowledge_base_articles.chunk_index
        title: maps to knowledge_base_articles.title
        content: maps to knowledge_base_articles.content

    Application-level audit metadata:
        heading_path: canonical ancestor heading hierarchy (presentation markers like '(Table Part M)'
                      are NEVER added to heading_path).
        token_count_est: estimated token length of content.
        content_hash: SHA-256 hex digest of content.
        chunking_version: algorithm version.
        category: optional category from frontmatter.
        source: optional source provenance descriptor from frontmatter.
        is_oversized: True if content exceeded heuristic ceiling (400 tokens).
        is_split_item: True if an indivisible element was sentence-split.
    """

    # Mandatory Database Contract Fields
    chunk_id: str
    document_id: str
    org_id: str
    access_level: str
    chunk_index: int
    title: str
    content: str

    # Internal Pipeline Metadata
    heading_path: List[str]
    token_count_est: int
    content_hash: str
    chunking_version: str = DEFAULT_CHUNKING_VERSION
    category: Optional[str] = None
    source: Optional[str] = None
    is_oversized: bool = False
    is_split_item: bool = False


# -----------------------------------------------------------------------------
# Mathematical & Cryptographic Utilities
# -----------------------------------------------------------------------------


def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """Produce deterministic RFC 4122 UUID v5 string from document_id and chunk_index."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}#{chunk_index}"))


def estimate_tokens(text: str) -> int:
    """Calculate estimated token length using standard whitespace heuristic: ceil(word_count * 1.33)."""
    word_count = len(text.split())
    return math.ceil(word_count * 1.33)


def compute_content_hash(text: str) -> str:
    """Return SHA-256 hex digest of the chunk content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

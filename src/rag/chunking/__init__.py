"""Chunking subsystem package."""
#Initializes the engine
from src.rag.chunking.engine import (
    ChunkingEngine,
    chunk_document,
    format_breadcrumb,
)
from src.rag.chunking.models import (
    DEFAULT_CHUNKING_VERSION,
    ChunkingError,
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

__all__ = [
    "ChunkingError",
    "EmptyDocumentError",
    "MissingFrontmatterError",
    "MalformedFrontmatterError",
    "InvalidMetadataError",
    "InvalidAccessLevelError",
    "FrontmatterMetadata",
    "ParsedSection",
    "DocumentChunk",
    "VALID_ACCESS_LEVELS",
    "DEFAULT_CHUNKING_VERSION",
    "validate_uuid_v4",
    "generate_chunk_id",
    "estimate_tokens",
    "compute_content_hash",
    "MarkdownFSMParser",
    "extract_frontmatter",
    "parse_sections",
    "parse_document",
    "ChunkingEngine",
    "chunk_document",
    "format_breadcrumb",
]

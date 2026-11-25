"""Structural Markdown Finite State Machine (FSM) parser.

Phase 2 deliverable grounded in:
- docs/specs/SPEC-02-chunking.md
- docs/plans/03-chunking-implementation-plan.md
- Shared Team Contract v3.0 Locked

This file validates Markdown metadata and uses a state-based parser to organize
document headings and content into a hierarchical structure for the chunking pipeline.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from src.rag.chunking.models import (
    EmptyDocumentError,
    FrontmatterMetadata,
    InvalidAccessLevelError,
    InvalidMetadataError,
    MalformedFrontmatterError,
    MissingFrontmatterError,
    ParsedSection,
    VALID_ACCESS_LEVELS,
    validate_uuid_v4,
)

logger = logging.getLogger("rag.chunking")

# -----------------------------------------------------------------------------
# Parser State Constants
# -----------------------------------------------------------------------------

STATE_BODY = "STATE_BODY"
STATE_CODE_FENCE = "STATE_CODE_FENCE"
STATE_TABLE = "STATE_TABLE"

# -----------------------------------------------------------------------------
# Grammar Patterns (SPEC-02 Section 3.2)
# -----------------------------------------------------------------------------

# Matches ATX headings (# through ######) with optional closing hashes
ATX_HEADING_PATTERN = re.compile(
    r"^\s{0,3}(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*#*\s*$"
)

# Matches code fence opening: 3+ backticks or 3+ tildes with optional language tag
CODE_FENCE_OPEN_PATTERN = re.compile(
    r"^\s{0,3}(?P<fence>```+|~~~+)(?P<lang>[a-zA-Z0-9_-]*)\s*$"
)

# Matches code fence closing: 3+ backticks or 3+ tildes
CODE_FENCE_CLOSE_PATTERN = re.compile(
    r"^\s{0,3}(?P<fence>```+|~~~+)\s*$"
)

# Matches markdown pipe table row
TABLE_ROW_PATTERN = re.compile(r"^\s*\|(?P<row>.*)\|\s*$")

# Matches table divider row
TABLE_DIVIDER_PATTERN = re.compile(r"^\s*\|(?:\s*:?-+:?\s*\|)+\s*$")


# -----------------------------------------------------------------------------
# Frontmatter Extraction & Validation
# -----------------------------------------------------------------------------

#This function takes the entire raw Markdown document and separates it into:
#1. Metadata
#2. Actual document body
#This function extracts the document's information and validates it before the document is processed further.
def extract_frontmatter(raw_text: str) -> Tuple[FrontmatterMetadata, str]:
    """Extract and validate YAML-style frontmatter from raw Markdown text.

    Returns:
        Tuple of (validated FrontmatterMetadata, body_markdown_text).

    Raises:
        EmptyDocumentError: If document is empty or whitespace-only.
        MissingFrontmatterError: If opening '---' delimiter is absent on line 1.
        MalformedFrontmatterError: If frontmatter is unclosed, structurally invalid,
                                   or missing required fields.
        InvalidMetadataError: If document_id or org_id fails strict UUID v4 validation.
        InvalidAccessLevelError: If access_level is not in {'public', 'internal_staff', 'admin_only'}.
    """
    if not raw_text or not raw_text.strip():
        raise EmptyDocumentError("Input document is empty or whitespace-only")

    lines = raw_text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise MissingFrontmatterError("Document missing opening '---' frontmatter fence on line 1")

    closing_index = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            closing_index = i
            break

    if closing_index == -1:
        raise MalformedFrontmatterError(
            f"Unclosed frontmatter fence: opening '---' on line 1 has no closing fence before line {len(lines) + 1}"
        )

    raw_fm_lines = lines[1:closing_index]
    body_text = "".join(lines[closing_index + 1 :])

    parsed_fm: dict[str, str] = {}
    for offset, line in enumerate(raw_fm_lines, start=2):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("- "):
            raise MalformedFrontmatterError(
                f"Unsupported YAML sequence at line {offset}: lists not supported in flat frontmatter"
            )
        if ":" not in stripped:
            raise MalformedFrontmatterError(
                f"Malformed frontmatter at line {offset}: missing ':' key-value separator"
            )

        key, val = stripped.split(":", 1)
        key = key.strip()
        raw_val = val.strip()

        # Handle quoted string values while preserving internal contents
        if len(raw_val) >= 2 and (
            (raw_val[0] == '"' and raw_val[-1] == '"')
            or (raw_val[0] == "'" and raw_val[-1] == "'")
        ):
            parsed_val = raw_val[1:-1]
        else:
            parsed_val = raw_val

        parsed_fm[key] = parsed_val

    # Validate required fields
    required_fields = ["document_id", "org_id", "access_level", "title"]
    for field in required_fields:
        if field not in parsed_fm:
            raise MalformedFrontmatterError(f"Missing required frontmatter field: '{field}'")
        if not parsed_fm[field].strip():
            raise MalformedFrontmatterError(f"Frontmatter field '{field}' must not be empty")

    doc_id = parsed_fm["document_id"]
    org_id = parsed_fm["org_id"]
    access_level = parsed_fm["access_level"]
    title = parsed_fm["title"].strip()
    category = parsed_fm.get("category")
    source = parsed_fm.get("source")

    # Strict UUID v4 validation (raises InvalidMetadataError on failure)
    validate_uuid_v4(doc_id)
    validate_uuid_v4(org_id)

    # Access level validation
    if access_level not in VALID_ACCESS_LEVELS:
        raise InvalidAccessLevelError(
            f"Invalid access_level '{access_level}'. Must be one of {sorted(VALID_ACCESS_LEVELS)}"
        )

    metadata = FrontmatterMetadata(
        document_id=doc_id,
        org_id=org_id,
        access_level=access_level,
        title=title,
        category=category.strip() if category else None,
        source=source.strip() if source else None,
    )

    return metadata, body_text


# -----------------------------------------------------------------------------
# Bounded Structural FSM Parser
# -----------------------------------------------------------------------------

#It uses a Finite State Machine (FSM) to read Markdown line by line and organize headings into a tree-like structure.
#An FSM (Finite-State Machine) is a simple rulebook that helps my program remember what kind of text it is currently
# reading—like a normal paragraph, a table, or a code block.
class MarkdownFSMParser:
    """Bounded Finite State Machine parser for B2B Knowledge Base Markdown documents.

    Transitions between STATE_BODY, STATE_CODE_FENCE, and STATE_TABLE.
    Guarantees complete suppression of headings inside code fences.
    """

    def __init__(self, document_id: Optional[str] = None):
        self.document_id = document_id or "unknown"

    def parse(
        self, body_markdown: str, root_title: str = "Document"
    ) -> ParsedSection:
        """Parse sanitized body markdown into a hierarchical ParsedSection tree.

        Args:
            body_markdown: Markdown body text (after frontmatter removal).
            root_title: Document title (from frontmatter or H1).

        Returns:
            Root ParsedSection representing the document overview and containing
            the section hierarchy as children.
        """
        lines = body_markdown.splitlines(keepends=True)

        root = ParsedSection(
            heading_level=1,
            heading_text=root_title,
            heading_path=[root_title],
            content_lines=[],
            children=[],
            start_line=1,
        )
#A stack helps the parser keep track of the current section and its parent sections.
        stack: List[ParsedSection] = [root]
        state = STATE_BODY

        fence_char = ""
        fence_len = 0
        fence_start_line = 0

        for line_idx, line in enumerate(lines, start=1):
            stripped_line = line.strip()

            # -----------------------------------------------------------------
            # STATE: Inside Code Fence
            # -----------------------------------------------------------------
            if state == STATE_CODE_FENCE:
                close_match = CODE_FENCE_CLOSE_PATTERN.match(line)
                if close_match:
                    close_fence = close_match.group("fence")
                    if close_fence[0] == fence_char and len(close_fence) >= fence_len:
                        state = STATE_BODY
                        fence_char = ""
                        fence_len = 0
                        fence_start_line = 0
                # In STATE_CODE_FENCE: heading evaluation is completely bypassed!
                stack[-1].content_lines.append(line)
                continue

            # -----------------------------------------------------------------
            # STATE: Body or Table
            # -----------------------------------------------------------------
            # Check for opening code fence
            open_fence_match = CODE_FENCE_OPEN_PATTERN.match(line)
            if open_fence_match:
                state = STATE_CODE_FENCE
                fence_str = open_fence_match.group("fence")
                fence_char = fence_str[0]
                fence_len = len(fence_str)
                fence_start_line = line_idx
                stack[-1].content_lines.append(line)
                continue

            # Check for ATX heading
            heading_match = ATX_HEADING_PATTERN.match(line)
            if heading_match:
                state = STATE_BODY
                hashes = heading_match.group("hashes")
                new_level = len(hashes)
                new_title = heading_match.group("title").strip()

                # Handle top-level H1 heading: if root has no content or matches root_title
                if new_level == 1 and len(stack) == 1:
                    if not root.content_lines and not root.children:
                        root.heading_text = new_title
                        root.heading_path = [new_title]
                        root.start_line = line_idx
                        continue
                    if root.heading_text == new_title and not root.children:
                        root.start_line = line_idx
                        continue

                # Pop sections of equal or deeper level
                while len(stack) > 1 and stack[-1].heading_level >= new_level:
                    stack.pop()

                parent = stack[-1]
                new_path = parent.heading_path + [new_title]

                new_section = ParsedSection(
                    heading_level=new_level,
                    heading_text=new_title,
                    heading_path=new_path,
                    content_lines=[],
                    children=[],
                    start_line=line_idx,
                )

                parent.children.append(new_section)
                stack.append(new_section)
                continue

            # Check for Table Row
            if TABLE_ROW_PATTERN.match(line):
                state = STATE_TABLE
                stack[-1].content_lines.append(line)
                continue

            # Regular body lines (paragraphs, lists, blockquotes, blank lines)
            if state == STATE_TABLE and not stripped_line:
                state = STATE_BODY

            stack[-1].content_lines.append(line)

        # ---------------------------------------------------------------------
        # Post-Processing: Graceful handling of unclosed code fence at EOF
        # ---------------------------------------------------------------------
        if state == STATE_CODE_FENCE:
            logger.warning(
                "Unclosed code fence encountered at end of document. "
                "[category=unclosed_code_fence, document_id=%s, start_line=%d, fence_char=%s]",
                self.document_id,
                fence_start_line,
                fence_char,
            )

        return root


# -----------------------------------------------------------------------------
# Functional Public API
# -----------------------------------------------------------------------------



#This function creates a Markdown parser and asks it to organize the document's content into a structured hierarchy.
def parse_sections(
    body_markdown: str,
    root_title: str = "Document",
    document_id: Optional[str] = None,
) -> ParsedSection:
    """Parse body markdown into a ParsedSection tree using default parser."""
    parser = MarkdownFSMParser(document_id=document_id)
    return parser.parse(body_markdown, root_title=root_title)


def parse_document(raw_text: str) -> Tuple[FrontmatterMetadata, ParsedSection]:
    """Extract frontmatter and parse markdown body into a structured section tree."""
    metadata, body_text = extract_frontmatter(raw_text)
    tree = parse_sections(
        body_text, root_title=metadata.title, document_id=metadata.document_id
    )
    return metadata, tree

#parse_document()	Extracts metadata first, then parses the body, and returns both.
#parse_sections()   Parses the Markdown body into a section tree.

"""Section-aware chunking engine, breadcrumb injection, and fallback protocol.

Phase 3 deliverable grounded in:
- docs/specs/SPEC-02-chunking.md
- docs/plans/03-chunking-implementation-plan.md
- Shared Team Contract v3.0 Locked
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from src.rag.chunking.models import (
    DEFAULT_CHUNKING_VERSION,
    DocumentChunk,
    FrontmatterMetadata,
    ParsedSection,
    compute_content_hash,
    estimate_tokens,
    generate_chunk_id,
)
from src.rag.chunking.parser import (
    CODE_FENCE_CLOSE_PATTERN,
    CODE_FENCE_OPEN_PATTERN,
    TABLE_DIVIDER_PATTERN,
    TABLE_ROW_PATTERN,
    extract_frontmatter,
    parse_sections,
)

logger = logging.getLogger("rag.chunking")

# Sizing Constants (SPEC-02 Section 9.2 & Section 11.2)
HEURISTIC_CEILING_TOKENS = 400
ZONE2_CRITICAL_TOKENS = 800
SMALL_SECTION_THRESHOLD_TOKENS = 60


# -----------------------------------------------------------------------------
# Breadcrumb Formatting (SPEC-02 Section 8)
# -----------------------------------------------------------------------------


#This function adds the section's location to the chunk so the AI doesn't lose context.
def format_breadcrumb(#Adds context to chunks
    doc_title: str,
    heading_path: List[str],
    sequence_marker: Optional[str] = None,
) -> str:
    """Format contextual heading breadcrumb header for chunk content payload.

    Formatting rules:
        Top-level intro: [Document: <doc_title> > Overview]
        H2 section:      [Document: <doc_title> > <H2 Title>]
        H3 subsection:   [Document: <doc_title> > <H2 Title> > <H3 Title>]
        With sequence:   [Document: <doc_title> > <H2 Title> (Table Part 2)]

    Note: sequence_marker is purely a presentation artifact. It must never
    modify canonical heading_path metadata (Invariant F-01).
    """
    if len(heading_path) <= 1:
        path_str = f"{doc_title} > Overview"
    else:
        # heading_path includes ancestor headings (e.g. [doc_title, h2, h3])
        path_str = " > ".join(heading_path)

    if sequence_marker:
        path_str = f"{path_str} {sequence_marker}"

    return f"[Document: {path_str}]"


# -----------------------------------------------------------------------------
# Text and Table Helper Functions
# -----------------------------------------------------------------------------

#This function splits text into sentences but preserves the spaces and newlines between them.
def split_sentences_preserving_whitespace(text: str) -> List[str]:
    """Split text along sentence boundaries while retaining all inter-sentence whitespace.

    Uses capturing delimiter groups so that "".join(returned_sentences) == text
    exactly, satisfying Whitespace Retention Invariant F-03.
    """
    parts = re.split(r"((?<=[.!?])\s+)", text)
    sentences: List[str] = []
    i = 0
    while i < len(parts):
        s = parts[i]
        if i + 1 < len(parts):
            s += parts[i + 1]
            i += 2
        else:
            i += 1
        if s:
            sentences.append(s)
    return sentences if sentences else [text]

#This function divides section content into different types of blocks:paragraphs, tables, and code blocks.
def extract_structural_blocks(lines: List[str]) -> List[Tuple[str, str]]:
    """Partition a section's raw content lines into discrete structural blocks.

    Returns:
        List of tuples: (block_type, block_text), where block_type is
        'table', 'code', or 'paragraph'.
    """
    blocks: List[Tuple[str, str]] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Check for fenced code block
        open_fence = CODE_FENCE_OPEN_PATTERN.match(line)
        if open_fence:
            fence_str = open_fence.group("fence")
            fence_char = fence_str[0]
            fence_len = len(fence_str)
            code_lines = [line]
            i += 1
            while i < n:
                code_line = lines[i]
                code_lines.append(code_line)
                close_match = CODE_FENCE_CLOSE_PATTERN.match(code_line)
                if close_match:
                    cf = close_match.group("fence")
                    if cf[0] == fence_char and len(cf) >= fence_len:
                        i += 1
                        break
                i += 1
            blocks.append(("code", "".join(code_lines)))
            continue

        # Check for Markdown pipe table
        if TABLE_ROW_PATTERN.match(line):
            table_lines = [line]
            i += 1
            while i < n and TABLE_ROW_PATTERN.match(lines[i]):
                table_lines.append(lines[i])
                i += 1
            blocks.append(("table", "".join(table_lines)))
            continue

        # Regular paragraph or list item group
        para_lines = [line]
        i += 1
        while i < n:
            next_line = lines[i]
            if not next_line.strip():
                break
            if CODE_FENCE_OPEN_PATTERN.match(next_line) or TABLE_ROW_PATTERN.match(next_line):
                break
            para_lines.append(next_line)
            i += 1
        blocks.append(("paragraph", "".join(para_lines)))

    return blocks


# -----------------------------------------------------------------------------
# Chunking Engine
# -----------------------------------------------------------------------------

#It takes the structured document tree and turns it into actual DocumentChunk objects.
class ChunkingEngine:
    """Stateful chunk assembly engine enforcing section boundaries, sizing, and fallback."""
#intializes the engine
    def __init__(
        self,
        metadata: FrontmatterMetadata,
        ceiling_tokens: int = HEURISTIC_CEILING_TOKENS,
        zone2_threshold: int = ZONE2_CRITICAL_TOKENS,
    ):
        self.metadata = metadata
        self.ceiling_tokens = ceiling_tokens
        self.zone2_threshold = zone2_threshold
        self.chunks: List[DocumentChunk] = []
        self.current_chunk_index = 0
#It takes content and turns it into a final DocumentChunk.
    def _emit_chunk(
        self,
        breadcrumb: str,
        body_content: str,
        canonical_heading_path: List[str],
        is_oversized: bool = False,
        is_split_item: bool = False,
    ) -> DocumentChunk:
        """Construct and register an immutable DocumentChunk instance."""
        if is_split_item:
            # Preserve trailing whitespace on split items to satisfy F-03 exact reassembly
            clean_body = body_content.lstrip("\n")
        else:
            clean_body = body_content.strip()
        full_content = f"{breadcrumb}\n\n{clean_body}"
        token_count = estimate_tokens(full_content)

        chunk = DocumentChunk(
            chunk_id=generate_chunk_id(self.metadata.document_id, self.current_chunk_index),
            document_id=self.metadata.document_id,
            org_id=self.metadata.org_id,
            access_level=self.metadata.access_level,
            chunk_index=self.current_chunk_index,
            title=self.metadata.title,
            content=full_content,
            heading_path=list(canonical_heading_path),  # strict canonical ancestor path
            token_count_est=token_count,
            content_hash=compute_content_hash(full_content),
            chunking_version=DEFAULT_CHUNKING_VERSION,
            category=self.metadata.category,
            source=self.metadata.source,
            is_oversized=is_oversized,
            is_split_item=is_split_item,
        )

        self.chunks.append(chunk)
        self.current_chunk_index += 1
        return chunk
#If the item exceeds 800 tokens, the system tries to split it at sentence boundaries.
    def _emit_oversized_item(#Handles content that is too large
        self,
        breadcrumb: str,
        item_text: str,
        canonical_heading_path: List[str],
        section_title: str,
        table_header_prefix: Optional[str] = None,
    ) -> None:
        """Execute Two-Zone Fallback Protocol for an indivisible element exceeding 400 tokens."""
        clean_item = item_text.strip()
        # Full constructed evaluation includes breadcrumb and any repeated table headers
        eval_payload = f"{breadcrumb}\n\n{table_header_prefix or ''}{clean_item}"
        token_est = estimate_tokens(eval_payload)

        # ---------------------------------------------------------------------
        # ZONE 1: Controlled Oversize (400 to 800 estimated tokens)
        # ---------------------------------------------------------------------
        if token_est <= self.zone2_threshold:
            logger.warning(
                "Chunk exceeds heuristic ceiling; emitted intact under Zone 1 fallback. "
                "[category=oversized_chunk_zone1, document_id=%s, chunk_index=%d, "
                "estimated_tokens=%d, threshold=%d]",
                self.metadata.document_id,
                self.current_chunk_index,
                token_est,
                self.ceiling_tokens,
            )
            body = f"{table_header_prefix or ''}{clean_item}"
            self._emit_chunk(
                breadcrumb=breadcrumb,
                body_content=body,
                canonical_heading_path=canonical_heading_path,
                is_oversized=True,
                is_split_item=False,
            )
            return

        # ---------------------------------------------------------------------
        # ZONE 2: Emergency Sentence-Boundary Split (>800 estimated tokens)
        # ---------------------------------------------------------------------
        sentences = split_sentences_preserving_whitespace(clean_item)

        # Group sentences into sub-chunks respecting the heuristic ceiling
        sub_chunk_sentence_groups: List[List[str]] = []
        current_group: List[str] = []

        for sent in sentences:
            test_group = current_group + [sent]
            test_body = "".join(test_group)
            continuation_marker = f"[Continuation: {section_title} (Part {len(sub_chunk_sentence_groups) + 1})]"
            candidate_payload = f"{breadcrumb}\n\n{continuation_marker}\n\n{table_header_prefix or ''}{test_body}"

            if current_group and estimate_tokens(candidate_payload) > self.ceiling_tokens:
                sub_chunk_sentence_groups.append(current_group)
                current_group = [sent]
            else:
                current_group.append(sent)

        if current_group:
            sub_chunk_sentence_groups.append(current_group)

        total_parts = len(sub_chunk_sentence_groups)
        logger.warning(
            "Chunk exceeds critical ceiling; split on sentence boundaries under Zone 2 fallback. "
            "[category=oversized_chunk_zone2_split, document_id=%s, chunk_index=%d, "
            "estimated_tokens=%d, split_parts_count=%d]",
            self.metadata.document_id,
            self.current_chunk_index,
            token_est,
            total_parts,
        )

        for part_idx, sent_group in enumerate(sub_chunk_sentence_groups, start=1):
            sent_text = "".join(sent_group)
            continuation_marker = f"[Continuation: {section_title} (Part {part_idx})]"
            body = f"{continuation_marker}\n\n{table_header_prefix or ''}{sent_text}"
            self._emit_chunk(
                breadcrumb=breadcrumb,
                body_content=body,
                canonical_heading_path=canonical_heading_path,
                is_oversized=True,
                is_split_item=True,
            )

    def _process_table_block(
        self,
        table_text: str,
        breadcrumb_base: str,
        canonical_heading_path: List[str],
        section_title: str,
        current_blocks: List[str],
    ) -> None:
        """Handle a Markdown table according to operational flow (M3 Plan Section 3.4)."""
        table_lines = [l if l.endswith("\n") else l + "\n" for l in table_text.strip().splitlines()]
        if len(table_lines) < 2:
            current_blocks.append(table_text)
            return

        header_row = table_lines[0]
        divider_row = table_lines[1]
        data_rows = table_lines[2:]
        table_header_prefix = f"{header_row}{divider_row}"

        # Case 1: Does entire table fit within active accumulated chunk?
        candidate_blocks = current_blocks + [table_text]
        candidate_body = "\n\n".join(candidate_blocks)
        if estimate_tokens(f"{breadcrumb_base}\n\n{candidate_body}") <= self.ceiling_tokens:
            current_blocks.append(table_text)
            return

        # Case 2: Flush active chunk first if present
        if current_blocks:
            self._emit_chunk(
                breadcrumb=breadcrumb_base,
                body_content="\n\n".join(current_blocks),
                canonical_heading_path=canonical_heading_path,
            )
            current_blocks.clear()

        # Case 3: Does entire table fit into a fresh standard chunk?
        if estimate_tokens(f"{breadcrumb_base}\n\n{table_text.strip()}") <= self.ceiling_tokens:
            current_blocks.append(table_text)
            return

        # Case 4: Oversized table (>400 tokens total) -> Table Row Partitioning
        active_table_rows: List[str] = []
        table_part = 1

        for row in data_rows:
            # Check if this individual data row alone exceeds ceiling
            row_tokens = estimate_tokens(f"{breadcrumb_base}\n\n{table_header_prefix}{row.strip()}")
            if row_tokens > self.ceiling_tokens:
                # Flush existing table rows if any
                if active_table_rows:
                    seq_marker = f"(Table Part {table_part})" if table_part > 1 else None
                    part_bc = format_breadcrumb(
                        self.metadata.title, canonical_heading_path, sequence_marker=seq_marker
                    )
                    part_body = f"{table_header_prefix}{''.join(active_table_rows)}"
                    self._emit_chunk(part_bc, part_body, canonical_heading_path)
                    active_table_rows.clear()
                    table_part += 1

                # Route oversized single row to fallback
                seq_marker = f"(Table Part {table_part})" if table_part > 1 else None
                part_bc = format_breadcrumb(
                    self.metadata.title, canonical_heading_path, sequence_marker=seq_marker
                )
                self._emit_oversized_item(
                    breadcrumb=part_bc,
                    item_text=row,
                    canonical_heading_path=canonical_heading_path,
                    section_title=section_title,
                    table_header_prefix=table_header_prefix,
                )
                table_part += 1
                continue

            test_rows = active_table_rows + [row]
            test_body = f"{table_header_prefix}{''.join(test_rows)}"
            seq_marker = f"(Table Part {table_part})" if table_part > 1 else None
            test_bc = format_breadcrumb(
                self.metadata.title, canonical_heading_path, sequence_marker=seq_marker
            )

            if active_table_rows and estimate_tokens(f"{test_bc}\n\n{test_body}") > self.ceiling_tokens:
                # Emit current table chunk
                part_body = f"{table_header_prefix}{''.join(active_table_rows)}"
                self._emit_chunk(test_bc, part_body, canonical_heading_path)
                active_table_rows = [row]
                table_part += 1
            else:
                active_table_rows.append(row)

        if active_table_rows:
            seq_marker = f"(Table Part {table_part})" if table_part > 1 else None
            final_bc = format_breadcrumb(
                self.metadata.title, canonical_heading_path, sequence_marker=seq_marker
            )
            final_body = f"{table_header_prefix}{''.join(active_table_rows)}"
            self._emit_chunk(final_bc, final_body, canonical_heading_path)
#This function processes paragraphs, code blocks and tables and decides when to create a chunk.
    def _process_content_blocks(#Manages chunk size
        self,
        blocks: List[Tuple[str, str]],
        breadcrumb: str,
        canonical_heading_path: List[str],
        section_title: str,
        initial_prefix: Optional[str] = None,
    ) -> None:
        """Accumulate structural blocks, enforce 400-token ceiling, and dispatch to fallback."""
        current_blocks: List[str] = [initial_prefix] if initial_prefix else []

        for block_type, block_text in blocks:
            if block_type == "table":
                self._process_table_block(
                    table_text=block_text,
                    breadcrumb_base=breadcrumb,
                    canonical_heading_path=canonical_heading_path,
                    section_title=section_title,
                    current_blocks=current_blocks,
                )
                continue

            # Paragraph or Code Block
            candidate_blocks = current_blocks + [block_text]
            candidate_body = "\n\n".join(candidate_blocks)

            if estimate_tokens(f"{breadcrumb}\n\n{candidate_body}") <= self.ceiling_tokens:
                current_blocks.append(block_text)
            else:
                # Flush current accumulated blocks first
                if current_blocks:
                    self._emit_chunk(
                        breadcrumb=breadcrumb,
                        body_content="\n\n".join(current_blocks),
                        canonical_heading_path=canonical_heading_path,
                    )
                    current_blocks.clear()

                # Evaluate single block alone
                single_block_tokens = estimate_tokens(f"{breadcrumb}\n\n{block_text.strip()}")
                if single_block_tokens <= self.ceiling_tokens:
                    current_blocks.append(block_text)
                else:
                    # Indivisible element exceeds 400 tokens -> Two-Zone Fallback
                    self._emit_oversized_item(
                        breadcrumb=breadcrumb,
                        item_text=block_text,
                        canonical_heading_path=canonical_heading_path,
                        section_title=section_title,
                    )

        if current_blocks:
            self._emit_chunk(
                breadcrumb=breadcrumb,
                body_content="\n\n".join(current_blocks),
                canonical_heading_path=canonical_heading_path,
            )
#It takes the section tree created by Phase 2 and transforms it into an ordered list of chunks.
    def chunk_section_tree(self, tree: ParsedSection) -> List[DocumentChunk]:#Processes the entire section tree
        """Transform a ParsedSection tree into an ordered list of DocumentChunks."""
        self.chunks.clear()
        self.current_chunk_index = 0

        # ---------------------------------------------------------------------
        # 1. Document Overview Content (preceding first H2)
        # ---------------------------------------------------------------------
        if tree.content_lines:
            overview_blocks = extract_structural_blocks(tree.content_lines)
            if overview_blocks:
                overview_bc = format_breadcrumb(self.metadata.title, [self.metadata.title])
                self._process_content_blocks(
                    blocks=overview_blocks,
                    breadcrumb=overview_bc,
                    canonical_heading_path=[self.metadata.title],
                    section_title="Overview",
                )

        # ---------------------------------------------------------------------
        # 2. Section Traversal (H2 Topic Boundaries)
        # ---------------------------------------------------------------------
        for h2_sec in tree.children:
            h2_title = h2_sec.heading_text
            h2_canonical_path = h2_sec.heading_path

            # Case A: Standalone H2 (No H3 Children)
            if not h2_sec.children:
                h2_blocks = extract_structural_blocks(h2_sec.content_lines)
                h2_bc = format_breadcrumb(self.metadata.title, h2_canonical_path)
                self._process_content_blocks(
                    blocks=h2_blocks,
                    breadcrumb=h2_bc,
                    canonical_heading_path=h2_canonical_path,
                    section_title=h2_title,
                )
                continue

            # Case B: H2 with Child H3 Subsections
            h2_intro_text = "".join(h2_sec.content_lines).strip()
            h2_intro_tokens = estimate_tokens(h2_intro_text) if h2_intro_text else 0
            pending_intro: Optional[str] = None

            if 0 < h2_intro_tokens < SMALL_SECTION_THRESHOLD_TOKENS:
                # Small intro (<60 tokens): merge forward into first child H3 subsection
                pending_intro = h2_intro_text
            elif h2_intro_tokens >= SMALL_SECTION_THRESHOLD_TOKENS:
                # Substantial intro (>=60 tokens): emit as independent chunk under H2
                h2_intro_blocks = extract_structural_blocks(h2_sec.content_lines)
                h2_bc = format_breadcrumb(self.metadata.title, h2_canonical_path)
                self._process_content_blocks(
                    blocks=h2_intro_blocks,
                    breadcrumb=h2_bc,
                    canonical_heading_path=h2_canonical_path,
                    section_title=h2_title,
                )

            # Traverse H3 child subsections
            for idx, h3_sec in enumerate(h2_sec.children):
                h3_title = h3_sec.heading_text
                h3_canonical_path = h3_sec.heading_path
                h3_bc = format_breadcrumb(self.metadata.title, h3_canonical_path)
                h3_blocks = extract_structural_blocks(h3_sec.content_lines)

                initial_prefix: Optional[str] = None
                if idx == 0 and pending_intro:
                    # Inject intro text followed by H3 header
                    initial_prefix = f"{pending_intro}\n\n### {h3_title}"

                self._process_content_blocks(
                    blocks=h3_blocks,
                    breadcrumb=h3_bc,
                    canonical_heading_path=h3_canonical_path,
                    section_title=h3_title,
                    initial_prefix=initial_prefix,
                )

        return list(self.chunks)


# -----------------------------------------------------------------------------
# Functional Public API
# -----------------------------------------------------------------------------

#This is the function that connects the entire pipeline.
def chunk_document(raw_markdown: str) -> List[DocumentChunk]:
    """Top-level functional API: extract frontmatter, parse Markdown, and generate chunks.

    Returns:
        List of immutable, sequentially indexed DocumentChunk objects.
    """
    metadata, body_text = extract_frontmatter(raw_markdown)
    tree = parse_sections(
        body_text, root_title=metadata.title, document_id=metadata.document_id
    )
    engine = ChunkingEngine(metadata=metadata)
    return engine.chunk_section_tree(tree)

#!/usr/bin/env python3
"""
M1 Dataset Reconnaissance Scanner.

A zero-external-dependency diagnostic tool to inspect knowledge-base files,
catalog metadata, compute quantitative distributions, detect duplicates,
safely audit PII patterns, and detect PDFs without extracting binary text.

Complies strictly with SPEC-01 and the approved M1 Implementation Plan.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any

# Canonical access levels specified by shared team contract
VALID_ACCESS_LEVELS = {"public", "internal_staff", "admin_only"}

# UUID v4 pattern
UUID_REGEX = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

# PII Patterns (used strictly for match counting - matches are NEVER printed or saved)
PII_PATTERNS = {
    "email": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"
    ),
    "phone": re.compile(
        r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "api_key": re.compile(
        r"\b(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|Bearer\s+[a-zA-Z0-9._-]+)\b"
    ),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}


def parse_lightweight_frontmatter(
    content: str,
) -> tuple[dict[str, str] | None, str, str | None]:
    """
    Parse a lightweight YAML frontmatter subset from document text.

    Supports flat key-value pairs (`key: value` or `key: "value"`).
    Does NOT claim full YAML compatibility.
    Returns:
        (frontmatter_dict, body_text, error_message)
    """
    if not content.startswith("---"):
        return None, content, None

    # Find the closing fence
    lines = content.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return None, content, None

    closing_index = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            closing_index = i
            break

    if closing_index == -1:
        return None, content, "Unclosed frontmatter fence: missing closing '---'"

    raw_fm_lines = lines[1:closing_index]
    body_text = "".join(lines[closing_index + 1 :])

    frontmatter: dict[str, str] = {}
    for line_num, line in enumerate(raw_fm_lines, start=2):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Reject unsupported complex YAML syntax
        if stripped.startswith("- "):
            return None, content, f"Unsupported YAML sequence at line {line_num}: lists not supported in flat frontmatter"
        if ":" not in stripped:
            return None, content, f"Malformed frontmatter line {line_num}: missing ':' key-value separator"

        key, val = stripped.split(":", 1)
        key = key.strip()
        val = val.strip()

        # Reject nested dictionary indentation
        if line.startswith(" ") or line.startswith("\t"):
            return None, content, f"Unsupported nested YAML structure at line {line_num}"

        # Strip optional surrounding quotes
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            val = val[1:-1]

        frontmatter[key] = val

    return frontmatter, body_text, None


def tokenize_for_jaccard(text: str) -> set[str]:
    """Tokenize text into lowercase word tokens for lexical Jaccard comparison."""
    words = re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())
    return set(words)


def compute_distribution(
    values: list[int],
) -> dict[str, float | int]:
    """Compute min, max, mean, median safely for integer metric lists."""
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
    }


def get_display_directory(data_dir: Path) -> str:
    """Return a normalized relative path representation to prevent exposing machine-specific absolute paths."""
    if not data_dir.is_absolute():
        return data_dir.as_posix()
    try:
        rel = data_dir.resolve().relative_to(Path.cwd().resolve())
        return rel.as_posix() if rel.as_posix() != "." else "./"
    except (ValueError, RuntimeError):
        return data_dir.name or "data_dir"


def scan_directory(data_dir: Path) -> dict[str, Any]:
    """
    Perform a read-only reconnaissance scan on the target directory.
    Returns structured results adhering to SPEC-01 Section 8.3 schema.
    """
    display_dir = get_display_directory(data_dir)
    resolved_dir = data_dir.resolve()
    timestamp = datetime.now(timezone.utc).isoformat()

    results: dict[str, Any] = {
        "scan_metadata": {
            "timestamp": timestamp,
            "scanned_directory": display_dir,
            "total_files": 0,
        },
        "counts_by_extension": {},
        "counts_by_tenant": {},
        "counts_by_access_tier": {},
        "detected_uninspected_pdfs": [],
        "length_statistics": {
            "character_counts": {"min": 0, "max": 0, "mean": 0.0, "median": 0.0},
            "word_counts": {"min": 0, "max": 0, "mean": 0.0, "median": 0.0},
            "estimated_tokens": {"min": 0, "max": 0, "mean": 0.0, "median": 0.0},
        },
        "exact_duplicates": [],
        "near_duplicate_candidates": [],
        "pii_findings": {
            "email_count": 0,
            "phone_count": 0,
            "credit_card_count": 0,
            "api_key_count": 0,
            "ssn_count": 0,
            "affected_documents": [],
        },
        "data_quality_issues": {
            "empty_files": [],
            "encoding_errors": [],
            "missing_frontmatter": [],
            "malformed_frontmatter": [],
            "missing_titles": [],
            "invalid_org_id": [],
            "invalid_access_tier": [],
        },
    }

    if not resolved_dir.exists() or not resolved_dir.is_dir():
        return results

    # Gather all file paths
    all_files = sorted([p for p in resolved_dir.rglob("*") if p.is_file()])
    results["scan_metadata"]["total_files"] = len(all_files)

    char_counts: list[int] = []
    word_counts: list[int] = []
    token_counts: list[int] = []

    # Map for exact duplicate hash detection
    hash_to_paths: dict[str, list[str]] = {}
    # Map for near-duplicate lexical comparison: relative_path -> set of tokens
    path_to_tokens: dict[str, set[str]] = {}

    affected_pii_docs: set[str] = set()

    for file_path in all_files:
        rel_path = file_path.relative_to(resolved_dir).as_posix()
        ext = file_path.suffix.lower() or "<none>"
        results["counts_by_extension"][ext] = (
            results["counts_by_extension"].get(ext, 0) + 1
        )

        file_size = file_path.stat().st_size
        if file_size == 0:
            results["data_quality_issues"]["empty_files"].append(rel_path)

        # PDF Handling: Detection only, NO content inspection
        if ext == ".pdf":
            results["detected_uninspected_pdfs"].append(
                {"relative_path": rel_path, "size_bytes": file_size}
            )
            continue

        # Only inspect text formats
        if ext not in {".md", ".txt", ".json"}:
            continue

        # Safe read with UTF-8 decoding
        try:
            raw_bytes = file_path.read_bytes()
            content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            results["data_quality_issues"]["encoding_errors"].append(
                {"relative_path": rel_path, "error": str(exc)}
            )
            continue

        # Exact duplicate detection via SHA-256
        sha256 = hashlib.sha256(raw_bytes).hexdigest()
        hash_to_paths.setdefault(sha256, []).append(rel_path)

        # Parse lightweight frontmatter
        fm, body, fm_error = parse_lightweight_frontmatter(content)
        if fm_error:
            results["data_quality_issues"]["malformed_frontmatter"].append(
                {"relative_path": rel_path, "error": fm_error}
            )
        elif fm is None:
            results["data_quality_issues"]["missing_frontmatter"].append(
                rel_path
            )
        else:
            # Validate frontmatter fields
            org_id = fm.get("org_id", "").strip()
            if not org_id or not UUID_REGEX.match(org_id):
                results["data_quality_issues"]["invalid_org_id"].append(
                    {"relative_path": rel_path, "org_id": org_id}
                )
            else:
                results["counts_by_tenant"][org_id] = (
                    results["counts_by_tenant"].get(org_id, 0) + 1
                )

            access_level = fm.get("access_level", "").strip()
            if access_level not in VALID_ACCESS_LEVELS:
                results["data_quality_issues"]["invalid_access_tier"].append(
                    {"relative_path": rel_path, "access_level": access_level}
                )
            else:
                results["counts_by_access_tier"][access_level] = (
                    results["counts_by_access_tier"].get(access_level, 0) + 1
                )

            title = fm.get("title", "").strip()
            if not title:
                results["data_quality_issues"]["missing_titles"].append(
                    rel_path
                )

        # Quantitative text metrics (on body text if frontmatter present, else whole content)
        eval_text = body if fm is not None else content
        chars = len(eval_text)
        words = len(eval_text.split())
        est_tokens = round(words * 1.33)

        char_counts.append(chars)
        word_counts.append(words)
        token_counts.append(est_tokens)

        # Token set for near-duplicate Jaccard analysis (only non-empty token sets)
        tokens = tokenize_for_jaccard(eval_text)
        if tokens:
            path_to_tokens[rel_path] = tokens

        # PII regex safety audit (counts only, ZERO sensitive values exposed)
        doc_has_pii = False
        for pii_name, pattern in PII_PATTERNS.items():
            matches = pattern.findall(eval_text)
            if matches:
                doc_has_pii = True
                results["pii_findings"][f"{pii_name}_count"] += len(matches)

        if doc_has_pii:
            affected_pii_docs.add(rel_path)

    results["pii_findings"]["affected_documents"] = sorted(
        list(affected_pii_docs)
    )

    # Compile length statistics
    results["length_statistics"]["character_counts"] = compute_distribution(
        char_counts
    )
    results["length_statistics"]["word_counts"] = compute_distribution(
        word_counts
    )
    results["length_statistics"]["estimated_tokens"] = compute_distribution(
        token_counts
    )

    # Compile exact duplicates (only hashes with >= 2 files)
    for sha, paths in hash_to_paths.items():
        if len(paths) > 1:
            results["exact_duplicates"].append(
                {"sha256": sha, "files": sorted(paths)}
            )

    # Compute near duplicates using pairwise Jaccard index (lexical heuristic)
    token_paths = sorted(path_to_tokens.keys())
    for i in range(len(token_paths)):
        for j in range(i + 1, len(token_paths)):
            p1, p2 = token_paths[i], token_paths[j]
            s1, s2 = path_to_tokens[p1], path_to_tokens[p2]
            intersection_size = len(s1 & s2)
            union_size = len(s1 | s2)
            if union_size > 0:
                jaccard = intersection_size / union_size
                if jaccard >= 0.85:
                    results["near_duplicate_candidates"].append(
                        {
                            "file_a": p1,
                            "file_b": p2,
                            "jaccard_similarity": round(jaccard, 4),
                        }
                    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M1 Dataset Reconnaissance Scanner"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Path to dataset directory (defaults to data/raw)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional destination path for JSON report (defaults to stdout if omitted)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Format JSON output with 2-space indentation",
    )
    args = parser.parse_args()

    results = scan_directory(args.data_dir)
    indent = 2 if args.pretty else None
    json_str = json.dumps(results, indent=indent)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json_str, encoding="utf-8")
    else:
        sys.stdout.write(json_str + "\n")


if __name__ == "__main__":
    main()

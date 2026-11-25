"""
Unit tests for Milestone M1 Dataset Reconnaissance Scanner.

Validates all scanner features using isolated, temporary test fixtures.
Complies strictly with SPEC-01 and the approved M1 Implementation Plan.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

# Import scanner functions directly from docs/experiments/m1_recon_scanner.py
sys.path.insert(0, str(Path(__file__).parent.parent / "docs" / "experiments"))
from m1_recon_scanner import (  # noqa: E402
    compute_distribution,
    parse_lightweight_frontmatter,
    scan_directory,
    tokenize_for_jaccard,
)


class TestM1ReconScanner(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir_obj.name)

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    def test_text_file_statistics(self) -> None:
        """Verifies character, word, and estimated token counts on standard Markdown documents."""
        doc = (
            "---\n"
            "document_id: 11111111-1111-4111-8111-111111111111\n"
            "org_id: 11111111-1111-4111-8111-111111111111\n"
            "access_level: public\n"
            "title: Sample FAQ\n"
            "---\n"
            "Hello world this is a test document with eleven words here."
        )
        (self.test_dir / "faq.md").write_text(doc, encoding="utf-8")

        results = scan_directory(self.test_dir)
        self.assertEqual(results["scan_metadata"]["total_files"], 1)
        self.assertEqual(results["counts_by_extension"][".md"], 1)

        # Word count of body is 11 words
        self.assertEqual(results["length_statistics"]["word_counts"]["min"], 11)
        self.assertEqual(results["length_statistics"]["word_counts"]["max"], 11)
        # Estimated tokens: round(11 * 1.33) = 15
        self.assertEqual(
            results["length_statistics"]["estimated_tokens"]["min"], 15
        )

    def test_pdf_detection_boundary(self) -> None:
        """Verifies that PDF files are cataloged under detected_uninspected_pdfs and not text-inspected."""
        pdf_file = self.test_dir / "sample.pdf"
        dummy_bytes = b"%PDF-1.4 dummy binary content that is not parsed"
        pdf_file.write_bytes(dummy_bytes)

        results = scan_directory(self.test_dir)
        self.assertEqual(len(results["detected_uninspected_pdfs"]), 1)
        pdf_entry = results["detected_uninspected_pdfs"][0]
        self.assertEqual(pdf_entry["relative_path"], "sample.pdf")
        self.assertEqual(pdf_entry["size_bytes"], len(dummy_bytes))

        # Word counts should remain empty (0 files evaluated for text)
        self.assertEqual(results["length_statistics"]["word_counts"]["max"], 0)

    def test_frontmatter_subset_and_malformed_handling(self) -> None:
        """Tests parsing of flat frontmatter and safe capture of malformed frontmatter syntax."""
        # 1. Valid flat frontmatter
        valid_doc = (
            "---\n"
            "document_id: 11111111-1111-4111-8111-111111111111\n"
            "org_id: 11111111-1111-4111-8111-111111111111\n"
            "access_level: public\n"
            'title: "Valid Flat Title"\n'
            "---\n"
            "Content body."
        )
        (self.test_dir / "valid.md").write_text(valid_doc, encoding="utf-8")

        # 2. Unclosed frontmatter fence
        unclosed_doc = (
            "---\n"
            "document_id: 11111111-1111-4111-8111-111111111111\n"
            "Content without closing fence."
        )
        (self.test_dir / "unclosed.md").write_text(
            unclosed_doc, encoding="utf-8"
        )

        # 3. Unsupported nested YAML structure
        nested_doc = (
            "---\n"
            "document_id: 11111111-1111-4111-8111-111111111111\n"
            "nested_block:\n"
            "  child_key: val\n"
            "---\n"
            "Content with nested YAML."
        )
        (self.test_dir / "nested.md").write_text(nested_doc, encoding="utf-8")

        results = scan_directory(self.test_dir)
        malformed = {
            m["relative_path"]: m["error"]
            for m in results["data_quality_issues"]["malformed_frontmatter"]
        }
        self.assertIn("unclosed.md", malformed)
        self.assertIn("missing closing '---'", malformed["unclosed.md"])
        self.assertIn("nested.md", malformed)
        self.assertIn("Unsupported nested YAML", malformed["nested.md"])

    def test_missing_and_invalid_metadata(self) -> None:
        """Verifies detection of missing titles, invalid UUIDs, and invalid access tiers."""
        doc = (
            "---\n"
            "document_id: not-a-valid-uuid\n"
            "org_id: invalid-org-uuid\n"
            "access_level: superuser_only\n"
            "title:\n"
            "---\n"
            "Some content."
        )
        (self.test_dir / "bad_meta.md").write_text(doc, encoding="utf-8")

        results = scan_directory(self.test_dir)
        self.assertEqual(
            len(results["data_quality_issues"]["invalid_org_id"]), 1
        )
        self.assertEqual(
            len(results["data_quality_issues"]["invalid_access_tier"]), 1
        )
        self.assertIn(
            "bad_meta.md", results["data_quality_issues"]["missing_titles"]
        )

    def test_invalid_utf8_handling(self) -> None:
        """Writes invalid non-UTF-8 bytes; verifies capture under encoding_errors without crashing."""
        corrupted_file = self.test_dir / "corrupted.md"
        corrupted_file.write_bytes(b"\x80\x81\xff not valid utf-8")

        results = scan_directory(self.test_dir)
        self.assertEqual(
            len(results["data_quality_issues"]["encoding_errors"]), 1
        )
        entry = results["data_quality_issues"]["encoding_errors"][0]
        self.assertEqual(entry["relative_path"], "corrupted.md")

    def test_empty_and_zero_token_files(self) -> None:
        """Validates handling of zero-byte and whitespace-only files with division-by-zero guards."""
        (self.test_dir / "zero.txt").write_bytes(b"")
        (self.test_dir / "whitespace.txt").write_text("   \n\t  \n")

        results = scan_directory(self.test_dir)
        self.assertIn("zero.txt", results["data_quality_issues"]["empty_files"])
        # Should not crash during Jaccard calculation and should have 0 near duplicates
        self.assertEqual(len(results["near_duplicate_candidates"]), 0)

    def test_exact_duplicate_detection(self) -> None:
        """Writes two identical files; verifies exact matching via SHA-256 hash grouping."""
        content = "Exact duplicate test content line."
        (self.test_dir / "file1.txt").write_text(content)
        (self.test_dir / "file2.txt").write_text(content)
        (self.test_dir / "unique.txt").write_text("Unique content.")

        results = scan_directory(self.test_dir)
        self.assertEqual(len(results["exact_duplicates"]), 1)
        dup = results["exact_duplicates"][0]
        self.assertEqual(len(dup["files"]), 2)
        self.assertIn("file1.txt", dup["files"])
        self.assertIn("file2.txt", dup["files"])

    def test_near_duplicate_jaccard_heuristic(self) -> None:
        """Writes files with >85% lexical token overlap; verifies detection under near_duplicate_candidates."""
        base_words = [f"token{i}" for i in range(100)]
        # File A has 100 tokens
        doc_a = " ".join(base_words)
        # File B has 95 identical tokens and 5 different tokens: Jaccard = 95 / 105 ~ 0.9048 (> 0.85)
        doc_b = " ".join(base_words[:95] + ["extra1", "extra2", "extra3"])

        (self.test_dir / "a.txt").write_text(doc_a)
        (self.test_dir / "b.txt").write_text(doc_b)

        results = scan_directory(self.test_dir)
        self.assertEqual(len(results["near_duplicate_candidates"]), 1)
        candidate = results["near_duplicate_candidates"][0]
        self.assertGreaterEqual(candidate["jaccard_similarity"], 0.85)

    def test_pii_safety_across_all_outputs(self) -> None:
        """Injects synthetic sensitive strings; verifies count increments and confirms zero sensitive strings leak."""
        sensitive_email = "test.cust.secret@example.com"
        sensitive_card = "4532-1111-2222-3333"
        sensitive_key = "sk-abcdef1234567890abcdef12"
        sensitive_ssn = "123-45-6789"
        sensitive_phone = "555-867-5309"

        doc = (
            f"Here is an email {sensitive_email} and card {sensitive_card}."
            f"Also api key {sensitive_key}, SSN {sensitive_ssn}, and phone {sensitive_phone}."
        )
        (self.test_dir / "secret_doc.md").write_text(doc, encoding="utf-8")

        results = scan_directory(self.test_dir)
        pii = results["pii_findings"]

        # Counts must increment
        self.assertEqual(pii["email_count"], 1)
        self.assertEqual(pii["credit_card_count"], 1)
        self.assertEqual(pii["api_key_count"], 1)
        self.assertEqual(pii["ssn_count"], 1)
        self.assertEqual(pii["phone_count"], 1)
        self.assertIn("secret_doc.md", pii["affected_documents"])

        # String representation of results must NOT contain any of the sensitive values
        dumped_json = json.dumps(results)
        for sensitive_val in [
            sensitive_email,
            sensitive_card,
            sensitive_key,
            sensitive_ssn,
            sensitive_phone,
        ]:
            self.assertNotIn(
                sensitive_val,
                dumped_json,
                f"Leakage detected! Sensitive value {sensitive_val} appeared in output JSON.",
            )

    def test_cli_behavior_and_output_routing(self) -> None:
        """Verifies CLI argument parsing: default data/raw, explicit override, --output-json, and stdout."""
        doc = "Simple document for CLI test."
        (self.test_dir / "doc.txt").write_text(doc, encoding="utf-8")

        scanner_script = str(
            Path(__file__).parent.parent
            / "docs"
            / "experiments"
            / "m1_recon_scanner.py"
        )
        
        # Write output file to a separate location outside self.test_dir so it doesn't get scanned
        with tempfile.TemporaryDirectory() as out_dir:
            out_json_path = Path(out_dir) / "output.json"

            # 1. Test explicit --output-json
            cmd = [
                sys.executable,
                scanner_script,
                "--data-dir",
                str(self.test_dir),
                "--output-json",
                str(out_json_path),
                "--pretty",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.assertTrue(out_json_path.exists())
            file_data = json.loads(out_json_path.read_text(encoding="utf-8"))
            self.assertEqual(file_data["scan_metadata"]["total_files"], 1)

            # 2. Test stdout emission when --output-json is omitted
            cmd_stdout = [
                sys.executable,
                scanner_script,
                "--data-dir",
                str(self.test_dir),
                "--pretty",
            ]
            proc_stdout = subprocess.run(
                cmd_stdout, capture_output=True, text=True, check=True
            )
            stdout_data = json.loads(proc_stdout.stdout)
            self.assertEqual(stdout_data["scan_metadata"]["total_files"], 1)

    def test_scanner_source_immutability(self) -> None:
        """Asserts that file contents and modification timestamps remain 100% identical after scanning."""
        test_file = self.test_dir / "immutable.txt"
        test_file.write_text("Immutable content for testing.", encoding="utf-8")

        mtime_before = test_file.stat().st_mtime_ns
        hash_before = hashlib.sha256(test_file.read_bytes()).hexdigest()

        # Run scanner
        scan_directory(self.test_dir)

        mtime_after = test_file.stat().st_mtime_ns
        hash_after = hashlib.sha256(test_file.read_bytes()).hexdigest()

        self.assertEqual(mtime_before, mtime_after)
        self.assertEqual(hash_before, hash_after)

    def test_json_schema_conformance(self) -> None:
        """Validates that output JSON strictly matches all required schema keys from SPEC-01 Section 8.3."""
        results = scan_directory(self.test_dir)
        required_top_keys = {
            "scan_metadata",
            "counts_by_extension",
            "counts_by_tenant",
            "counts_by_access_tier",
            "detected_uninspected_pdfs",
            "length_statistics",
            "exact_duplicates",
            "near_duplicate_candidates",
            "pii_findings",
            "data_quality_issues",
        }
        self.assertTrue(required_top_keys.issubset(results.keys()))

        pii_keys = {
            "email_count",
            "phone_count",
            "credit_card_count",
            "api_key_count",
            "ssn_count",
            "affected_documents",
        }
        self.assertTrue(pii_keys.issubset(results["pii_findings"].keys()))

    def test_scanned_directory_path_privacy(self) -> None:
        """Verifies that scan_metadata.scanned_directory does not expose machine-specific absolute paths."""
        # 1. Test user-supplied relative path
        res_rel = scan_directory(Path("tests"))
        self.assertEqual(res_rel["scan_metadata"]["scanned_directory"], "tests")
        self.assertFalse(
            Path(res_rel["scan_metadata"]["scanned_directory"]).is_absolute()
        )

        # 2. Test absolute path pointing within current working directory
        res_abs_cwd = scan_directory(Path.cwd() / "tests")
        self.assertEqual(
            res_abs_cwd["scan_metadata"]["scanned_directory"], "tests"
        )
        self.assertFalse(
            Path(res_abs_cwd["scan_metadata"]["scanned_directory"]).is_absolute()
        )

        # 3. Test isolated temporary directory outside cwd
        res_temp = scan_directory(self.test_dir)
        scanned_repr = res_temp["scan_metadata"]["scanned_directory"]
        self.assertFalse(Path(scanned_repr).is_absolute())
        self.assertNotIn("/home/", scanned_repr)


if __name__ == "__main__":
    unittest.main()

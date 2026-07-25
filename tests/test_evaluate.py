"""
Tests for ID-VLM Evaluation Harness
Tests CER computation, field accuracy, JSON parsing, report generation,
comparison reports, and tamper detection evaluation.
"""

import os
import sys
import json
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.evaluate import (
    compute_cer,
    compute_field_accuracy,
    parse_vlm_json_output,
    evaluate_predictions,
    generate_comparison_report,
    format_report_table,
    save_report,
    evaluate_tamper_predictions,
    _levenshtein_distance,
)


# ──────────────────────────────────────────────
# CER Tests
# ──────────────────────────────────────────────

class TestCER:
    """Tests for Character Error Rate computation."""

    def test_exact_match(self):
        """CER should be 0.0 for identical strings."""
        assert compute_cer("SMITH", "SMITH") == 0.0

    def test_case_insensitive(self):
        """CER should be 0.0 for case-different but same strings."""
        assert compute_cer("smith", "SMITH") == 0.0

    def test_complete_mismatch(self):
        """CER should be high for completely different strings."""
        cer = compute_cer("XXXXX", "SMITH")
        assert cer > 0.5

    def test_single_char_error(self):
        """CER for one character error in 5-char string."""
        cer = compute_cer("SMITT", "SMITH")
        assert abs(cer - 0.2) < 0.01  # 1 edit / 5 chars

    def test_empty_both(self):
        """CER should be 0.0 when both strings are empty."""
        assert compute_cer("", "") == 0.0

    def test_empty_ground_truth(self):
        """CER should be 1.0 when ground truth is empty but prediction is not."""
        assert compute_cer("SMITH", "") == 1.0

    def test_empty_prediction(self):
        """CER should be 1.0 when prediction is empty."""
        cer = compute_cer("", "SMITH")
        assert cer == 1.0  # 5 deletions / 5 chars

    def test_whitespace_handling(self):
        """CER should strip whitespace before comparison."""
        assert compute_cer("  SMITH  ", "SMITH") == 0.0

    def test_partial_match(self):
        """Test CER for partial match (insertion)."""
        cer = compute_cer("SMITHX", "SMITH")
        assert abs(cer - 0.2) < 0.01  # 1 insertion / 5 chars

    def test_date_format(self):
        """Test CER on date strings."""
        assert compute_cer("15-03-1990", "15-03-1990") == 0.0
        cer = compute_cer("15/03/1990", "15-03-1990")
        assert cer > 0.0  # Different separators


class TestLevenshtein:
    """Tests for the Levenshtein distance helper."""

    def test_identical(self):
        assert _levenshtein_distance("abc", "abc") == 0

    def test_insertion(self):
        assert _levenshtein_distance("abc", "ab") == 1

    def test_deletion(self):
        assert _levenshtein_distance("ab", "abc") == 1

    def test_substitution(self):
        assert _levenshtein_distance("abc", "axc") == 1

    def test_empty_strings(self):
        assert _levenshtein_distance("", "") == 0
        assert _levenshtein_distance("abc", "") == 3
        assert _levenshtein_distance("", "abc") == 3


# ──────────────────────────────────────────────
# Field Accuracy Tests
# ──────────────────────────────────────────────

class TestFieldAccuracy:
    """Tests for field-level accuracy computation."""

    def test_perfect_match(self):
        """All fields match exactly."""
        gt = {"surname": "SMITH", "birth_date": "15-03-1990"}
        pred = {"surname": "SMITH", "birth_date": "15-03-1990"}

        result = compute_field_accuracy(pred, gt)
        assert result["exact_match_rate"] == 1.0
        assert result["mean_cer"] == 0.0
        assert result["all_correct"] is True

    def test_all_wrong(self):
        """No fields match."""
        gt = {"surname": "SMITH", "birth_date": "15-03-1990"}
        pred = {"surname": "JONES", "birth_date": "01-01-2000"}

        result = compute_field_accuracy(pred, gt)
        assert result["exact_match_rate"] == 0.0
        assert result["mean_cer"] > 0.0
        assert result["all_correct"] is False

    def test_partial_match(self):
        """Some fields match, others don't."""
        gt = {"surname": "SMITH", "birth_date": "15-03-1990"}
        pred = {"surname": "SMITH", "birth_date": "WRONG"}

        result = compute_field_accuracy(pred, gt)
        assert result["exact_match_rate"] == 0.5

    def test_missing_fields(self):
        """Predicted output is missing some fields."""
        gt = {"surname": "SMITH", "birth_date": "15-03-1990", "doc_num": "AB123"}
        pred = {"surname": "SMITH"}

        result = compute_field_accuracy(pred, gt)
        assert result["fields_missing"] == 2
        assert result["fields_found"] == 1

    def test_extra_fields(self):
        """Predicted output has extra fields not in ground truth."""
        gt = {"surname": "SMITH"}
        pred = {"surname": "SMITH", "extra_field": "VALUE"}

        result = compute_field_accuracy(pred, gt)
        assert result["fields_extra"] == 1
        assert result["exact_match_rate"] == 1.0  # GT fields are all correct

    def test_empty_prediction(self):
        """Empty prediction against non-empty ground truth."""
        gt = {"surname": "SMITH", "birth_date": "15-03-1990"}
        pred = {}

        result = compute_field_accuracy(pred, gt)
        assert result["exact_match_rate"] == 0.0
        assert result["fields_missing"] == 2

    def test_empty_both(self):
        """Both empty."""
        result = compute_field_accuracy({}, {})
        assert result["exact_match_rate"] == 0.0
        assert result["all_correct"] is False

    def test_non_string_field_values(self):
        """Test handling of non-string values (dicts, ints, lists, None)."""
        gt = {"surname": "SMITH", "age": "30"}
        pred = {"surname": {"name": "SMITH"}, "age": 30}

        result = compute_field_accuracy(pred, gt)
        assert result["exact_match_rate"] == 0.5  # age matches '30' vs '30'
        assert isinstance(result["per_field"]["surname"]["predicted"], str)


# ──────────────────────────────────────────────
# JSON Parsing Tests
# ──────────────────────────────────────────────

class TestVLMJsonParsing:
    """Tests for parsing VLM text output as JSON."""

    def test_clean_json(self):
        """Test parsing clean JSON output."""
        text = '{"surname": "SMITH", "birth_date": "15-03-1990"}'
        result, valid = parse_vlm_json_output(text)
        assert valid is True
        assert result["surname"] == "SMITH"

    def test_markdown_code_fence(self):
        """Test extracting JSON from markdown code fence."""
        text = '```json\n{"surname": "SMITH"}\n```'
        result, valid = parse_vlm_json_output(text)
        assert valid is True
        assert result["surname"] == "SMITH"

    def test_json_with_surrounding_text(self):
        """Test extracting JSON from text with preamble/epilogue."""
        text = 'Here is the result: {"surname": "SMITH"} Hope this helps!'
        result, valid = parse_vlm_json_output(text)
        assert valid is True
        assert result["surname"] == "SMITH"

    def test_single_quotes(self):
        """Test handling of single-quoted JSON (common VLM error)."""
        text = "{'surname': 'SMITH', 'birth_date': '15-03-1990'}"
        result, valid = parse_vlm_json_output(text)
        assert valid is True
        assert result["surname"] == "SMITH"

    def test_trailing_comma(self):
        """Test handling of trailing commas."""
        text = '{"surname": "SMITH", "birth_date": "15-03-1990",}'
        result, valid = parse_vlm_json_output(text)
        assert valid is True

    def test_empty_output(self):
        """Test handling of empty output."""
        result, valid = parse_vlm_json_output("")
        assert valid is False
        assert result is None

    def test_non_json_output(self):
        """Test handling of completely non-JSON output."""
        result, valid = parse_vlm_json_output("I cannot read this document clearly")
        assert valid is False

    def test_json_array_not_dict(self):
        """Test handling of JSON array (should not be valid for our use case)."""
        result, valid = parse_vlm_json_output('[1, 2, 3]')
        assert valid is False


# ──────────────────────────────────────────────
# Full Evaluation Pipeline Tests
# ──────────────────────────────────────────────

class TestEvaluationPipeline:
    """Tests for the end-to-end evaluation pipeline."""

    def test_evaluate_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        predictions = [
            {
                "raw_output": '{"surname": "SMITH", "birth_date": "15-03-1990"}',
                "metadata": {"doc_type": "alb_id", "capture_mode": "scan"},
            }
        ]
        ground_truths = [
            {
                "fields": {"surname": "SMITH", "birth_date": "15-03-1990"},
                "metadata": {"doc_type": "alb_id", "capture_mode": "scan"},
            }
        ]

        report = evaluate_predictions(predictions, ground_truths)

        assert report["overall"]["mean_exact_match"] == 1.0
        assert report["json_parse_rate"] == 1.0
        assert report["n_samples"] == 1

    def test_evaluate_broken_json(self):
        """Test evaluation when VLM output is not JSON."""
        predictions = [
            {
                "raw_output": "I see a document with name SMITH",
                "metadata": {"doc_type": "alb_id", "capture_mode": "photo"},
            }
        ]
        ground_truths = [
            {
                "fields": {"surname": "SMITH"},
                "metadata": {"doc_type": "alb_id", "capture_mode": "photo"},
            }
        ]

        report = evaluate_predictions(predictions, ground_truths)
        assert report["json_parse_rate"] == 0.0

    def test_evaluate_multiple_samples(self):
        """Test evaluation with multiple samples and different doc types."""
        predictions = [
            {
                "raw_output": '{"surname": "SMITH"}',
                "metadata": {"doc_type": "alb_id", "capture_mode": "scan"},
            },
            {
                "raw_output": '{"surname": "WRONG"}',
                "metadata": {"doc_type": "esp_id", "capture_mode": "photo"},
            },
        ]
        ground_truths = [
            {
                "fields": {"surname": "SMITH"},
                "metadata": {"doc_type": "alb_id", "capture_mode": "scan"},
            },
            {
                "fields": {"surname": "DOE"},
                "metadata": {"doc_type": "esp_id", "capture_mode": "photo"},
            },
        ]

        report = evaluate_predictions(predictions, ground_truths)
        assert report["n_samples"] == 2
        assert "alb_id" in report["by_doc_type"]
        assert "esp_id" in report["by_doc_type"]
        assert "scan" in report["by_capture_mode"]
        assert "photo" in report["by_capture_mode"]

    def test_mismatched_counts_raises(self):
        """Test that mismatched prediction/GT counts raise an error."""
        with pytest.raises(AssertionError):
            evaluate_predictions(
                [{"raw_output": "{}"}],
                [{"fields": {}}, {"fields": {}}],
            )


# ──────────────────────────────────────────────
# Comparison Report Tests
# ──────────────────────────────────────────────

class TestComparisonReport:
    """Tests for baseline vs. fine-tuned comparison."""

    def test_comparison_report_structure(self):
        """Test that comparison report has expected structure."""
        baseline = {
            "overall": {"mean_exact_match": 0.3, "mean_cer": 0.4, "document_accuracy": 0.1},
            "json_parse_rate": 0.6,
            "by_doc_type": {},
            "by_capture_mode": {},
            "by_field": {},
        }
        finetuned = {
            "overall": {"mean_exact_match": 0.8, "mean_cer": 0.1, "document_accuracy": 0.6},
            "json_parse_rate": 0.95,
            "by_doc_type": {},
            "by_capture_mode": {},
            "by_field": {},
        }

        comparison = generate_comparison_report(baseline, finetuned)

        assert "overall" in comparison
        assert comparison["overall"]["delta"]["mean_exact_match"] == pytest.approx(0.5)
        assert comparison["json_parse_rate"]["delta"] == pytest.approx(0.35)


# ──────────────────────────────────────────────
# Report Formatting Tests
# ──────────────────────────────────────────────

class TestReportFormatting:
    """Tests for report formatting and saving."""

    def test_format_report_table(self):
        """Test that report formats as readable text."""
        report = {
            "overall": {
                "mean_exact_match": 0.75,
                "mean_cer": 0.12,
                "document_accuracy": 0.50,
                "n_samples": 100,
            },
            "json_parse_rate": 0.90,
            "n_samples": 100,
            "by_doc_type": {},
            "by_capture_mode": {},
            "by_field": {},
        }

        text = format_report_table(report)
        assert "75.0%" in text
        assert "90.0%" in text
        assert "100 samples" in text

    def test_save_report(self, tmp_path):
        """Test saving report to JSON file."""
        report = {
            "overall": {"mean_exact_match": 0.75},
            "json_parse_rate": 0.90,
            "n_samples": 10,
            "sample_results": [{"field_accuracy": {}}] * 10,
        }

        output_path = str(tmp_path / "report.json")
        save_report(report, output_path)

        assert os.path.exists(output_path)
        with open(output_path) as f:
            saved = json.load(f)
        assert "overall" in saved
        # sample_results should be stripped for file size
        assert "sample_results" not in saved


# ──────────────────────────────────────────────
# Tamper Detection Evaluation Tests
# ──────────────────────────────────────────────

class TestTamperEvaluation:
    """Tests for tamper detection evaluation."""

    def test_perfect_tamper_detection(self):
        """Test evaluation with perfect tamper predictions."""
        predictions = [
            {"raw_output": '{"verdict": "tampered", "suspicious_field": "birth_date"}'},
            {"raw_output": '{"verdict": "authentic"}'},
        ]
        ground_truths = [
            {"is_tampered": True, "tampered_field": "birth_date"},
            {"is_tampered": False, "tampered_field": None},
        ]

        result = evaluate_tamper_predictions(predictions, ground_truths)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        assert result["accuracy"] == 1.0

    def test_all_false_negatives(self):
        """Test when model misses all tampering."""
        predictions = [
            {"raw_output": '{"verdict": "authentic"}'},
        ]
        ground_truths = [
            {"is_tampered": True, "tampered_field": "surname"},
        ]

        result = evaluate_tamper_predictions(predictions, ground_truths)
        assert result["recall"] == 0.0
        assert result["fn"] == 1

    def test_field_identification(self):
        """Test that correct field identification is tracked."""
        predictions = [
            {"raw_output": '{"verdict": "tampered", "suspicious_field": "surname"}'},
        ]
        ground_truths = [
            {"is_tampered": True, "tampered_field": "surname"},
        ]

        result = evaluate_tamper_predictions(predictions, ground_truths)
        assert result["field_identification_accuracy"] == 1.0

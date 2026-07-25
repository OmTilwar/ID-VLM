"""
ID-VLM Evaluation Harness
Comprehensive evaluation metrics for VLM-based identity document field extraction:
  - Field-level exact match
  - Character Error Rate (CER) via Levenshtein distance
  - Document-level accuracy
  - JSON parse rate
  - Breakdowns by capture mode, document type, and field type
  - Before/after comparison report generation
"""

import os
import json
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ──────────────────────────────────────────────
# Core Metrics
# ──────────────────────────────────────────────

def compute_cer(predicted: str, ground_truth: str) -> float:
    """
    Compute Character Error Rate (CER) between two strings.

    CER = levenshtein_distance(predicted, ground_truth) / len(ground_truth)

    Uses dynamic programming for Levenshtein distance computation
    (no external dependency required, but python-Levenshtein is faster).

    Args:
        predicted: Model output string.
        ground_truth: Ground truth string.

    Returns:
        CER as a float in [0.0, ∞). 0.0 means exact match.
        Returns 0.0 if both strings are empty.
        Returns 1.0 if ground_truth is empty but predicted is not.
    """
    if not ground_truth and not predicted:
        return 0.0
    if not ground_truth:
        return 1.0

    # Normalize: strip whitespace, lowercase for fair comparison
    pred = predicted.strip().lower()
    gt = ground_truth.strip().lower()

    if pred == gt:
        return 0.0

    # Levenshtein distance via DP
    distance = _levenshtein_distance(pred, gt)
    return distance / len(gt)


def compute_field_accuracy(
    pred_fields: Dict[str, str],
    gt_fields: Dict[str, str],
    normalize: bool = True,
) -> Dict[str, Any]:
    """
    Compute per-field exact match accuracy between predicted and ground truth.

    Args:
        pred_fields: Dict of field_name → predicted value.
        gt_fields: Dict of field_name → ground truth value.
        normalize: If True, normalize strings (strip, lowercase) before comparison.

    Returns:
        Dict with:
            "per_field": {field_name: {"match": bool, "cer": float, "pred": str, "gt": str}}
            "exact_match_rate": float (fraction of fields that match exactly)
            "mean_cer": float (mean CER across all fields)
            "all_correct": bool (all fields match exactly)
    """
    results = {}

    # Evaluate all ground truth fields
    all_gt_fields = set(gt_fields.keys())
    all_pred_fields = set(pred_fields.keys())

    for field_name in all_gt_fields:
        gt_value = gt_fields[field_name]
        pred_value = pred_fields.get(field_name, "")

        if normalize:
            gt_norm = gt_value.strip().lower()
            pred_norm = pred_value.strip().lower()
        else:
            gt_norm = gt_value
            pred_norm = pred_value

        match = (gt_norm == pred_norm)
        cer = compute_cer(pred_value, gt_value)

        results[field_name] = {
            "match": match,
            "cer": cer,
            "predicted": pred_value,
            "ground_truth": gt_value,
        }

    # Aggregate
    n_fields = len(results)
    n_correct = sum(1 for r in results.values() if r["match"])
    mean_cer = np.mean([r["cer"] for r in results.values()]) if results else 0.0

    return {
        "per_field": results,
        "exact_match_rate": n_correct / n_fields if n_fields > 0 else 0.0,
        "mean_cer": float(mean_cer),
        "all_correct": (n_correct == n_fields) and n_fields > 0,
        "fields_found": len(all_pred_fields & all_gt_fields),
        "fields_missing": len(all_gt_fields - all_pred_fields),
        "fields_extra": len(all_pred_fields - all_gt_fields),
    }


def parse_vlm_json_output(raw_output: str) -> Tuple[Optional[Dict], bool]:
    """
    Attempt to parse a VLM's text output as JSON.

    VLMs sometimes wrap JSON in markdown code fences, add commentary,
    or produce slightly malformed JSON. This function tries several
    extraction strategies.

    Args:
        raw_output: Raw text output from the VLM.

    Returns:
        Tuple of (parsed_dict_or_None, is_valid_json_bool).
    """
    if not raw_output or not raw_output.strip():
        return None, False

    text = raw_output.strip()

    # Strategy 1: Direct JSON parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result, True
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code fence
    code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if code_block_match:
        try:
            result = json.loads(code_block_match.group(1).strip())
            if isinstance(result, dict):
                return result, True
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find the first { ... } block
    brace_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if brace_match:
        try:
            result = json.loads(brace_match.group(0))
            if isinstance(result, dict):
                return result, True
        except json.JSONDecodeError:
            pass

    # Strategy 4: Try fixing common JSON issues
    # (trailing commas, single quotes)
    cleaned = text.replace("'", '"')
    cleaned = re.sub(r",\s*}", "}", cleaned)
    cleaned = re.sub(r",\s*]", "]", cleaned)
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result, True
    except json.JSONDecodeError:
        pass

    return None, False


# ──────────────────────────────────────────────
# Full Evaluation Pipeline
# ──────────────────────────────────────────────

def evaluate_predictions(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run the full evaluation pipeline on a set of predictions.

    Args:
        predictions: List of dicts, each with:
            {"raw_output": str, "metadata": {"doc_type": str, "capture_mode": str, ...}}
        ground_truths: List of dicts, each with:
            {"fields": {"surname": "SMITH", ...}, "metadata": {...}}

    Returns:
        Comprehensive evaluation report dict.
    """
    assert len(predictions) == len(ground_truths), \
        f"Prediction count ({len(predictions)}) != ground truth count ({len(ground_truths)})"

    # Per-sample results
    sample_results = []
    json_parse_successes = 0

    # Aggregation buckets
    by_doc_type = defaultdict(list)
    by_capture_mode = defaultdict(list)
    by_field_name = defaultdict(list)

    for pred, gt in zip(predictions, ground_truths):
        raw_output = pred.get("raw_output", "")
        gt_fields = gt.get("fields", {})
        metadata = gt.get("metadata", pred.get("metadata", {}))
        doc_type = metadata.get("doc_type", "unknown")
        capture_mode = metadata.get("capture_mode", "unknown")

        # Parse VLM output
        parsed, is_valid = parse_vlm_json_output(raw_output)
        if is_valid:
            json_parse_successes += 1

        pred_fields = parsed if parsed else {}

        # Compute field accuracy
        field_result = compute_field_accuracy(pred_fields, gt_fields)

        sample_result = {
            "field_accuracy": field_result,
            "json_valid": is_valid,
            "doc_type": doc_type,
            "capture_mode": capture_mode,
            "raw_output": raw_output,
        }
        sample_results.append(sample_result)

        # Aggregate by doc type and capture mode
        by_doc_type[doc_type].append(field_result)
        by_capture_mode[capture_mode].append(field_result)

        # Aggregate by field name
        for field_name, field_data in field_result["per_field"].items():
            by_field_name[field_name].append(field_data)

    # Compute overall metrics
    n_total = len(sample_results)
    overall = _aggregate_results(sample_results)

    # Compute breakdowns
    doc_type_breakdown = {}
    for dt, results in by_doc_type.items():
        dt_samples = [s for s in sample_results if s["doc_type"] == dt]
        doc_type_breakdown[dt] = _aggregate_results(dt_samples)

    capture_mode_breakdown = {}
    for cm, results in by_capture_mode.items():
        cm_samples = [s for s in sample_results if s["capture_mode"] == cm]
        capture_mode_breakdown[cm] = _aggregate_results(cm_samples)

    field_breakdown = {}
    for fname, field_results in by_field_name.items():
        n_match = sum(1 for r in field_results if r["match"])
        mean_cer = np.mean([r["cer"] for r in field_results])
        field_breakdown[fname] = {
            "exact_match_rate": n_match / len(field_results),
            "mean_cer": float(mean_cer),
            "n_samples": len(field_results),
        }

    return {
        "overall": overall,
        "json_parse_rate": json_parse_successes / n_total if n_total > 0 else 0.0,
        "n_samples": n_total,
        "by_doc_type": doc_type_breakdown,
        "by_capture_mode": capture_mode_breakdown,
        "by_field": field_breakdown,
        "sample_results": sample_results,
    }


def generate_comparison_report(
    baseline_report: Dict[str, Any],
    finetuned_report: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate a before/after comparison between baseline and fine-tuned models.

    Args:
        baseline_report: Evaluation report from zero-shot baseline.
        finetuned_report: Evaluation report from fine-tuned model.

    Returns:
        Comparison report with deltas for all metrics.
    """
    comparison = {
        "overall": {
            "baseline": baseline_report["overall"],
            "finetuned": finetuned_report["overall"],
            "delta": _compute_delta(
                baseline_report["overall"],
                finetuned_report["overall"],
            ),
        },
        "json_parse_rate": {
            "baseline": baseline_report["json_parse_rate"],
            "finetuned": finetuned_report["json_parse_rate"],
            "delta": finetuned_report["json_parse_rate"] - baseline_report["json_parse_rate"],
        },
    }

    # Doc type comparison
    all_doc_types = set(
        list(baseline_report.get("by_doc_type", {}).keys())
        + list(finetuned_report.get("by_doc_type", {}).keys())
    )
    comparison["by_doc_type"] = {}
    for dt in all_doc_types:
        b = baseline_report.get("by_doc_type", {}).get(dt, {})
        f = finetuned_report.get("by_doc_type", {}).get(dt, {})
        comparison["by_doc_type"][dt] = {
            "baseline": b,
            "finetuned": f,
            "delta": _compute_delta(b, f),
        }

    # Capture mode comparison
    all_modes = set(
        list(baseline_report.get("by_capture_mode", {}).keys())
        + list(finetuned_report.get("by_capture_mode", {}).keys())
    )
    comparison["by_capture_mode"] = {}
    for cm in all_modes:
        b = baseline_report.get("by_capture_mode", {}).get(cm, {})
        f = finetuned_report.get("by_capture_mode", {}).get(cm, {})
        comparison["by_capture_mode"][cm] = {
            "baseline": b,
            "finetuned": f,
            "delta": _compute_delta(b, f),
        }

    # Field comparison
    all_fields = set(
        list(baseline_report.get("by_field", {}).keys())
        + list(finetuned_report.get("by_field", {}).keys())
    )
    comparison["by_field"] = {}
    for fn in all_fields:
        b = baseline_report.get("by_field", {}).get(fn, {})
        f = finetuned_report.get("by_field", {}).get(fn, {})
        comparison["by_field"][fn] = {
            "baseline": b,
            "finetuned": f,
            "delta": _compute_delta(b, f),
        }

    return comparison


def format_report_table(report: Dict[str, Any]) -> str:
    """
    Format an evaluation report as a readable text table.

    Args:
        report: Evaluation report from evaluate_predictions().

    Returns:
        Formatted string with tables.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("ID-VLM EVALUATION REPORT")
    lines.append("=" * 70)

    overall = report["overall"]
    lines.append(f"\nOverall Metrics ({report['n_samples']} samples):")
    lines.append(f"  Field Exact Match:     {overall.get('mean_exact_match', 0):.1%}")
    lines.append(f"  Mean CER:              {overall.get('mean_cer', 0):.4f}")
    lines.append(f"  Document-Level Acc:    {overall.get('document_accuracy', 0):.1%}")
    lines.append(f"  JSON Parse Rate:       {report['json_parse_rate']:.1%}")

    # By document type
    if report.get("by_doc_type"):
        lines.append(f"\nBy Document Type:")
        lines.append(f"  {'Type':<20} {'Exact Match':>12} {'CER':>8} {'Samples':>8}")
        lines.append(f"  {'-'*20} {'-'*12} {'-'*8} {'-'*8}")
        for dt, metrics in sorted(report["by_doc_type"].items()):
            lines.append(
                f"  {dt:<20} "
                f"{metrics.get('mean_exact_match', 0):>11.1%} "
                f"{metrics.get('mean_cer', 0):>7.4f} "
                f"{metrics.get('n_samples', 0):>8}"
            )

    # By capture mode
    if report.get("by_capture_mode"):
        lines.append(f"\nBy Capture Mode:")
        lines.append(f"  {'Mode':<20} {'Exact Match':>12} {'CER':>8} {'Samples':>8}")
        lines.append(f"  {'-'*20} {'-'*12} {'-'*8} {'-'*8}")
        for cm, metrics in sorted(report["by_capture_mode"].items()):
            lines.append(
                f"  {cm:<20} "
                f"{metrics.get('mean_exact_match', 0):>11.1%} "
                f"{metrics.get('mean_cer', 0):>7.4f} "
                f"{metrics.get('n_samples', 0):>8}"
            )

    # By field
    if report.get("by_field"):
        lines.append(f"\nBy Field Name:")
        lines.append(f"  {'Field':<20} {'Exact Match':>12} {'CER':>8} {'Samples':>8}")
        lines.append(f"  {'-'*20} {'-'*12} {'-'*8} {'-'*8}")
        for fn, metrics in sorted(report["by_field"].items()):
            lines.append(
                f"  {fn:<20} "
                f"{metrics.get('exact_match_rate', 0):>11.1%} "
                f"{metrics.get('mean_cer', 0):>7.4f} "
                f"{metrics.get('n_samples', 0):>8}"
            )

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def save_report(report: Dict[str, Any], output_path: str) -> None:
    """Save evaluation report to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Remove non-serializable sample_results to keep file small
    save_report = {k: v for k, v in report.items() if k != "sample_results"}
    save_report["n_sample_results"] = len(report.get("sample_results", []))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_report, f, indent=2, ensure_ascii=False, default=str)

    print(f"Report saved to {output_path}")


# ──────────────────────────────────────────────
# Tamper Detection Evaluation (Stretch)
# ──────────────────────────────────────────────

def evaluate_tamper_predictions(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluate tamper detection predictions.

    Args:
        predictions: List with {"raw_output": str, ...}
        ground_truths: List with {"is_tampered": bool, "tampered_field": str|None, ...}

    Returns:
        Evaluation report with precision, recall, F1 for tamper detection.
    """
    tp = fp = fn = tn = 0
    field_correct = 0
    field_total = 0

    for pred, gt in zip(predictions, ground_truths):
        parsed, is_valid = parse_vlm_json_output(pred.get("raw_output", ""))

        gt_tampered = gt.get("is_tampered", False)

        if parsed:
            pred_tampered = parsed.get("verdict", "").lower() == "tampered"
        else:
            pred_tampered = False

        if gt_tampered and pred_tampered:
            tp += 1
            # Check if the correct field was identified
            if gt.get("tampered_field") and parsed:
                pred_field = parsed.get("suspicious_field", "")
                if pred_field and pred_field.lower() == gt["tampered_field"].lower():
                    field_correct += 1
                field_total += 1
        elif gt_tampered and not pred_tampered:
            fn += 1
        elif not gt_tampered and pred_tampered:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "field_identification_accuracy": (
            field_correct / field_total if field_total > 0 else 0.0
        ),
        "n_samples": len(predictions),
    }


# ──────────────────────────────────────────────
# Internal Helpers
# ──────────────────────────────────────────────

def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def _aggregate_results(sample_results: List[Dict]) -> Dict[str, Any]:
    """Aggregate sample-level results into summary statistics."""
    if not sample_results:
        return {
            "mean_exact_match": 0.0,
            "mean_cer": 0.0,
            "document_accuracy": 0.0,
            "n_samples": 0,
        }

    exact_matches = [s["field_accuracy"]["exact_match_rate"] for s in sample_results]
    cers = [s["field_accuracy"]["mean_cer"] for s in sample_results]
    doc_correct = sum(1 for s in sample_results if s["field_accuracy"]["all_correct"])

    return {
        "mean_exact_match": float(np.mean(exact_matches)),
        "mean_cer": float(np.mean(cers)),
        "document_accuracy": doc_correct / len(sample_results),
        "n_samples": len(sample_results),
    }


def _compute_delta(baseline: Dict, finetuned: Dict) -> Dict[str, float]:
    """Compute deltas between baseline and fine-tuned metrics."""
    delta = {}
    for key in ["mean_exact_match", "mean_cer", "document_accuracy",
                 "exact_match_rate"]:
        b_val = baseline.get(key, 0.0) if baseline else 0.0
        f_val = finetuned.get(key, 0.0) if finetuned else 0.0
        delta[key] = f_val - b_val
    return delta

"""
CLI Script: Run Evaluation
Evaluate model predictions against ground truth using the evaluation harness.

Usage:
    python scripts/run_evaluation.py --predictions outputs/baseline/predictions.json --ground-truth data/processed/test.jsonl
    python scripts/run_evaluation.py --compare outputs/baseline/predictions.json outputs/finetuned/predictions.json --ground-truth data/processed/test.jsonl
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.evaluate import (
    evaluate_predictions,
    generate_comparison_report,
    format_report_table,
    save_report,
)
from src.dataset import load_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ID-VLM predictions"
    )
    parser.add_argument(
        "--predictions", type=str, required=True,
        help="Path to predictions JSON file"
    )
    parser.add_argument(
        "--ground-truth", type=str, required=True,
        help="Path to ground truth JSONL file"
    )
    parser.add_argument(
        "--compare", type=str, default=None,
        help="Path to second predictions file for comparison (e.g., fine-tuned)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=config.ANALYSIS_DIR,
        help="Directory to save evaluation reports"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ID-VLM Evaluation")
    print("=" * 60)

    # Load predictions
    with open(args.predictions, "r") as f:
        predictions = json.load(f)
    print(f"Loaded {len(predictions)} predictions from {args.predictions}")

    # Load ground truth
    gt_dataset = load_dataset(args.ground_truth)

    # Extract ground truth fields for evaluation
    ground_truths = []
    for sample in gt_dataset:
        gt_text = sample["messages"][1]["content"][0]["text"]
        try:
            fields = json.loads(gt_text)
        except json.JSONDecodeError:
            fields = {}
        ground_truths.append({
            "fields": fields,
            "metadata": sample.get("metadata", {}),
        })

    # Align counts
    n = min(len(predictions), len(ground_truths))
    predictions = predictions[:n]
    ground_truths = ground_truths[:n]

    # Evaluate
    print(f"\nEvaluating {n} samples...")
    report = evaluate_predictions(predictions, ground_truths)

    # Display
    print("\n" + format_report_table(report))

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    report_name = os.path.splitext(os.path.basename(args.predictions))[0]
    save_report(report, os.path.join(args.output_dir, f"report_{report_name}.json"))

    # Comparison mode
    if args.compare:
        print(f"\n{'=' * 60}")
        print("COMPARISON MODE")
        print(f"{'=' * 60}")

        with open(args.compare, "r") as f:
            finetuned_predictions = json.load(f)
        finetuned_predictions = finetuned_predictions[:n]

        finetuned_report = evaluate_predictions(finetuned_predictions, ground_truths)

        print("\nFine-tuned model:")
        print(format_report_table(finetuned_report))

        comparison = generate_comparison_report(report, finetuned_report)

        # Display deltas
        delta = comparison["overall"]["delta"]
        print(f"\n{'=' * 60}")
        print("IMPROVEMENT SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Exact Match:    {delta.get('mean_exact_match', 0):+.1%}")
        print(f"  CER:            {delta.get('mean_cer', 0):+.4f}")
        print(f"  Doc Accuracy:   {delta.get('document_accuracy', 0):+.1%}")
        print(f"  JSON Parse:     {comparison['json_parse_rate']['delta']:+.1%}")

        # Save comparison
        save_report(
            finetuned_report,
            os.path.join(args.output_dir, "report_finetuned.json"),
        )
        with open(os.path.join(args.output_dir, "comparison.json"), "w") as f:
            json.dump(comparison, f, indent=2, default=str)
        print(f"\nComparison saved to {args.output_dir}/comparison.json")


if __name__ == "__main__":
    main()

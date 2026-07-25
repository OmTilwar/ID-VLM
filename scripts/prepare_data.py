"""
CLI Script: Prepare MIDV-2020 Data
Converts raw MIDV-2020 annotations + images into Unsloth-compatible
instruction/answer pairs (JSONL format).

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --data-dir data/raw --max-samples 200
    python scripts/prepare_data.py --doc-types alb_id esp_id
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.dataset import create_dataset, split_dataset, save_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MIDV-2020 data for ID-VLM fine-tuning"
    )
    parser.add_argument(
        "--data-dir", type=str, default=config.RAW_DATA_DIR,
        help="Path to raw MIDV-2020 data directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default=config.PROCESSED_DATA_DIR,
        help="Path to save processed JSONL files"
    )
    parser.add_argument(
        "--doc-types", nargs="+", default=None,
        help="Document types to include (e.g., alb_id esp_id)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum total samples to generate"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=config.TRAIN_RATIO,
        help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=config.VAL_RATIO,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=config.TEST_RATIO,
        help="Test set ratio"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ID-VLM Data Preparation")
    print("=" * 60)
    print(f"Data directory:    {args.data_dir}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Document types:    {args.doc_types or config.SELECTED_DOC_TYPES}")
    print(f"Max samples:       {args.max_samples or 'all'}")
    print(f"Split ratios:      {args.train_ratio}/{args.val_ratio}/{args.test_ratio}")
    print()

    # Check data directory exists
    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        print(f"Please download MIDV-2020 subset to {args.data_dir}")
        print("See README.md for download instructions.")
        sys.exit(1)

    # Create dataset
    print("Step 1/3: Loading and converting annotations...")
    dataset = create_dataset(
        data_dir=args.data_dir,
        doc_types=args.doc_types,
        max_samples=args.max_samples,
    )

    if not dataset:
        print("Error: No instruction pairs generated. Check your data directory structure.")
        sys.exit(1)

    # Split
    print("\nStep 2/3: Splitting dataset...")
    train, val, test = split_dataset(
        dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    # Save
    print("\nStep 3/3: Saving JSONL files...")
    os.makedirs(args.output_dir, exist_ok=True)

    save_dataset(train, os.path.join(args.output_dir, "train.jsonl"))
    save_dataset(val, os.path.join(args.output_dir, "val.jsonl"))
    save_dataset(test, os.path.join(args.output_dir, "test.jsonl"))

    # Save full dataset too (for Colab upload)
    save_dataset(dataset, os.path.join(args.output_dir, "full.jsonl"))

    # Summary
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"  Train: {len(train)} samples")
    print(f"  Val:   {len(val)} samples")
    print(f"  Test:  {len(test)} samples")
    print(f"  Total: {len(dataset)} samples")
    print(f"\nFiles saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

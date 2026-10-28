"""
CLI Script: Apply Schema Prompts
Rewrite existing train/val/test JSONL splits so each sample's extraction prompt
lists the exact keys for its document type. Splits are kept as-is, so results
stay comparable with earlier runs on the same test set.

Usage:
    python scripts/apply_schema_prompts.py --data-dir data/processed
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.dataset import (
    load_dataset,
    save_dataset,
    collect_doc_type_schemas,
    apply_schema_prompts,
)

SPLITS = ["train", "val", "test", "full"]


def main():
    parser = argparse.ArgumentParser(description="Add per-document-type schema prompts to JSONL splits")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing train/val/test JSONL files")
    args = parser.parse_args()

    splits = {}
    for name in SPLITS:
        path = os.path.join(args.data_dir, f"{name}.jsonl")
        if os.path.exists(path):
            splits[name] = load_dataset(path)

    if not splits:
        sys.exit(f"No JSONL splits found in {args.data_dir}")

    # One schema per document type, built from every split so all keys are covered
    all_samples = [s for samples in splits.values() for s in samples]
    schemas = collect_doc_type_schemas(all_samples)

    print("\nSchemas per document type:")
    for doc_type, keys in sorted(schemas.items()):
        print(f"  {doc_type:<22} {len(keys):>2} keys: {', '.join(keys)}")

    for name, samples in splits.items():
        apply_schema_prompts(samples, schemas)
        save_dataset(samples, os.path.join(args.data_dir, f"{name}.jsonl"))


if __name__ == "__main__":
    main()

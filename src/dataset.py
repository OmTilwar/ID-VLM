"""
ID-VLM Dataset Module
Converts MIDV-2020 raw annotations into Unsloth-compatible instruction/answer
pairs for VQA-style fine-tuning of Qwen2-VL.

MIDV-2020 annotation format (VIA v2 compatible JSON):
    Each JSON file maps image filenames to regions with field_name/value pairs
    and bounding box coordinates.

Output format (ChatML for Unsloth FastVisionModel):
    {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": <PIL.Image or path>},
                {"type": "text", "text": <extraction prompt>}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": <JSON string of field values>}
            ]}
        ]
    }
"""

import os
import json
import glob
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ──────────────────────────────────────────────
# Annotation Parsing
# ──────────────────────────────────────────────

def parse_midv_annotation(json_path: str) -> Dict[str, Any]:
    """
    Parse a MIDV-2020 annotation JSON file.

    MIDV-2020 uses VIA v2 format where each image entry contains
    'regions' with 'region_attributes' holding field_name and value,
    plus 'shape_attributes' for bounding boxes.

    Args:
        json_path: Path to the annotation JSON file.

    Returns:
        Dict mapping image filenames to their field annotations:
        {
            "image_001.jpg": {
                "fields": {"surname": "SMITH", "birth_date": "15-03-1990", ...},
                "bboxes": {"surname": [x, y, w, h], ...},
                "doc_type": "alb_id"
            }
        }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    parsed = {}

    # MIDV-2020 JSON structure: top-level keys can be image filenames or
    # a nested structure. Handle both common formats.
    annotations = raw if isinstance(raw, dict) else {}

    # If the JSON has a top-level wrapper (e.g., "_via_img_metadata")
    if "_via_img_metadata" in annotations:
        annotations = annotations["_via_img_metadata"]

    for key, entry in annotations.items():
        # Skip non-image entries (metadata keys)
        if not isinstance(entry, dict):
            continue

        filename = entry.get("filename", key)
        regions = entry.get("regions", [])

        if not regions:
            continue

        fields = {}
        bboxes = {}

        for region in regions:
            region_attrs = region.get("region_attributes", {})
            shape_attrs = region.get("shape_attributes", {})

            field_name = region_attrs.get("field_name", "")
            field_value = region_attrs.get("value", "")

            if field_name and field_value:
                # Normalize field name to snake_case
                normalized_name = _normalize_field_name(field_name)
                fields[normalized_name] = field_value

                # Extract bounding box if available
                if shape_attrs.get("name") == "rect":
                    bboxes[normalized_name] = [
                        shape_attrs.get("x", 0),
                        shape_attrs.get("y", 0),
                        shape_attrs.get("width", 0),
                        shape_attrs.get("height", 0),
                    ]

        if fields:
            # Infer doc_type from path
            doc_type = _infer_doc_type(json_path)
            parsed[filename] = {
                "fields": fields,
                "bboxes": bboxes,
                "doc_type": doc_type,
            }

    return parsed


def parse_midv_ground_truth(gt_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Parse MIDV-2020 ground truth from the per-document text field files.

    Many MIDV-2020 distributions store ground truth as individual text files
    or a consolidated JSON per document type, rather than VIA format.

    Args:
        gt_dir: Directory containing ground truth files for a document type.

    Returns:
        Dict mapping document IDs to field values:
        {"00001": {"surname": "SMITH", "given_name": "JOHN", ...}}
    """
    ground_truth = {}

    # Try JSON ground truth first
    json_files = glob.glob(os.path.join(gt_dir, "*.json"))
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for doc_id, fields in data.items():
                if isinstance(fields, dict):
                    ground_truth[str(doc_id)] = {
                        _normalize_field_name(k): str(v)
                        for k, v in fields.items()
                    }

    # Try text-file-based ground truth (one file per field per document)
    txt_files = glob.glob(os.path.join(gt_dir, "**", "*.txt"), recursive=True)
    for tf in txt_files:
        parts = Path(tf).stem.split("_")
        if len(parts) >= 2:
            doc_id = parts[0]
            field_name = "_".join(parts[1:])
            with open(tf, "r", encoding="utf-8") as f:
                value = f.read().strip()
            if doc_id not in ground_truth:
                ground_truth[doc_id] = {}
            ground_truth[doc_id][_normalize_field_name(field_name)] = value

    return ground_truth


# ──────────────────────────────────────────────
# Instruction Pair Generation
# ──────────────────────────────────────────────

def build_instruction_pair(
    image_path: str,
    fields: Dict[str, str],
    doc_type: str,
    prompt: Optional[str] = None,
    use_pil: bool = False,
) -> Dict[str, Any]:
    """
    Create a ChatML-format instruction/answer pair for VLM fine-tuning.

    Args:
        image_path: Path to the document image.
        fields: Dict of field_name → value (ground truth).
        doc_type: Document type code (e.g., "alb_id").
        prompt: Custom extraction prompt. Defaults to config.EXTRACTION_PROMPT.
        use_pil: If True, load image as PIL object; otherwise use path string.

    Returns:
        ChatML-format dict with messages list.
    """
    if prompt is None:
        prompt = config.EXTRACTION_PROMPT

    # Build the ground truth JSON response
    answer_json = json.dumps(fields, ensure_ascii=False, indent=None)

    # Image content — either PIL or path
    if use_pil:
        image_content = Image.open(image_path).convert("RGB")
    else:
        image_content = image_path

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_content},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer_json},
                ],
            },
        ],
        "metadata": {
            "doc_type": doc_type,
            "image_path": image_path,
            "capture_mode": get_capture_mode(image_path),
            "num_fields": len(fields),
        },
    }


def build_tamper_instruction_pair(
    image_path: str,
    is_tampered: bool,
    tampered_field: Optional[str] = None,
    tamper_type: Optional[str] = None,
    doc_type: str = "unknown",
    use_pil: bool = False,
) -> Dict[str, Any]:
    """
    Create a ChatML instruction/answer pair for tamper detection.

    Args:
        image_path: Path to the document image.
        is_tampered: Whether this document has been tampered with.
        tampered_field: Name of the tampered field (None if authentic).
        tamper_type: Type of tampering applied (None if authentic).
        doc_type: Document type code.
        use_pil: If True, load image as PIL object.

    Returns:
        ChatML-format dict for tamper detection task.
    """
    if is_tampered:
        answer = {
            "verdict": "tampered",
            "confidence": 0.95,
            "suspicious_field": tampered_field,
            "reason": f"{tamper_type} artifact detected in {tampered_field} field region",
        }
    else:
        answer = {
            "verdict": "authentic",
            "confidence": 0.95,
            "suspicious_field": None,
            "reason": "No signs of tampering detected",
        }

    answer_json = json.dumps(answer, ensure_ascii=False)

    if use_pil:
        image_content = Image.open(image_path).convert("RGB")
    else:
        image_content = image_path

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_content},
                    {"type": "text", "text": config.TAMPER_DETECTION_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer_json},
                ],
            },
        ],
        "metadata": {
            "doc_type": doc_type,
            "image_path": image_path,
            "is_tampered": is_tampered,
            "tampered_field": tampered_field,
            "tamper_type": tamper_type,
        },
    }


# ──────────────────────────────────────────────
# Dataset Construction
# ──────────────────────────────────────────────

def create_dataset(
    data_dir: str,
    doc_types: Optional[List[str]] = None,
    capture_modes: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Build the full instruction-pair dataset from MIDV-2020 raw data.

    Expects the following directory structure under data_dir:
        <doc_type>/
            images/
                <capture_mode>/
                    <doc_id>_<frame>.jpg
            ground_truth/
                <annotations>.json or <doc_id>_<field>.txt

    Args:
        data_dir: Root directory of raw MIDV-2020 data.
        doc_types: List of document type codes to include. Defaults to config.
        capture_modes: List of capture modes to include. Defaults to all.
        max_samples: Maximum number of samples to return.

    Returns:
        List of ChatML-format instruction pairs.
    """
    if doc_types is None:
        doc_types = config.SELECTED_DOC_TYPES
    if capture_modes is None:
        capture_modes = config.CAPTURE_MODES

    dataset = []

    for doc_type in doc_types:
        doc_dir = os.path.join(data_dir, doc_type)
        if not os.path.isdir(doc_dir):
            print(f"Warning: Directory not found for {doc_type}: {doc_dir}")
            continue

        # Load ground truth
        gt_dir = os.path.join(doc_dir, "ground_truth")
        if os.path.isdir(gt_dir):
            ground_truth = parse_midv_ground_truth(gt_dir)
        else:
            # Try parsing annotation JSONs directly
            json_files = glob.glob(os.path.join(doc_dir, "**", "*.json"), recursive=True)
            ground_truth = {}
            for jf in json_files:
                parsed = parse_midv_annotation(jf)
                for fname, data in parsed.items():
                    doc_id = Path(fname).stem.split("_")[0]
                    ground_truth[doc_id] = data["fields"]

        if not ground_truth:
            print(f"Warning: No ground truth found for {doc_type}")
            continue

        # Find images
        images_dir = os.path.join(doc_dir, "images")
        if not os.path.isdir(images_dir):
            # Try flat structure
            images_dir = doc_dir

        for mode in capture_modes:
            mode_dir = os.path.join(images_dir, mode)
            if not os.path.isdir(mode_dir):
                # Try finding images directly
                mode_dir = images_dir

            image_files = (
                glob.glob(os.path.join(mode_dir, "*.jpg"))
                + glob.glob(os.path.join(mode_dir, "*.png"))
                + glob.glob(os.path.join(mode_dir, "*.jpeg"))
                + glob.glob(os.path.join(mode_dir, "*.tif"))
            )

            for img_path in image_files:
                # Extract document ID from filename
                doc_id = _extract_doc_id(img_path)
                if doc_id in ground_truth:
                    pair = build_instruction_pair(
                        image_path=img_path,
                        fields=ground_truth[doc_id],
                        doc_type=doc_type,
                    )
                    dataset.append(pair)

    # Shuffle and limit
    random.seed(config.RANDOM_SEED)
    random.shuffle(dataset)

    if max_samples and len(dataset) > max_samples:
        dataset = dataset[:max_samples]

    print(f"Created dataset with {len(dataset)} instruction pairs")
    return dataset


def split_dataset(
    dataset: List[Dict[str, Any]],
    train_ratio: Optional[float] = None,
    val_ratio: Optional[float] = None,
    test_ratio: Optional[float] = None,
) -> Tuple[List, List, List]:
    """
    Split dataset into train/val/test sets with stratification by doc_type.

    Args:
        dataset: Full list of instruction pairs.
        train_ratio: Fraction for training. Defaults to config.TRAIN_RATIO.
        val_ratio: Fraction for validation. Defaults to config.VAL_RATIO.
        test_ratio: Fraction for testing. Defaults to config.TEST_RATIO.

    Returns:
        Tuple of (train_set, val_set, test_set).
    """
    if train_ratio is None:
        train_ratio = config.TRAIN_RATIO
    if val_ratio is None:
        val_ratio = config.VAL_RATIO
    if test_ratio is None:
        test_ratio = config.TEST_RATIO

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    # Group by doc_type for stratified splitting
    by_type = {}
    for item in dataset:
        doc_type = item.get("metadata", {}).get("doc_type", "unknown")
        if doc_type not in by_type:
            by_type[doc_type] = []
        by_type[doc_type].append(item)

    train, val, test = [], [], []

    random.seed(config.RANDOM_SEED)
    for doc_type, items in by_type.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    # Final shuffle
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def save_dataset(dataset: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save instruction pairs to a JSONL file.

    Note: PIL images are converted back to paths for serialization.
    The metadata field is preserved for evaluation.

    Args:
        dataset: List of instruction pairs.
        output_path: Path to save the JSONL file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            # Convert PIL images to paths for serialization
            serializable = _make_serializable(item)
            f.write(json.dumps(serializable, ensure_ascii=False) + "\n")

    print(f"Saved {len(dataset)} samples to {output_path}")


def load_dataset(input_path: str) -> List[Dict[str, Any]]:
    """
    Load instruction pairs from a JSONL file.

    Args:
        input_path: Path to the JSONL file.

    Returns:
        List of instruction pair dicts.
    """
    dataset = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))
    print(f"Loaded {len(dataset)} samples from {input_path}")
    return dataset


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def get_capture_mode(image_path: str) -> str:
    """
    Determine the capture mode from the image path.

    MIDV-2020 organizes images into scan/photo/video subdirectories.
    Falls back to heuristics based on filename patterns.

    Args:
        image_path: Path to the image file.

    Returns:
        One of "scan", "photo", or "video_frame".
    """
    path_lower = image_path.lower()

    if "scan" in path_lower:
        return "scan"
    elif "video" in path_lower or "clip" in path_lower or "frame" in path_lower:
        return "video_frame"
    elif "photo" in path_lower:
        return "photo"

    # Heuristic: scanned images tend to have higher resolution
    # and specific naming patterns
    filename = os.path.basename(path_lower)
    if filename.startswith("scan") or "_scan" in filename:
        return "scan"
    elif any(x in filename for x in ["frame", "vid", "clip"]):
        return "video_frame"

    return "photo"  # default


def validate_image(image_path: str) -> bool:
    """
    Validate that an image file exists, is readable, and meets size constraints.

    Args:
        image_path: Path to the image file.

    Returns:
        True if the image is valid.
    """
    if not os.path.isfile(image_path):
        return False

    try:
        with Image.open(image_path) as img:
            w, h = img.size
            min_dim = min(w, h)
            max_dim = max(w, h)

            if min_dim < 50:  # Too small to be useful
                return False
            if max_dim > 10000:  # Suspiciously large
                return False

            return True
    except Exception:
        return False


def resize_for_training(image_path: str, output_path: Optional[str] = None) -> str:
    """
    Resize an image to fit within Unsloth's recommended range (300-1000px).

    Args:
        image_path: Path to the input image.
        output_path: Path for the resized image. If None, overwrites original.

    Returns:
        Path to the resized image.
    """
    if output_path is None:
        output_path = image_path

    with Image.open(image_path) as img:
        w, h = img.size
        max_dim = max(w, h)

        if max_dim > config.IMAGE_MAX_SIZE:
            scale = config.IMAGE_MAX_SIZE / max_dim
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        elif max_dim < config.IMAGE_MIN_SIZE:
            scale = config.IMAGE_MIN_SIZE / max_dim
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        img.save(output_path, quality=95)

    return output_path


def _normalize_field_name(name: str) -> str:
    """Normalize a field name to snake_case."""
    # Replace common separators with underscores
    result = name.strip().lower()
    for char in [" ", "-", ".", ":"]:
        result = result.replace(char, "_")
    # Remove consecutive underscores
    while "__" in result:
        result = result.replace("__", "_")
    return result.strip("_")


def _infer_doc_type(path: str) -> str:
    """Infer document type from file path components."""
    path_lower = path.lower()
    for doc_type in config.SELECTED_DOC_TYPES:
        if doc_type in path_lower:
            return doc_type
    return "unknown"


def _extract_doc_id(image_path: str) -> str:
    """Extract document ID from image filename."""
    stem = Path(image_path).stem
    # Common patterns: "00001_frame_001", "00001_scan", "doc_00001"
    # Extract leading numeric part
    parts = stem.split("_")
    for part in parts:
        if part.isdigit():
            return part
    # Fallback: use first part
    return parts[0] if parts else stem


def _make_serializable(item: Dict[str, Any]) -> Dict[str, Any]:
    """Convert non-serializable objects (PIL images) to serializable form."""
    result = {}
    for key, value in item.items():
        if key == "messages":
            messages = []
            for msg in value:
                new_msg = {"role": msg["role"]}
                content = []
                for c in msg["content"]:
                    if isinstance(c.get("image"), Image.Image):
                        # Convert PIL to path if stored in metadata
                        content.append({"type": "image", "image": "[PIL_IMAGE]"})
                    else:
                        content.append(c)
                new_msg["content"] = content
                messages.append(new_msg)
            result["messages"] = messages
        else:
            result[key] = value
    return result

"""
ID-VLM Tamper Detection Module (Stretch Goal)
Generates synthetic tampered identity documents using OpenCV operations:
  - Splice: copy-paste a field region from one document to another
  - Blur: apply localized Gaussian blur to a field (simulates re-capture)
  - Font mismatch: re-render text with a different font
  - Color shift: alter hue/saturation in the tampered region

Leverages the same OpenCV skills from AuthNet/PhotoRetouch projects.
"""

import os
import json
import random
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
    print("Warning: OpenCV not installed. Tamper generation requires: pip install opencv-python")

from PIL import Image

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ──────────────────────────────────────────────
# Tamper Operations
# ──────────────────────────────────────────────

def splice_field(
    src_image: np.ndarray,
    dst_image: np.ndarray,
    src_bbox: List[int],
    dst_bbox: List[int],
    blend_margin: int = 3,
) -> np.ndarray:
    """
    Copy a field region from source image and paste onto destination.

    Applies feathered blending at the edges for realism.

    Args:
        src_image: Source document image (BGR numpy array).
        dst_image: Destination document image (BGR numpy array).
        src_bbox: [x, y, w, h] bounding box in source.
        dst_bbox: [x, y, w, h] bounding box in destination.
        blend_margin: Pixels of feathered blending at edges.

    Returns:
        Tampered image (BGR numpy array).
    """
    if cv2 is None:
        raise RuntimeError("OpenCV required for tamper generation")

    result = dst_image.copy()

    sx, sy, sw, sh = src_bbox
    dx, dy, dw, dh = dst_bbox

    # Extract source region
    src_region = src_image[sy:sy + sh, sx:sx + sw]

    # Resize source region to match destination bbox
    if sw != dw or sh != dh:
        src_region = cv2.resize(src_region, (dw, dh), interpolation=cv2.INTER_LANCZOS4)

    # Create a feathered mask for smooth blending
    mask = np.ones((dh, dw), dtype=np.float32)
    if blend_margin > 0:
        for m in range(blend_margin):
            alpha = (m + 1) / (blend_margin + 1)
            mask[m, :] *= alpha
            mask[-(m + 1), :] *= alpha
            mask[:, m] *= alpha
            mask[:, -(m + 1)] *= alpha

    # Blend
    mask_3ch = np.stack([mask] * 3, axis=-1)
    dst_region = result[dy:dy + dh, dx:dx + dw].astype(np.float32)
    src_float = src_region.astype(np.float32)

    blended = src_float * mask_3ch + dst_region * (1 - mask_3ch)
    result[dy:dy + dh, dx:dx + dw] = blended.astype(np.uint8)

    return result


def blur_field(
    image: np.ndarray,
    bbox: List[int],
    kernel_size: Optional[int] = None,
) -> np.ndarray:
    """
    Apply localized Gaussian blur to a field region.

    Simulates the artifact left when someone edits a field and
    re-photographs/re-scans the document.

    Args:
        image: Document image (BGR numpy array).
        bbox: [x, y, w, h] bounding box of the field.
        kernel_size: Gaussian kernel size. If None, randomly chosen from config range.

    Returns:
        Image with blurred field region.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV required for tamper generation")

    result = image.copy()
    x, y, w, h = bbox

    if kernel_size is None:
        k_min, k_max = config.BLUR_KERNEL_RANGE
        kernel_size = random.choice(range(k_min, k_max + 1, 2))  # Must be odd
        if kernel_size % 2 == 0:
            kernel_size += 1

    region = result[y:y + h, x:x + w]
    blurred = cv2.GaussianBlur(region, (kernel_size, kernel_size), 0)
    result[y:y + h, x:x + w] = blurred

    return result


def color_shift_field(
    image: np.ndarray,
    bbox: List[int],
    hue_shift: Optional[int] = None,
) -> np.ndarray:
    """
    Shift the hue/saturation of a field region.

    Creates a subtle color inconsistency that suggests digital manipulation.

    Args:
        image: Document image (BGR numpy array).
        bbox: [x, y, w, h] bounding box of the field.
        hue_shift: Hue shift in degrees. If None, randomly chosen.

    Returns:
        Image with color-shifted field region.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV required for tamper generation")

    result = image.copy()
    x, y, w, h = bbox

    if hue_shift is None:
        shift_min, shift_max = config.COLOR_SHIFT_RANGE
        hue_shift = random.randint(shift_min, shift_max)
        if random.random() > 0.5:
            hue_shift = -hue_shift

    region = result[y:y + h, x:x + w]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv = hsv.astype(np.uint8)
    result[y:y + h, x:x + w] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return result


def font_mismatch_field(
    image: np.ndarray,
    bbox: List[int],
    text: str,
    font_scale: float = 0.6,
    color: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    Re-render a text field with a mismatched font.

    Replaces the field region with white background + re-drawn text
    using OpenCV's default font, creating a visible font mismatch.

    Args:
        image: Document image (BGR numpy array).
        bbox: [x, y, w, h] bounding box of the field.
        text: Text to render in the field.
        font_scale: OpenCV font scale.
        color: Text color (BGR). If None, estimates from original region.

    Returns:
        Image with font-mismatched field.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV required for tamper generation")

    result = image.copy()
    x, y, w, h = bbox

    # Estimate background and text colors from original region
    region = result[y:y + h, x:x + w]
    if color is None:
        # Use darkest color in region as text color
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        dark_mask = gray < np.percentile(gray, 30)
        if dark_mask.any():
            color = tuple(int(c) for c in region[dark_mask].mean(axis=0))
        else:
            color = (0, 0, 0)

    # Estimate background color
    bg_gray = gray > np.percentile(gray, 70)
    if bg_gray.any():
        bg_color = tuple(int(c) for c in region[bg_gray].mean(axis=0))
    else:
        bg_color = (255, 255, 255)

    # Fill region with background color
    result[y:y + h, x:x + w] = bg_color

    # Render text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, 1)[0]

    # Center the text in the bounding box
    text_x = x + max(0, (w - text_size[0]) // 2)
    text_y = y + h - max(0, (h - text_size[1]) // 2)

    cv2.putText(result, text, (text_x, text_y), font, font_scale, color, 1)

    return result


# ──────────────────────────────────────────────
# Dataset Generation
# ──────────────────────────────────────────────

def generate_tampered_dataset(
    clean_dataset: List[Dict[str, Any]],
    output_dir: str,
    tamper_ratio: Optional[float] = None,
    tamper_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate a balanced dataset of authentic and tampered documents.

    Args:
        clean_dataset: List of clean instruction pairs (from dataset.py).
        output_dir: Directory to save tampered images.
        tamper_ratio: Fraction of documents to tamper. Defaults to config.
        tamper_types: List of tamper operations to apply. Defaults to config.

    Returns:
        Extended dataset with both authentic and tampered samples,
        each annotated with tamper metadata.
    """
    if tamper_ratio is None:
        tamper_ratio = config.TAMPER_RATIO
    if tamper_types is None:
        tamper_types = config.TAMPER_TYPES

    os.makedirs(output_dir, exist_ok=True)
    random.seed(config.RANDOM_SEED)

    tampered_dataset = []
    n_to_tamper = int(len(clean_dataset) * tamper_ratio)

    # Select samples to tamper
    tamper_indices = set(random.sample(range(len(clean_dataset)), min(n_to_tamper, len(clean_dataset))))

    for i, sample in enumerate(clean_dataset):
        metadata = sample.get("metadata", {})
        image_path = metadata.get("image_path", "")
        doc_type = metadata.get("doc_type", "unknown")

        # Get the ground truth fields
        gt_text = sample["messages"][1]["content"][0]["text"]
        try:
            fields = json.loads(gt_text)
        except (json.JSONDecodeError, KeyError):
            fields = {}

        if i in tamper_indices and image_path and os.path.isfile(image_path) and fields:
            # Generate tampered version
            tamper_type = random.choice(tamper_types)
            tampered_field = random.choice(list(fields.keys()))

            try:
                tampered_path = _apply_tamper(
                    image_path=image_path,
                    field_name=tampered_field,
                    tamper_type=tamper_type,
                    output_dir=output_dir,
                    index=i,
                )

                if tampered_path:
                    from src.dataset import build_tamper_instruction_pair
                    tamper_pair = build_tamper_instruction_pair(
                        image_path=tampered_path,
                        is_tampered=True,
                        tampered_field=tampered_field,
                        tamper_type=tamper_type,
                        doc_type=doc_type,
                    )
                    tampered_dataset.append(tamper_pair)

            except Exception as e:
                print(f"Warning: Tamper generation failed for sample {i}: {e}")

        # Add authentic version
        from src.dataset import build_tamper_instruction_pair
        authentic_pair = build_tamper_instruction_pair(
            image_path=image_path,
            is_tampered=False,
            doc_type=doc_type,
        )
        tampered_dataset.append(authentic_pair)

    random.shuffle(tampered_dataset)
    print(f"Generated tamper dataset: {len(tampered_dataset)} samples "
          f"({n_to_tamper} tampered + {len(clean_dataset)} authentic)")
    return tampered_dataset


def _apply_tamper(
    image_path: str,
    field_name: str,
    tamper_type: str,
    output_dir: str,
    index: int,
    bbox: Optional[List[int]] = None,
) -> Optional[str]:
    """
    Apply a tamper operation to an image and save the result.

    If no bounding box is provided, generates a synthetic one
    in the typical location for the given field type.

    Returns:
        Path to the tampered image, or None on failure.
    """
    if cv2 is None:
        return None

    image = cv2.imread(image_path)
    if image is None:
        return None

    h, w = image.shape[:2]

    # Generate synthetic bbox if not provided
    if bbox is None:
        bbox = _generate_field_bbox(field_name, w, h)

    # Clamp bbox to image bounds
    bx, by, bw, bh = bbox
    bx = max(0, min(bx, w - 10))
    by = max(0, min(by, h - 10))
    bw = min(bw, w - bx)
    bh = min(bh, h - by)

    if bw < 5 or bh < 5:
        return None

    clamped_bbox = [bx, by, bw, bh]

    # Apply tamper
    if tamper_type == "blur":
        tampered = blur_field(image, clamped_bbox)
    elif tamper_type == "color_shift":
        tampered = color_shift_field(image, clamped_bbox)
    elif tamper_type == "splice":
        # Self-splice with offset (simulates copy from another doc)
        offset_bbox = [bx + 20, by, bw, bh]
        tampered = splice_field(image, image, offset_bbox, clamped_bbox)
    elif tamper_type == "font_mismatch":
        tampered = font_mismatch_field(image, clamped_bbox, "ALTERED")
    else:
        return None

    # Save
    output_path = os.path.join(
        output_dir, f"tampered_{index}_{tamper_type}_{field_name}.jpg"
    )
    cv2.imwrite(output_path, tampered)
    return output_path


def _generate_field_bbox(
    field_name: str,
    image_width: int,
    image_height: int,
) -> List[int]:
    """
    Generate a plausible bounding box for a field based on typical ID layouts.

    This is a fallback when actual MIDV-2020 annotations don't include
    per-field bounding boxes for a specific image.
    """
    # Typical field regions (as fractions of image size)
    field_regions = {
        "surname":         (0.30, 0.20, 0.40, 0.06),
        "given_name":      (0.30, 0.28, 0.40, 0.06),
        "birth_date":      (0.30, 0.36, 0.30, 0.06),
        "document_number": (0.30, 0.12, 0.40, 0.06),
        "expiry_date":     (0.30, 0.44, 0.30, 0.06),
        "nationality":     (0.30, 0.52, 0.30, 0.06),
        "sex":             (0.30, 0.60, 0.15, 0.06),
        "birth_place":     (0.30, 0.68, 0.40, 0.06),
    }

    fracs = field_regions.get(field_name, (0.25, 0.40, 0.35, 0.06))
    fx, fy, fw, fh = fracs

    return [
        int(fx * image_width),
        int(fy * image_height),
        int(fw * image_width),
        int(fh * image_height),
    ]

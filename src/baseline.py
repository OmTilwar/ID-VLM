"""
ID-VLM Baseline Module
Runs the un-fine-tuned (zero-shot) Qwen2-VL on identity document images
to establish the baseline performance for comparison.

This module is designed to run on Colab/Kaggle with GPU access.
Locally, it can be imported for result loading and analysis.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def run_baseline_extraction(
    test_dataset: List[Dict[str, Any]],
    model=None,
    tokenizer=None,
    output_dir: Optional[str] = None,
    batch_size: int = 1,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run zero-shot (un-fine-tuned) Qwen2-VL on the test dataset.

    This function is intended to run in a Colab/Kaggle notebook where
    the model is already loaded. It processes images one at a time
    (VLMs typically need batch_size=1 on T4).

    Args:
        test_dataset: List of instruction pairs from dataset.py.
        model: Loaded Qwen2-VL model (from FastVisionModel or transformers).
        tokenizer: Model tokenizer.
        output_dir: Directory to save predictions. Defaults to config.BASELINE_DIR.
        batch_size: Processing batch size (default 1 for T4 GPU).
        max_samples: Maximum samples to process (for debugging).

    Returns:
        List of prediction dicts: {"raw_output": str, "metadata": dict, "time_s": float}
    """
    if output_dir is None:
        output_dir = config.BASELINE_DIR

    if max_samples:
        test_dataset = test_dataset[:max_samples]

    predictions = []

    for i, sample in enumerate(test_dataset):
        metadata = sample.get("metadata", {})
        user_message = sample["messages"][0]

        # Extract image and prompt from the user message
        image_content = None
        text_prompt = ""
        for content_item in user_message["content"]:
            if content_item.get("type") == "image":
                image_content = content_item.get("image")
            elif content_item.get("type") == "text":
                text_prompt = content_item.get("text", "")

        if model is None or tokenizer is None:
            # Dry run / local testing — generate placeholder
            raw_output = '{"note": "model not loaded — dry run"}'
            elapsed = 0.0
        else:
            # Real inference
            raw_output, elapsed = _run_inference(
                model, tokenizer, image_content, text_prompt
            )

        predictions.append({
            "raw_output": raw_output,
            "metadata": metadata,
            "time_s": elapsed,
            "sample_index": i,
        })

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(test_dataset)} samples...")

    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "predictions.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(predictions)} baseline predictions to {output_path}")

    # Save timing stats
    times = [p["time_s"] for p in predictions]
    if times and any(t > 0 for t in times):
        timing = {
            "mean_s": sum(times) / len(times),
            "median_s": sorted(times)[len(times) // 2],
            "total_s": sum(times),
            "n_samples": len(times),
        }
        with open(os.path.join(output_dir, "timing.json"), "w") as f:
            json.dump(timing, f, indent=2)

    return predictions


def load_predictions(predictions_path: str) -> List[Dict[str, Any]]:
    """Load saved predictions from a JSON file."""
    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    print(f"Loaded {len(predictions)} predictions from {predictions_path}")
    return predictions


def _run_inference(
    model,
    tokenizer,
    image,
    prompt: str,
) -> tuple:
    """
    Run a single inference pass on the model.

    Supports both Unsloth FastVisionModel and HuggingFace transformers.

    Args:
        model: Loaded model.
        tokenizer: Model tokenizer/processor.
        image: PIL Image or image path.
        prompt: Text prompt for the model.

    Returns:
        Tuple of (output_text: str, elapsed_seconds: float).
    """
    try:
        from PIL import Image as PILImage

        # Ensure image is PIL
        if isinstance(image, str):
            image = PILImage.open(image).convert("RGB")

        # Build the chat message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        start = time.time()

        # Try Unsloth's inference path first
        try:
            from unsloth import FastVisionModel
            FastVisionModel.for_inference(model)

            # Use the tokenizer's chat template
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process with the model
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = tokenizer(
                text=[input_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            import torch
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=False,
                )

            # Decode only the generated tokens
            generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
            output_text = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

        except ImportError:
            # Fallback to HuggingFace transformers
            from transformers import AutoProcessor
            import torch

            processor = tokenizer  # In HF, processor handles both text and images

            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=512)

            output_text = processor.batch_decode(
                output_ids, skip_special_tokens=True
            )[0]

        elapsed = time.time() - start
        return output_text.strip(), elapsed

    except Exception as e:
        print(f"Inference error: {e}")
        return f'{{"error": "{str(e)}"}}', 0.0

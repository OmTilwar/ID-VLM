"""
ID-VLM Configuration
Central configuration for model settings, data paths, training hyperparameters,
and evaluation settings for identity document VLM fine-tuning.
"""

import os

# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
MODEL_NAME = "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit"
MODEL_BASE = "Qwen/Qwen2-VL-2B-Instruct"  # For tokenizer / zero-shot baseline
LOAD_IN_4BIT = True
USE_GRADIENT_CHECKPOINTING = "unsloth"

# LoRA
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
LORA_BIAS = "none"
FINETUNE_VISION_LAYERS = True
FINETUNE_LANGUAGE_LAYERS = True
FINETUNE_ATTENTION_MODULES = True
FINETUNE_MLP_MODULES = True

# ──────────────────────────────────────────────
# Data — MIDV-2020 Subset
# ──────────────────────────────────────────────
SELECTED_DOC_TYPES = [
    "alb_id",           # ID Card of Albania
    "aze_passport",     # Passport of Azerbaijan
    "esp_id",           # ID Card of Spain
    "grc_passport",     # Passport of Greece
]

# Fields expected per document type (subset — extend as you explore the data)
DOC_TYPE_FIELDS = {
    "alb_id": [
        "surname", "given_name", "birth_date", "document_number",
        "expiry_date", "nationality", "sex",
    ],
    "aze_passport": [
        "surname", "given_name", "birth_date", "document_number",
        "expiry_date", "nationality", "sex", "birth_place",
    ],
    "esp_id": [
        "surname", "given_name", "birth_date", "document_number",
        "expiry_date", "nationality", "sex",
    ],
    "grc_passport": [
        "surname", "given_name", "birth_date", "document_number",
        "expiry_date", "nationality", "sex", "birth_place",
    ],
}

# Capture modes in MIDV-2020
CAPTURE_MODES = ["scan", "photo", "video_frame"]

# Image constraints (Unsloth recommendation for VLM fine-tuning)
IMAGE_MIN_SIZE = 300   # pixels (shorter side)
IMAGE_MAX_SIZE = 1000  # pixels (longer side)

# Dataset splits
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ──────────────────────────────────────────────
# Training Hyperparameters
# ──────────────────────────────────────────────
BATCH_SIZE = 1                    # Per-device (VLMs need batch=1 on T4)
GRADIENT_ACCUMULATION_STEPS = 4   # Effective batch = 4
LEARNING_RATE = 2e-4
WARMUP_STEPS = 20
MAX_STEPS = 300                   # ~300 steps for a few hundred images
LOGGING_STEPS = 1
SAVE_STEPS = 50

# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
CER_THRESHOLD = 0.1    # CER below this = "acceptable"
EXACT_MATCH_TARGET = 0.7  # Target field-level exact match after fine-tuning

# ──────────────────────────────────────────────
# Tamper Detection (Stretch)
# ──────────────────────────────────────────────
TAMPER_RATIO = 0.5           # 50% of documents get tampered versions
TAMPER_TYPES = ["splice", "blur", "font_mismatch", "color_shift"]
BLUR_KERNEL_RANGE = (5, 15)  # Gaussian blur kernel size range
COLOR_SHIFT_RANGE = (10, 40)  # Hue shift range in degrees

# ──────────────────────────────────────────────
# Prompt Templates
# ──────────────────────────────────────────────
EXTRACTION_PROMPT = (
    "Extract all text fields from this identity document. "
    "Return a JSON object with field names as keys and their values as strings. "
    "Use snake_case for field names (e.g., birth_date, document_number)."
)

TAMPER_DETECTION_PROMPT = (
    "Examine this identity document for signs of tampering or forgery. "
    "Return a JSON object with keys: 'verdict' (authentic/tampered), "
    "'confidence' (0.0-1.0), 'suspicious_field' (field name or null), "
    "and 'reason' (brief explanation)."
)

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
BASELINE_DIR = os.path.join(OUTPUT_DIR, "baseline")
FINETUNED_DIR = os.path.join(OUTPUT_DIR, "finetuned")
ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "analysis")
CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")

# ──────────────────────────────────────────────
# Create directories
# ──────────────────────────────────────────────
for _dir in [
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    OUTPUT_DIR, BASELINE_DIR, FINETUNED_DIR, ANALYSIS_DIR, CHECKPOINTS_DIR,
    NOTEBOOKS_DIR,
]:
    os.makedirs(_dir, exist_ok=True)

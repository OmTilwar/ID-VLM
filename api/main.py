"""
ID-VLM FastAPI Server
Serves model predictions via REST API with 3 endpoints:
  POST /extract — Upload ID image → JSON with extracted fields
  POST /verify  — Upload ID image → Authentic/tampered verdict
  GET  /health  — Health check

Follows the same API pattern as AuthNet for consistency.
"""

import os
import sys
import io
import json
from typing import Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ── Global State ──
model = None
tokenizer = None
_model_loaded = False


def _load_model():
    """
    Load the fine-tuned ID-VLM model.

    Attempts to load from the checkpoint directory. If no fine-tuned
    model is found, falls back to the base model for demo purposes.
    """
    global model, tokenizer, _model_loaded

    merged_path = os.path.join(config.CHECKPOINTS_DIR, "merged_model")
    lora_path = os.path.join(config.CHECKPOINTS_DIR, "lora_adapter")

    try:
        if os.path.isdir(merged_path):
            # Load merged model
            print(f"Loading merged model from {merged_path}...")
            from unsloth import FastVisionModel
            model, tokenizer = FastVisionModel.from_pretrained(
                merged_path,
                load_in_4bit=config.LOAD_IN_4BIT,
            )
            FastVisionModel.for_inference(model)
            _model_loaded = True
            print("Fine-tuned model loaded!")

        elif os.path.isdir(lora_path):
            # Load base + LoRA adapter
            print(f"Loading base model + LoRA adapter from {lora_path}...")
            from unsloth import FastVisionModel
            model, tokenizer = FastVisionModel.from_pretrained(
                config.MODEL_NAME,
                load_in_4bit=config.LOAD_IN_4BIT,
            )
            model.load_adapter(lora_path)
            FastVisionModel.for_inference(model)
            _model_loaded = True
            print("Model with LoRA adapter loaded!")

        else:
            print("No fine-tuned model found. API will run in demo mode.")
            print(f"  Checked: {merged_path}")
            print(f"  Checked: {lora_path}")
            _model_loaded = False

    except ImportError:
        print("Unsloth not installed. API running in demo mode.")
        print("Install with: pip install unsloth")
        _model_loaded = False
    except Exception as e:
        print(f"Model loading failed: {e}")
        _model_loaded = False


@asynccontextmanager
async def lifespan(app):
    """Load model on startup, clean up on shutdown."""
    _load_model()
    yield


app = FastAPI(
    title="ID-VLM API",
    description=(
        "Identity Document Understanding API. "
        "Uses a fine-tuned Qwen2-VL model for structured field extraction "
        "from identity documents and tamper detection."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Response Schemas ──

class HealthResponse(BaseModel):
    status: str
    model: str
    model_loaded: bool
    device: str


class ExtractionResponse(BaseModel):
    fields: Dict[str, str]
    confidence: float
    doc_type_hint: Optional[str] = None
    raw_output: str


class VerifyResponse(BaseModel):
    verdict: str
    confidence: float
    suspicious_field: Optional[str] = None
    reason: str


# ── Helper ──

async def read_image(file: UploadFile) -> Image.Image:
    """Read an uploaded file into a PIL Image."""
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")


def _run_model_inference(image: Image.Image, prompt: str) -> str:
    """Run inference on the loaded model."""
    if not _model_loaded or model is None:
        # Demo mode — return a placeholder response
        return json.dumps({
            "note": "Demo mode — model not loaded",
            "surname": "DEMO",
            "given_name": "USER",
            "birth_date": "01-01-2000",
            "document_number": "DEMO12345",
        })

    from src.baseline import _run_inference
    output_text, _ = _run_inference(model, tokenizer, image, prompt)
    return output_text


# ── Endpoints ──

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    import torch
    return HealthResponse(
        status="healthy",
        model=config.MODEL_BASE,
        model_loaded=_model_loaded,
        device=str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
    )


@app.post("/extract", response_model=ExtractionResponse)
async def extract_fields(image: UploadFile = File(...)):
    """
    Extract structured fields from an identity document image.

    Upload a photo, scan, or video frame of an ID card or passport.
    Returns a JSON object with extracted field names and values.
    """
    pil_image = await read_image(image)

    raw_output = _run_model_inference(pil_image, config.EXTRACTION_PROMPT)

    # Parse the model output
    from src.evaluate import parse_vlm_json_output
    parsed, is_valid = parse_vlm_json_output(raw_output)

    if not is_valid or parsed is None:
        return ExtractionResponse(
            fields={},
            confidence=0.0,
            raw_output=raw_output,
        )

    # Remove non-field keys if present
    fields = {k: str(v) for k, v in parsed.items()
              if k not in ("note", "error", "confidence")}

    return ExtractionResponse(
        fields=fields,
        confidence=0.9 if _model_loaded else 0.0,
        raw_output=raw_output,
    )


@app.post("/verify", response_model=VerifyResponse)
async def verify_document(image: UploadFile = File(...)):
    """
    Verify the authenticity of an identity document.

    Upload a document image to check for signs of tampering or forgery.
    Returns a verdict (authentic/tampered), confidence score,
    and the suspicious field if tampering is detected.
    """
    pil_image = await read_image(image)

    raw_output = _run_model_inference(pil_image, config.TAMPER_DETECTION_PROMPT)

    from src.evaluate import parse_vlm_json_output
    parsed, is_valid = parse_vlm_json_output(raw_output)

    if not is_valid or parsed is None:
        return VerifyResponse(
            verdict="unknown",
            confidence=0.0,
            reason=f"Could not parse model output: {raw_output[:200]}",
        )

    return VerifyResponse(
        verdict=parsed.get("verdict", "unknown"),
        confidence=float(parsed.get("confidence", 0.0)),
        suspicious_field=parsed.get("suspicious_field"),
        reason=parsed.get("reason", "No explanation provided"),
    )


# ── Main ──
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# ID-VLM: VLM Fine-Tuning for Identity Document Understanding

Fine-tuned **Qwen2-VL-2B** via LoRA ([Unsloth](https://github.com/unslothai/unsloth)) on [MIDV-2020](https://arxiv.org/abs/2107.00396) for structured field extraction from identity documents, with synthetic tamper detection.

> **Why this project?** Scoped explicitly around four of HyperVerge's stated research areas: Document Understanding, OCR Enhancement, Identity Verification, and Image Forgery Detection — using the same LoRA/Unsloth fine-tuning stack as [FinSight AI](https://github.com/OmTilwar/FinSight-AI), applied to vision.

## Key Results

| Metric | Zero-Shot Baseline | Fine-Tuned (LoRA) | Δ |
|---|---|---|---|
| Field Exact Match | 7.7% | 9.4% | +1.8% |
| Character Error Rate (CER) | 0.9365 | 0.9976 | +0.0611 |
| JSON Parse Rate | 67.3% | 86.0% | +18.7% |
| Document-Level Accuracy | 0.0% | 0.0% | +0.0% |

### Performance by Capture Condition

| Condition | Baseline Exact Match | Fine-Tuned Exact Match |
|---|---|---|
| Photo | 7.7% | 9.4% |

### Performance by Document Type

| Document Type | Baseline Exact Match | Fine-Tuned Exact Match | Δ |
|---|---|---|---|
| Albania ID (`alb_id`) | 19.4% | 18.8% | -0.6% |
| Azerbaijan Passport (`aze_passport`) | 2.1% | 2.8% | +0.7% |
| Spain ID (`esp_id`) | 5.5% | 15.2% | **+9.7%** |
| Estonia ID (`est_id`) | 6.7% | 12.7% | **+6.0%** |
| Finland ID (`fin_id`) | 7.3% | 2.4% | -4.9% |
| Greece Passport (`grc_passport`) | 3.5% | 7.0% | +3.5% |
| Latvia Passport (`lva_passport`) | 4.3% | 2.4% | -1.9% |
| Russia Internal Passport (`rus_internalpassport`) | 6.2% | 2.7% | -3.5% |
| Serbia Passport (`srb_passport`) | 2.6% | 7.8% | **+5.2%** |
| Slovakia ID (`svk_id`) | 19.3% | 22.7% | **+3.4%** |

## Architecture

```
Image of ID/Passport ──► Qwen2-VL-2B (LoRA fine-tuned) ──► Structured JSON
                                                             {
                                                               "surname": "SMITH",
                                                               "given_name": "JOHN",
                                                               "birth_date": "15-03-1990",
                                                               "document_number": "AB1234567",
                                                               "expiry_date": "20-03-2030"
                                                             }
```

**Two tasks, same model:**
1. **Structured field extraction** — image of an ID/passport → JSON with name, DOB, document number, expiry
2. **Tamper detection** (stretch) — classify authentic vs. tampered, name the suspicious field

## Stack

| Component | Technology |
|---|---|
| Base Model | [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) |
| Fine-Tuning | [Unsloth](https://github.com/unslothai/unsloth) `FastVisionModel` + LoRA |
| Dataset | [MIDV-2020](https://arxiv.org/abs/2107.00396) (synthetic identity documents) |
| RAG & Evaluation | [Ragas](https://github.com/explodinggradients/ragas) (Context Precision, Recall, Faithfulness, Relevancy) |
| Tamper Generation | OpenCV (splice, blur, font mismatch) |
| Evaluation | Custom harness + Ragas Multimodal Document evaluation |
| API | FastAPI (`/extract`, `/verify`, `/health`) |
| Testing | pytest (78+ automated tests) |

---

## 📐 Multimodal Document RAG & KYC Evaluation with Ragas

To evaluate document extraction and zero-hallucination compliance across international identity documents (Spain ID, Estonia ID, Greece Passport, Slovakia ID, Albania ID), we integrated the **Ragas** framework evaluating across four core dimensions:

1. **Context Precision:** Measures whether visual/OCR bounding box regions containing target KYC fields are ranked at position #1.
2. **Context Recall:** Measures whether all required KYC fields (Surname, Given Name, DOB, Document Number, Expiry, Nationality) are captured.
3. **Faithfulness (Grounded):** Critical zero-hallucination verification—ensures extracted alphanumeric document numbers and dates strictly originate from the ID image.
4. **Answer Relevancy:** Measures structured adherence and completeness against downstream KYC verification prompts.

### 📊 Ragas Evaluation Scorecard

```text
================================================================================
ID-VLM: RAGAS MULTIMODAL DOCUMENT EVALUATION SCORECARD
================================================================================
Ragas Metric             | Zero-Shot Baseline | Fine-Tuned ID-VLM  | Delta / Gain    
--------------------------------------------------------------------------------
Context Precision        |           0.4333   |           0.4333   | 0.00% (Parity)  
Context Recall           |           0.6500   |           0.6500   | 0.00% (Parity)  
Faithfulness (Grounded)  |           0.6000   |           0.9100   | +31.00% 🏆      
Answer Relevancy         |           0.7459   |           0.7772   | +3.13% 🏆       
Region Latency           |          50.382 ms |          17.777 ms | Real-time ⚡    
================================================================================
```

To run the automated Ragas Multimodal evaluation suite:
```bash
python src/ragas_doc_eval.py
```
*(Results are saved to `outputs/ragas_id_vlm_benchmark.json`)*

## Project Structure

```
id-vlm/
├── config.py                    # Central configuration
├── requirements.txt             # Dependencies
├── README.md
│
├── src/
│   ├── dataset.py               # MIDV-2020 → instruction pairs
│   ├── evaluate.py              # Evaluation harness (CER, exact match, breakdowns)
│   ├── baseline.py              # Zero-shot baseline runner
│   └── tamper.py                # Synthetic tamper generation (stretch)
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_baseline_evaluation.ipynb
│   ├── 03_finetune.ipynb
│   └── 04_evaluate_finetuned.ipynb
│
├── api/                         # FastAPI service (stretch)
│   └── main.py
│
├── tests/
│   ├── test_dataset.py
│   ├── test_evaluate.py
│   └── test_api.py
│
├── scripts/
│   ├── prepare_data.py          # CLI: raw MIDV-2020 → instruction pairs
│   └── run_evaluation.py        # CLI: evaluate predictions
│
└── outputs/                     # Generated (gitignored)
    ├── baseline/
    ├── finetuned/
    └── analysis/
```

## Quick Start

### Local Setup (data processing + evaluation)

```bash
cd projects/id-vlm
pip install -r requirements.txt

# Prepare dataset (after downloading MIDV-2020 subset to data/raw/)
python scripts/prepare_data.py

# Run tests
python -m pytest tests/ -v
```

### Colab/Kaggle (fine-tuning)

1. Upload `src/` and `config.py` to your Colab/Kaggle session
2. Open `notebooks/03_finetune.ipynb`
3. Follow the notebook instructions

### API (stretch goal)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# POST /extract — upload ID image → JSON fields
# POST /verify  — upload ID image → authentic/tampered verdict
# GET  /health  — health check
```

## Dataset: MIDV-2020

[MIDV-2020](https://arxiv.org/abs/2107.00396) consists of 1,000 mock identity documents across 10 document types, with unique synthetic faces, signatures, and text field values. We use a **4-type subset** (Albania ID, Azerbaijan Passport, Spain ID, Greece Passport) for tractable Colab training.

Key properties:
- **Synthetic identities** — no real PII, clean for public repos
- **Multiple capture modes** — scans, photos, and video frames (testing robustness)
- **Rich annotations** — field-level bounding boxes and ground-truth values in JSON

## Evaluation Methodology

The evaluation harness measures:
- **Field-level exact match**: per-field binary accuracy
- **Character Error Rate (CER)**: Levenshtein distance / ground truth length
- **JSON parse rate**: % of VLM outputs that are valid JSON
- **Breakdowns**: by capture condition, document type, and field type

This before/after comparison (zero-shot → fine-tuned) is the core story of the project.

## License

MIT

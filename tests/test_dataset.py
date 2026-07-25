"""
Tests for ID-VLM Dataset Module
Tests annotation parsing, instruction pair generation, capture mode detection,
dataset splitting, and edge cases.
"""

import os
import sys
import json
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.dataset import (
    parse_midv_annotation,
    parse_midv_ground_truth,
    build_instruction_pair,
    build_tamper_instruction_pair,
    get_capture_mode,
    split_dataset,
    save_dataset,
    load_dataset,
    validate_image,
    _normalize_field_name,
    _extract_doc_id,
)
import config


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def mock_via_annotation(tmp_path):
    """Create a mock VIA v2 annotation JSON file."""
    annotation = {
        "_via_img_metadata": {
            "doc_001.jpg12345": {
                "filename": "doc_001.jpg",
                "size": 12345,
                "regions": [
                    {
                        "shape_attributes": {
                            "name": "rect",
                            "x": 100, "y": 50, "width": 200, "height": 30
                        },
                        "region_attributes": {
                            "field_name": "surname",
                            "value": "SMITH"
                        }
                    },
                    {
                        "shape_attributes": {
                            "name": "rect",
                            "x": 100, "y": 90, "width": 200, "height": 30
                        },
                        "region_attributes": {
                            "field_name": "given_name",
                            "value": "JOHN"
                        }
                    },
                    {
                        "shape_attributes": {
                            "name": "rect",
                            "x": 100, "y": 130, "width": 150, "height": 30
                        },
                        "region_attributes": {
                            "field_name": "birth_date",
                            "value": "15-03-1990"
                        }
                    },
                ],
            }
        }
    }

    json_path = tmp_path / "alb_id" / "annotations.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(annotation, f)

    return str(json_path)


@pytest.fixture
def mock_ground_truth(tmp_path):
    """Create mock ground truth files."""
    gt_dir = tmp_path / "alb_id" / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    # JSON-based ground truth
    gt_data = {
        "00001": {
            "surname": "SMITH",
            "given_name": "JOHN",
            "birth_date": "15-03-1990",
            "document_number": "AB1234567",
        },
        "00002": {
            "surname": "DOE",
            "given_name": "JANE",
            "birth_date": "20-07-1985",
            "document_number": "CD7654321",
        },
    }

    with open(gt_dir / "fields.json", "w") as f:
        json.dump(gt_data, f)

    return str(gt_dir)


@pytest.fixture
def sample_fields():
    """Sample field values for testing."""
    return {
        "surname": "SMITH",
        "given_name": "JOHN",
        "birth_date": "15-03-1990",
        "document_number": "AB1234567",
        "expiry_date": "20-03-2030",
    }


@pytest.fixture
def mock_image(tmp_path):
    """Create a minimal test image."""
    from PIL import Image
    img = Image.new("RGB", (640, 480), color=(255, 255, 255))
    img_path = tmp_path / "scan" / "alb_id" / "00001_scan.jpg"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(img_path))
    return str(img_path)


# ──────────────────────────────────────────────
# Annotation Parsing Tests
# ──────────────────────────────────────────────

class TestAnnotationParsing:
    """Tests for MIDV-2020 annotation parsing."""

    def test_parse_via_annotation(self, mock_via_annotation):
        """Test parsing VIA v2 format annotation."""
        result = parse_midv_annotation(mock_via_annotation)

        assert len(result) == 1
        entry = list(result.values())[0]
        assert "fields" in entry
        assert entry["fields"]["surname"] == "SMITH"
        assert entry["fields"]["given_name"] == "JOHN"
        assert entry["fields"]["birth_date"] == "15-03-1990"

    def test_parse_via_bboxes(self, mock_via_annotation):
        """Test that bounding boxes are correctly extracted."""
        result = parse_midv_annotation(mock_via_annotation)
        entry = list(result.values())[0]

        assert "bboxes" in entry
        assert "surname" in entry["bboxes"]
        assert entry["bboxes"]["surname"] == [100, 50, 200, 30]

    def test_parse_ground_truth_json(self, mock_ground_truth):
        """Test parsing JSON-based ground truth."""
        result = parse_midv_ground_truth(mock_ground_truth)

        assert "00001" in result
        assert result["00001"]["surname"] == "SMITH"
        assert "00002" in result
        assert result["00002"]["given_name"] == "JANE"

    def test_parse_empty_annotation(self, tmp_path):
        """Test parsing an empty annotation file."""
        json_path = tmp_path / "empty.json"
        with open(json_path, "w") as f:
            json.dump({}, f)

        result = parse_midv_annotation(str(json_path))
        assert len(result) == 0

    def test_parse_missing_fields(self, tmp_path):
        """Test handling of annotations with missing field_name or value."""
        annotation = {
            "_via_img_metadata": {
                "img.jpg123": {
                    "filename": "img.jpg",
                    "regions": [
                        {
                            "shape_attributes": {"name": "rect", "x": 0, "y": 0, "width": 10, "height": 10},
                            "region_attributes": {"field_name": "", "value": "SMITH"}
                        },
                        {
                            "shape_attributes": {"name": "rect", "x": 0, "y": 0, "width": 10, "height": 10},
                            "region_attributes": {"field_name": "name", "value": ""}
                        },
                    ]
                }
            }
        }
        json_path = tmp_path / "partial.json"
        with open(json_path, "w") as f:
            json.dump(annotation, f)

        result = parse_midv_annotation(str(json_path))
        # Both entries should be skipped (empty field_name or value)
        if result:
            entry = list(result.values())[0]
            assert "" not in entry["fields"]


# ──────────────────────────────────────────────
# Instruction Pair Tests
# ──────────────────────────────────────────────

class TestInstructionPairGeneration:
    """Tests for ChatML instruction pair generation."""

    def test_build_extraction_pair(self, mock_image, sample_fields):
        """Test building a field extraction instruction pair."""
        pair = build_instruction_pair(
            image_path=mock_image,
            fields=sample_fields,
            doc_type="alb_id",
        )

        assert "messages" in pair
        assert len(pair["messages"]) == 2
        assert pair["messages"][0]["role"] == "user"
        assert pair["messages"][1]["role"] == "assistant"

    def test_user_message_has_image_and_text(self, mock_image, sample_fields):
        """Test that user message contains both image and text content."""
        pair = build_instruction_pair(mock_image, sample_fields, "alb_id")
        user_content = pair["messages"][0]["content"]

        types = [c["type"] for c in user_content]
        assert "image" in types
        assert "text" in types

    def test_assistant_message_is_valid_json(self, mock_image, sample_fields):
        """Test that assistant response is valid JSON."""
        pair = build_instruction_pair(mock_image, sample_fields, "alb_id")
        answer_text = pair["messages"][1]["content"][0]["text"]

        parsed = json.loads(answer_text)
        assert isinstance(parsed, dict)
        assert parsed["surname"] == "SMITH"
        assert parsed["birth_date"] == "15-03-1990"

    def test_metadata_included(self, mock_image, sample_fields):
        """Test that metadata is included in the pair."""
        pair = build_instruction_pair(mock_image, sample_fields, "alb_id")

        assert "metadata" in pair
        assert pair["metadata"]["doc_type"] == "alb_id"
        assert pair["metadata"]["num_fields"] == 5

    def test_custom_prompt(self, mock_image, sample_fields):
        """Test using a custom extraction prompt."""
        custom_prompt = "What fields are on this ID?"
        pair = build_instruction_pair(
            mock_image, sample_fields, "alb_id", prompt=custom_prompt
        )

        text_content = [
            c for c in pair["messages"][0]["content"] if c["type"] == "text"
        ][0]
        assert text_content["text"] == custom_prompt

    def test_tamper_instruction_pair_authentic(self, mock_image):
        """Test building a tamper detection pair for authentic document."""
        pair = build_tamper_instruction_pair(
            image_path=mock_image,
            is_tampered=False,
            doc_type="alb_id",
        )

        answer = json.loads(pair["messages"][1]["content"][0]["text"])
        assert answer["verdict"] == "authentic"
        assert answer["suspicious_field"] is None

    def test_tamper_instruction_pair_tampered(self, mock_image):
        """Test building a tamper detection pair for tampered document."""
        pair = build_tamper_instruction_pair(
            image_path=mock_image,
            is_tampered=True,
            tampered_field="birth_date",
            tamper_type="splice",
            doc_type="alb_id",
        )

        answer = json.loads(pair["messages"][1]["content"][0]["text"])
        assert answer["verdict"] == "tampered"
        assert answer["suspicious_field"] == "birth_date"


# ──────────────────────────────────────────────
# Capture Mode Tests
# ──────────────────────────────────────────────

class TestCaptureMode:
    """Tests for capture mode detection from file paths."""

    def test_scan_detection(self):
        assert get_capture_mode("/data/alb_id/scans/001.jpg") == "scan"
        assert get_capture_mode("/data/scan_001.jpg") == "scan"

    def test_photo_detection(self):
        assert get_capture_mode("/data/alb_id/photos/001.jpg") == "photo"

    def test_video_frame_detection(self):
        assert get_capture_mode("/data/alb_id/video/frame_001.jpg") == "video_frame"
        assert get_capture_mode("/data/clips/001.jpg") == "video_frame"

    def test_default_is_photo(self):
        assert get_capture_mode("/data/unknown/001.jpg") == "photo"


# ──────────────────────────────────────────────
# Dataset Split Tests
# ──────────────────────────────────────────────

class TestDatasetSplit:
    """Tests for dataset splitting."""

    def test_split_ratios(self):
        """Test that split ratios produce correct sizes."""
        dataset = [
            {"metadata": {"doc_type": "alb_id"}} for _ in range(100)
        ]
        train, val, test = split_dataset(dataset, 0.7, 0.15, 0.15)

        total = len(train) + len(val) + len(test)
        assert total == 100
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_split_no_overlap(self):
        """Test that splits don't overlap (using identity)."""
        dataset = [
            {"metadata": {"doc_type": "alb_id"}, "id": i} for i in range(50)
        ]
        train, val, test = split_dataset(dataset, 0.6, 0.2, 0.2)

        train_ids = {d["id"] for d in train}
        val_ids = {d["id"] for d in val}
        test_ids = {d["id"] for d in test}

        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0

    def test_split_invalid_ratios(self):
        """Test that invalid split ratios raise an error."""
        dataset = [{"metadata": {"doc_type": "alb_id"}} for _ in range(10)]
        with pytest.raises(AssertionError):
            split_dataset(dataset, 0.5, 0.3, 0.3)  # sums to 1.1

    def test_split_empty_dataset(self):
        """Test splitting an empty dataset."""
        train, val, test = split_dataset([], 0.7, 0.15, 0.15)
        assert len(train) == 0
        assert len(val) == 0
        assert len(test) == 0


# ──────────────────────────────────────────────
# Serialization Tests
# ──────────────────────────────────────────────

class TestSerialization:
    """Tests for dataset save/load."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Test that save → load preserves data."""
        dataset = [
            {
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": "Extract fields"},
                        {"type": "image", "image": "/path/to/img.jpg"},
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": '{"surname": "SMITH"}'},
                    ]},
                ],
                "metadata": {"doc_type": "alb_id", "image_path": "/path/to/img.jpg"},
            }
        ]

        output_path = str(tmp_path / "test.jsonl")
        save_dataset(dataset, output_path)

        loaded = load_dataset(output_path)
        assert len(loaded) == 1
        assert loaded[0]["metadata"]["doc_type"] == "alb_id"


# ──────────────────────────────────────────────
# Helper Tests
# ──────────────────────────────────────────────

class TestHelpers:
    """Tests for internal helper functions."""

    def test_normalize_field_name(self):
        assert _normalize_field_name("Birth Date") == "birth_date"
        assert _normalize_field_name("SURNAME") == "surname"
        assert _normalize_field_name("document-number") == "document_number"
        assert _normalize_field_name("  expiry.date  ") == "expiry_date"

    def test_extract_doc_id(self):
        assert _extract_doc_id("/data/00001_scan.jpg") == "00001"
        assert _extract_doc_id("/data/doc_00042_frame_001.jpg") == "00042"

    def test_validate_image(self, mock_image):
        """Test image validation with a real image."""
        assert validate_image(mock_image) is True
        assert validate_image("/nonexistent/path.jpg") is False

    def test_validate_image_too_small(self, tmp_path):
        """Test that tiny images are rejected."""
        from PIL import Image
        tiny = Image.new("RGB", (10, 10))
        path = str(tmp_path / "tiny.jpg")
        tiny.save(path)
        assert validate_image(path) is False

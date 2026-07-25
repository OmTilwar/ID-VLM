"""
Tests for ID-VLM FastAPI Service
Tests health, extract, and verify endpoints with mock model.
"""

import os
import sys
import io
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


@pytest.fixture
def test_image_bytes():
    """Create a minimal test image as bytes."""
    from PIL import Image
    img = Image.new("RGB", (640, 480), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()


# ──────────────────────────────────────────────
# Health Endpoint Tests
# ──────────────────────────────────────────────

class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, client):
        """Test that health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model" in data
        assert "model_loaded" in data

    def test_health_response_schema(self, client):
        """Test health response has all required fields."""
        response = client.get("/health")
        data = response.json()
        required_fields = {"status", "model", "model_loaded", "device"}
        assert required_fields.issubset(set(data.keys()))


# ──────────────────────────────────────────────
# Extract Endpoint Tests
# ──────────────────────────────────────────────

class TestExtractEndpoint:
    """Tests for /extract endpoint."""

    def test_extract_returns_200(self, client, test_image_bytes):
        """Test that extract endpoint accepts an image and returns 200."""
        response = client.post(
            "/extract",
            files={"image": ("test_id.jpg", test_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_extract_response_schema(self, client, test_image_bytes):
        """Test extract response has all required fields."""
        response = client.post(
            "/extract",
            files={"image": ("test_id.jpg", test_image_bytes, "image/jpeg")},
        )
        data = response.json()
        assert "fields" in data
        assert "confidence" in data
        assert "raw_output" in data
        assert isinstance(data["fields"], dict)

    def test_extract_no_image(self, client):
        """Test extract endpoint without image returns 422."""
        response = client.post("/extract")
        assert response.status_code == 422

    def test_extract_invalid_image(self, client):
        """Test extract endpoint with invalid image data returns 400."""
        response = client.post(
            "/extract",
            files={"image": ("bad.jpg", b"not an image", "image/jpeg")},
        )
        assert response.status_code == 400


# ──────────────────────────────────────────────
# Verify Endpoint Tests
# ──────────────────────────────────────────────

class TestVerifyEndpoint:
    """Tests for /verify endpoint."""

    def test_verify_returns_200(self, client, test_image_bytes):
        """Test that verify endpoint accepts an image and returns 200."""
        response = client.post(
            "/verify",
            files={"image": ("test_id.jpg", test_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_verify_response_schema(self, client, test_image_bytes):
        """Test verify response has all required fields."""
        response = client.post(
            "/verify",
            files={"image": ("test_id.jpg", test_image_bytes, "image/jpeg")},
        )
        data = response.json()
        assert "verdict" in data
        assert "confidence" in data
        assert "reason" in data

    def test_verify_no_image(self, client):
        """Test verify endpoint without image returns 422."""
        response = client.post("/verify")
        assert response.status_code == 422

    def test_verify_invalid_image(self, client):
        """Test verify endpoint with invalid image data returns 400."""
        response = client.post(
            "/verify",
            files={"image": ("bad.jpg", b"not an image", "image/jpeg")},
        )
        assert response.status_code == 400


# ──────────────────────────────────────────────
# Integration Tests
# ──────────────────────────────────────────────

class TestIntegration:
    """Integration tests across endpoints."""

    def test_extract_then_verify(self, client, test_image_bytes):
        """Test calling extract followed by verify on the same image."""
        extract_response = client.post(
            "/extract",
            files={"image": ("test_id.jpg", test_image_bytes, "image/jpeg")},
        )
        assert extract_response.status_code == 200

        verify_response = client.post(
            "/verify",
            files={"image": ("test_id.jpg", test_image_bytes, "image/jpeg")},
        )
        assert verify_response.status_code == 200

    def test_multiple_extractions(self, client, test_image_bytes):
        """Test multiple consecutive extract calls."""
        for _ in range(3):
            response = client.post(
                "/extract",
                files={"image": ("test_id.jpg", test_image_bytes, "image/jpeg")},
            )
            assert response.status_code == 200

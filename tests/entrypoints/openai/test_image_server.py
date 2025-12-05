# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for image generation API server.

This module contains unit tests and integration tests (with mocking) for the
OpenAI-compatible text-to-image generation API.
"""

import base64
import io
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from vllm_omni.entrypoints.openai.image_server import (
    create_app,
    encode_image_base64,
    parse_size,
)
from vllm_omni.entrypoints.openai.protocol.images import ResponseFormat


# Unit Tests


def test_parse_size_valid():
    """Test size parsing with valid inputs"""
    assert parse_size("1024x1024") == (1024, 1024)
    assert parse_size("512x768") == (512, 768)
    assert parse_size("256x256") == (256, 256)
    assert parse_size("1792x1024") == (1792, 1024)
    assert parse_size("1024x1792") == (1024, 1792)


def test_parse_size_invalid():
    """Test size parsing with invalid inputs"""
    with pytest.raises(ValueError, match="Invalid size format"):
        parse_size("invalid")

    with pytest.raises(ValueError, match="Invalid size format"):
        parse_size("1024")

    with pytest.raises(ValueError, match="Invalid size format"):
        parse_size("1024x")

    with pytest.raises(ValueError, match="Invalid size format"):
        parse_size("x1024")


def test_parse_size_negative():
    """Test size parsing with negative or zero dimensions"""
    with pytest.raises(ValueError, match="positive integers"):
        parse_size("0x1024")

    with pytest.raises(ValueError, match="positive integers"):
        parse_size("1024x0")

    with pytest.raises(ValueError):
        parse_size("-1024x1024")


def test_encode_image_base64():
    """Test image encoding to base64"""
    # Create a simple test image
    img = Image.new("RGB", (64, 64), color="red")
    b64_str = encode_image_base64(img)

    # Should be valid base64
    assert isinstance(b64_str, str)
    assert len(b64_str) > 0

    # Should decode back to PNG
    decoded = base64.b64decode(b64_str)
    decoded_img = Image.open(io.BytesIO(decoded))

    # Verify properties
    assert decoded_img.size == (64, 64)
    assert decoded_img.format == "PNG"


# Integration Tests (with mocking)


@pytest.fixture
def mock_omni():
    """Mock Omni instance that returns fake images"""
    mock = Mock()

    def generate(**kwargs):
        # Return n PIL images
        n = kwargs.get("num_images_per_prompt", 1)
        return [Image.new("RGB", (64, 64), color="blue") for _ in range(n)]

    mock.generate = generate
    return mock


@pytest.fixture
def test_client(mock_omni):
    """Create test client with mocked model"""
    with patch("vllm_omni.entrypoints.openai.image_server.Omni", return_value=mock_omni):
        app = create_app(model="test-model")

        # Manually set the global instance for testing
        import vllm_omni.entrypoints.openai.image_server as server_module

        server_module.omni_instance = mock_omni

        return TestClient(app)


def test_health_endpoint(test_client):
    """Test health check endpoint"""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert data["ready"] is True


def test_generate_single_image(test_client):
    """Test generating a single image"""
    response = test_client.post(
        "/v1/images/generations",
        json={
            "prompt": "a cat",
            "n": 1,
            "size": "1024x1024",
        },
    )
    assert response.status_code == 200
    data = response.json()

    # Check response structure
    assert "created" in data
    assert isinstance(data["created"], int)
    assert "data" in data
    assert len(data["data"]) == 1
    assert "b64_json" in data["data"][0]

    # Verify image can be decoded
    img_bytes = base64.b64decode(data["data"][0]["b64_json"])
    img = Image.open(io.BytesIO(img_bytes))
    assert img.size == (64, 64)  # Our mock returns 64x64 images


def test_generate_multiple_images(test_client):
    """Test generating multiple images"""
    response = test_client.post(
        "/v1/images/generations",
        json={
            "prompt": "a dog",
            "n": 3,
            "size": "512x512",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 3

    # All images should be valid
    for img_data in data["data"]:
        assert "b64_json" in img_data
        img_bytes = base64.b64decode(img_data["b64_json"])
        img = Image.open(io.BytesIO(img_bytes))
        assert img.format == "PNG"


def test_with_negative_prompt(test_client):
    """Test with negative prompt"""
    response = test_client.post(
        "/v1/images/generations",
        json={
            "prompt": "beautiful landscape",
            "negative_prompt": "blurry, low quality",
            "size": "1024x1024",
        },
    )
    assert response.status_code == 200


def test_with_seed(test_client):
    """Test with seed for reproducibility"""
    response = test_client.post(
        "/v1/images/generations",
        json={
            "prompt": "a tree",
            "seed": 42,
            "size": "1024x1024",
        },
    )
    assert response.status_code == 200


def test_with_custom_parameters(test_client):
    """Test with custom diffusion parameters"""
    response = test_client.post(
        "/v1/images/generations",
        json={
            "prompt": "a mountain",
            "size": "1024x1024",
            "num_inference_steps": 100,
            "true_cfg_scale": 5.5,
            "seed": 123,
        },
    )
    assert response.status_code == 200


def test_invalid_size(test_client):
    """Test with invalid size parameter"""
    response = test_client.post(
        "/v1/images/generations",
        json={
            "prompt": "a cat",
            "size": "invalid",
        },
    )
    assert response.status_code == 400
    assert "invalid" in response.json()["detail"].lower()


def test_missing_prompt(test_client):
    """Test with missing required prompt field"""
    response = test_client.post(
        "/v1/images/generations",
        json={
            "size": "1024x1024",
        },
    )
    # Pydantic validation error
    assert response.status_code == 422


def test_invalid_n_parameter(test_client):
    """Test with invalid n parameter (out of range)"""
    # n < 1
    response = test_client.post(
        "/v1/images/generations",
        json={
            "prompt": "a cat",
            "n": 0,
        },
    )
    assert response.status_code == 422

    # n > 10
    response = test_client.post(
        "/v1/images/generations",
        json={
            "prompt": "a cat",
            "n": 11,
        },
    )
    assert response.status_code == 422


def test_url_response_format_not_supported(test_client):
    """Test that URL format returns error"""
    response = test_client.post(
        "/v1/images/generations",
        json={
            "prompt": "a cat",
            "response_format": "url",
        },
    )
    assert response.status_code == 400
    assert "not supported" in response.json()["detail"].lower()


def test_model_not_loaded():
    """Test error when model is not loaded"""
    # Create app without setting omni_instance
    app = create_app(model="test-model")

    # Clear the instance to simulate not loaded state
    import vllm_omni.entrypoints.openai.image_server as server_module

    server_module.omni_instance = None

    client = TestClient(app)
    response = client.post(
        "/v1/images/generations",
        json={
            "prompt": "a cat",
        },
    )
    assert response.status_code == 503
    assert "not loaded" in response.json()["detail"].lower()


def test_different_image_sizes(test_client):
    """Test various valid image sizes"""
    sizes = ["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]

    for size in sizes:
        response = test_client.post(
            "/v1/images/generations",
            json={
                "prompt": "a test image",
                "size": size,
            },
        )
        assert response.status_code == 200, f"Failed for size {size}"


def test_parameter_validation():
    """Test Pydantic model validation"""
    from vllm_omni.entrypoints.openai.protocol.images import ImageGenerationRequest

    # Valid request
    req = ImageGenerationRequest(prompt="test")
    assert req.prompt == "test"
    assert req.n == 1
    assert req.num_inference_steps == 50
    assert req.true_cfg_scale == 4.0

    # Invalid num_inference_steps (out of range)
    with pytest.raises(ValueError):
        ImageGenerationRequest(prompt="test", num_inference_steps=0)

    with pytest.raises(ValueError):
        ImageGenerationRequest(prompt="test", num_inference_steps=201)

    # Invalid guidance_scale (out of range)
    with pytest.raises(ValueError):
        ImageGenerationRequest(prompt="test", guidance_scale=-1.0)

    with pytest.raises(ValueError):
        ImageGenerationRequest(prompt="test", guidance_scale=21.0)

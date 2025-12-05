# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FastAPI server for OpenAI-compatible text-to-image generation.

This module provides a standalone HTTP API server that wraps vllm-omni's
diffusion capabilities with an OpenAI DALL-E compatible interface.
"""

import base64
import io
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from vllm.logger import init_logger
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.entrypoints.openai.protocol.images import (
    ImageData,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ResponseFormat,
)

logger = init_logger(__name__)

# Global state for model instance (managed via lifespan)
omni_instance: Optional[Omni] = None
model_name: str = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage model lifecycle.

    Loads the diffusion model on startup and cleans up on shutdown.
    """
    global omni_instance, model_name

    logger.info(f"Loading diffusion model: {model_name}")
    try:
        omni_instance = Omni(
            model=model_name,
            vae_use_slicing=True,  # Memory optimization for large images
            vae_use_tiling=True,  # Memory optimization for large images
        )
        logger.info("Model loaded successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise
    finally:
        logger.info("Shutting down image server")
        omni_instance = None


def create_app(model: str) -> FastAPI:
    """
    Factory function to create the FastAPI application.

    Args:
        model: Model name or path (e.g., "Qwen/Qwen-Image")

    Returns:
        Configured FastAPI application
    """
    global model_name
    model_name = model

    app = FastAPI(
        title="vLLM-Omni Image Generation API",
        description="OpenAI DALL-E compatible text-to-image generation API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health():
        """Health check endpoint"""
        return {"status": "ok", "model": model_name, "ready": omni_instance is not None}

    @app.post("/v1/images/generations", response_model=ImageGenerationResponse)
    async def create_image(request: ImageGenerationRequest) -> ImageGenerationResponse:
        """
        Generate images from text prompts.

        This endpoint follows the OpenAI DALL-E API specification while providing
        additional parameters for fine-tuning the diffusion process.

        Args:
            request: Image generation request with prompt and parameters

        Returns:
            ImageGenerationResponse with base64-encoded PNG images

        Raises:
            HTTPException: On validation errors (400), server errors (500), or service unavailable (503)
        """
        global omni_instance

        if omni_instance is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Server may still be initializing.",
            )

        try:
            # Parse size parameter (e.g., "1024x1024" → (1024, 1024))
            width, height = parse_size(request.size)

            # Create generator for reproducibility if seed is provided
            generator = None
            if request.seed is not None:
                from vllm_omni.utils.platform_utils import detect_device_type

                device = detect_device_type()
                generator = torch.Generator(device=device).manual_seed(request.seed)

            # Log generation request
            logger.info(
                f"Generating {request.n} image(s) - prompt: '{request.prompt[:50]}...' "
                f"size: {width}x{height}, steps: {request.num_inference_steps}, "
                f"cfg: {request.true_cfg_scale}, seed: {request.seed}"
            )

            # Build generation parameters, filtering out None values
            # (vllm-omni's validation doesn't handle None comparisons well)
            gen_params = {
                "prompt": request.prompt,
                "height": height,
                "width": width,
                "num_images_per_prompt": request.n,
                "num_outputs_per_prompt": request.n,
                "num_inference_steps": request.num_inference_steps,
            }

            # Add optional parameters only if not None
            if request.negative_prompt is not None:
                gen_params["negative_prompt"] = request.negative_prompt
            if request.true_cfg_scale is not None:
                gen_params["true_cfg_scale"] = request.true_cfg_scale
            if request.guidance_scale is not None:
                gen_params["guidance_scale"] = request.guidance_scale
            if generator is not None:
                gen_params["generator"] = generator

            # Generate images using the diffusion engine
            images = omni_instance.generate(**gen_params)

            logger.info(f"Successfully generated {len(images)} image(s)")

            # Encode images based on response format
            if request.response_format == ResponseFormat.B64_JSON:
                image_data = [
                    ImageData(b64_json=encode_image_base64(img), revised_prompt=None)
                    for img in images
                ]
            elif request.response_format == ResponseFormat.URL:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="'url' response format is not supported. Use 'b64_json' instead.",
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported response_format: {request.response_format}",
                )

            return ImageGenerationResponse(
                created=int(time.time()),
                data=image_data,
            )

        except ValueError as e:
            # Validation errors (e.g., invalid size format)
            logger.error(f"Validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )
        except Exception as e:
            # Unexpected errors during generation
            logger.error(f"Image generation failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Image generation failed: {str(e)}",
            )

    return app


def parse_size(size_str: str) -> tuple[int, int]:
    """
    Parse size string to width and height tuple.

    Args:
        size_str: Size in format "WIDTHxHEIGHT" (e.g., "1024x1024")

    Returns:
        Tuple of (width, height)

    Raises:
        ValueError: If size format is invalid
    """
    try:
        parts = size_str.split("x")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid size format: '{size_str}'. Expected format: 'WIDTHxHEIGHT' (e.g., '1024x1024')"
            )

        width = int(parts[0])
        height = int(parts[1])

        if width <= 0 or height <= 0:
            raise ValueError(f"Width and height must be positive integers, got: {width}x{height}")

        return width, height
    except ValueError as e:
        # Re-raise ValueError with clear message
        raise ValueError(f"Failed to parse size '{size_str}': {e}")


def encode_image_base64(image) -> str:
    """
    Encode PIL Image to base64 PNG string.

    Args:
        image: PIL Image object

    Returns:
        Base64-encoded PNG image as string
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

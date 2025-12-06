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
from vllm_omni.entrypoints.openai.image_model_profiles import (
    DiffusionModelProfile,
    get_model_profile,
)
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
model_profile: Optional[DiffusionModelProfile] = None


def validate_request_model(request_model: Optional[str], server_model: str) -> None:
    """
    Validate that request model matches server model.

    Args:
        request_model: Model name from request (may be None)
        server_model: Model name configured for this server instance

    Raises:
        HTTPException: If request_model is specified but doesn't match server_model
    """
    if request_model is not None and request_model != server_model:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Model mismatch: request specifies '{request_model}' but "
                f"server is running '{server_model}'. Either omit the 'model' "
                f"field or use '{server_model}'."
            ),
        )


def build_generation_params(
    request: ImageGenerationRequest,
    profile: DiffusionModelProfile,
    width: int,
    height: int,
    generator: Optional[torch.Generator],
) -> dict:
    """
    Build generation kwargs for Omni.generate() using model profile.

    This function translates the API request parameters into model-specific
    generation kwargs, applying defaults, constraints, and forced values from
    the model profile.

    Args:
        request: Image generation request
        profile: Model profile with defaults and constraints
        width: Image width in pixels
        height: Image height in pixels
        generator: Optional random generator for reproducibility

    Returns:
        Dictionary of kwargs to pass to omni_instance.generate()

    Raises:
        ValueError: If parameters violate model constraints (e.g., steps exceed max)
    """
    gen_params = {
        "prompt": request.prompt,
        "height": height,
        "width": width,
        "num_images_per_prompt": request.n,
        "num_outputs_per_prompt": request.n,
    }

    # Apply model defaults and limits for num_inference_steps
    num_steps = request.num_inference_steps or profile.default_num_inference_steps
    if num_steps > profile.max_num_inference_steps:
        raise ValueError(
            f"num_inference_steps={num_steps} exceeds maximum for "
            f"{profile.model_name} (max={profile.max_num_inference_steps})"
        )
    gen_params["num_inference_steps"] = num_steps

    # Add negative_prompt if supported
    if request.negative_prompt and profile.supports_negative_prompt:
        gen_params["negative_prompt"] = request.negative_prompt

    # Handle guidance_scale with forced override support
    if profile.supports_guidance_scale:
        if profile.force_guidance_scale is not None:
            # Model requires specific guidance_scale value (e.g., Z-Image Turbo requires 0.0)
            gen_params["guidance_scale"] = profile.force_guidance_scale
            if (
                request.guidance_scale is not None
                and request.guidance_scale != profile.force_guidance_scale
            ):
                logger.warning(
                    f"Ignoring guidance_scale={request.guidance_scale}, "
                    f"{profile.model_name} requires guidance_scale={profile.force_guidance_scale}"
                )
        elif request.guidance_scale is not None:
            # Use user-provided value
            gen_params["guidance_scale"] = request.guidance_scale
        elif profile.default_guidance_scale is not None:
            # Use profile default
            gen_params["guidance_scale"] = profile.default_guidance_scale

    # Handle true_cfg_scale (Qwen-specific)
    if profile.supports_true_cfg_scale:
        cfg = request.true_cfg_scale or profile.default_true_cfg_scale
        if cfg is not None:
            gen_params["true_cfg_scale"] = cfg
    elif request.true_cfg_scale is not None:
        # Warn if user provides true_cfg_scale for non-Qwen model
        logger.warning(
            f"Ignoring true_cfg_scale={request.true_cfg_scale}, "
            f"{profile.model_name} doesn't support this parameter"
        )

    # Add generator for reproducibility
    if generator:
        gen_params["generator"] = generator

    return gen_params


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage model lifecycle.

    Loads the diffusion model on startup and cleans up on shutdown.
    """
    global omni_instance, model_name, model_profile

    logger.info(f"Loading diffusion model: {model_name}")
    try:
        # Load model profile
        model_profile = get_model_profile(model_name)
        logger.info(
            f"Model profile loaded - "
            f"steps: {model_profile.default_num_inference_steps} "
            f"(max {model_profile.max_num_inference_steps})"
        )

        # Create Omni instance with profile-specific kwargs
        omni_kwargs = {"model": model_name, **model_profile.omni_kwargs}
        omni_instance = Omni(**omni_kwargs)

        logger.info("Model loaded successfully")
        yield
    except ValueError as e:
        # Model not supported (raised by get_model_profile)
        logger.error(f"Unsupported model: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise
    finally:
        logger.info("Shutting down image server")
        omni_instance = None
        model_profile = None


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
        """Health check endpoint with model profile information"""
        return {
            "status": "ok",
            "model": model_name,
            "ready": omni_instance is not None,
            "profile": (
                {
                    "default_steps": model_profile.default_num_inference_steps,
                    "max_steps": model_profile.max_num_inference_steps,
                }
                if model_profile
                else None
            ),
        }

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
        global omni_instance, model_name, model_profile

        if omni_instance is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Server may still be initializing.",
            )

        try:
            # Validate request model matches server model
            validate_request_model(request.model, model_name)

            # Parse size parameter (e.g., "1024x1024" → (1024, 1024))
            width, height = parse_size(request.size)

            # Create generator for reproducibility if seed is provided
            generator = None
            if request.seed is not None:
                from vllm_omni.utils.platform_utils import detect_device_type

                device = detect_device_type()
                generator = torch.Generator(device=device).manual_seed(request.seed)

            # Build generation parameters using model profile
            gen_params = build_generation_params(
                request, model_profile, width, height, generator
            )

            # Log generation request
            logger.info(
                f"[{model_name}] Generating {request.n} image(s) - "
                f"prompt: '{request.prompt[:50]}...', size: {width}x{height}, "
                f"steps: {gen_params['num_inference_steps']}, seed: {request.seed}"
            )

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

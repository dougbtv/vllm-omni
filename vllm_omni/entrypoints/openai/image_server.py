# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FastAPI server for OpenAI-compatible text-to-image generation.

This module provides a standalone HTTP API server that wraps vllm-omni's
diffusion capabilities with an OpenAI DALL-E compatible interface.
"""

import base64
import io
import math
import time
from contextlib import asynccontextmanager
from typing import Optional

import PIL.Image
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
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


def validate_image_file(file: UploadFile, max_size_mb: int = 4) -> None:
    """
    Validate uploaded image file.

    Args:
        file: Uploaded file from FastAPI
        max_size_mb: Maximum file size in megabytes

    Raises:
        HTTPException: If file is invalid or too large
    """
    # Check file is provided
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Image file is required"
        )

    # Check content type
    allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image format: {file.content_type}. Supported: PNG, JPEG, WebP",
        )

    # Check file size (read first chunk to estimate)
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning

    max_bytes = max_size_mb * 1024 * 1024
    if file_size > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image file too large: {file_size / 1024 / 1024:.1f}MB (max {max_size_mb}MB)",
        )


async def read_image_file(file: UploadFile) -> PIL.Image.Image:
    """
    Read uploaded file into PIL Image.

    Args:
        file: Uploaded file from FastAPI

    Returns:
        PIL Image object

    Raises:
        HTTPException: If file cannot be read as image
    """
    try:
        contents = await file.read()
        image = PIL.Image.open(io.BytesIO(contents))

        # Convert to RGB if needed (handle RGBA, grayscale, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to read image file: {str(e)}"
        )
    finally:
        await file.seek(0)  # Reset file pointer for potential reuse


def build_edit_params(
    prompt: str,
    image: PIL.Image.Image,
    profile: DiffusionModelProfile,
    negative_prompt: Optional[str],
    n: int,
    size: Optional[str],
    num_inference_steps: Optional[int],
    guidance_scale: Optional[float],
    true_cfg_scale: Optional[float],
    generator: Optional[torch.Generator],
) -> dict:
    """
    Build edit generation kwargs for Omni.generate() using model profile.

    Similar to build_generation_params but handles image input and auto-sizing.

    Args:
        prompt: Edit instruction
        image: Input PIL Image
        profile: Model profile with defaults and constraints
        negative_prompt: Optional negative prompt
        n: Number of images to generate
        size: Optional output size (if None, auto-calculate from input)
        num_inference_steps: Optional custom step count
        guidance_scale: Optional CFG scale
        true_cfg_scale: Optional true CFG scale
        generator: Optional random generator

    Returns:
        Dictionary of kwargs to pass to omni_instance.generate()
    """
    # Calculate dimensions from input image if size not specified
    if size:
        width, height = parse_size(size)
    else:
        # Auto-calculate maintaining ~1024x1024 area
        # This matches the pipeline's calculate_dimensions behavior
        img_width, img_height = image.size
        aspect_ratio = img_width / img_height
        target_area = 1024 * 1024

        width = int(math.sqrt(target_area * aspect_ratio))
        height = int(width / aspect_ratio)

        # Round to multiples of 32 (VAE constraint)
        width = (width // 32) * 32
        height = (height // 32) * 32

        logger.info(
            f"Auto-calculated dimensions from input {img_width}x{img_height}: "
            f"{width}x{height} (aspect ratio: {aspect_ratio:.2f})"
        )

    gen_params = {
        "prompt": prompt,
        "image": image,  # Pass PIL Image directly
        "height": height,
        "width": width,
        "num_images_per_prompt": n,
        "num_outputs_per_prompt": n,
    }

    # Apply model defaults and limits for num_inference_steps
    num_steps = num_inference_steps or profile.default_num_inference_steps
    if num_steps > profile.max_num_inference_steps:
        raise ValueError(
            f"num_inference_steps={num_steps} exceeds maximum for "
            f"{profile.model_name} (max={profile.max_num_inference_steps})"
        )
    gen_params["num_inference_steps"] = num_steps

    # Add negative_prompt if supported
    if negative_prompt and profile.supports_negative_prompt:
        gen_params["negative_prompt"] = negative_prompt

    # Handle guidance_scale
    if profile.supports_guidance_scale:
        if profile.force_guidance_scale is not None:
            gen_params["guidance_scale"] = profile.force_guidance_scale
            if guidance_scale is not None and guidance_scale != profile.force_guidance_scale:
                logger.warning(
                    f"Ignoring guidance_scale={guidance_scale}, "
                    f"{profile.model_name} requires guidance_scale={profile.force_guidance_scale}"
                )
        elif guidance_scale is not None:
            gen_params["guidance_scale"] = guidance_scale
        elif profile.default_guidance_scale is not None:
            gen_params["guidance_scale"] = profile.default_guidance_scale

    # Handle true_cfg_scale
    if profile.supports_true_cfg_scale:
        cfg = true_cfg_scale or profile.default_true_cfg_scale
        if cfg is not None:
            gen_params["true_cfg_scale"] = cfg
    elif true_cfg_scale is not None:
        logger.warning(
            f"Ignoring true_cfg_scale={true_cfg_scale}, "
            f"{profile.model_name} doesn't support this parameter"
        )

    # Add generator
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

    @app.post("/v1/images/edits", response_model=ImageGenerationResponse)
    async def edit_image(
        prompt: str = Form(..., description="Text instruction for how to edit the image"),
        image: UploadFile = File(..., description="Image to edit"),
        model: Optional[str] = Form(None),
        mask: Optional[UploadFile] = File(None),
        n: Optional[int] = Form(1),
        size: Optional[str] = Form(None),
        response_format: Optional[str] = Form("b64_json"),
        negative_prompt: Optional[str] = Form(None),
        num_inference_steps: Optional[int] = Form(None),
        guidance_scale: Optional[float] = Form(None),
        true_cfg_scale: Optional[float] = Form(None),
        seed: Optional[int] = Form(None),
        user: Optional[str] = Form(None),
    ) -> ImageGenerationResponse:
        """
        Edit images based on text instructions.

        This endpoint follows the OpenAI DALL-E image editing API specification.
        It takes an input image and a text prompt describing the desired edits.

        Args:
            prompt: Text instruction for how to edit the image
            image: Image file to edit (PNG, JPEG, WebP, max 4MB)
            model: Model name (optional, must match server if specified)
            mask: Mask image (PNG with transparency) - CURRENTLY IGNORED with warning
            n: Number of edited images to generate (1-10)
            size: Output size in WxH format (auto-calculated from input if omitted)
            response_format: Response format (b64_json only)
            negative_prompt: What to avoid in the edited image
            num_inference_steps: Number of diffusion steps
            guidance_scale: CFG scale
            true_cfg_scale: True CFG scale (Qwen-specific)
            seed: Random seed for reproducibility
            user: User identifier

        Returns:
            ImageGenerationResponse with base64-encoded edited images

        Raises:
            HTTPException: On validation errors (400), server errors (500), or unavailable (503)
        """
        global omni_instance, model_name, model_profile

        if omni_instance is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Server may still be initializing.",
            )

        try:
            # Validate request model matches server model
            validate_request_model(model, model_name)

            # Validate and read input image
            validate_image_file(image, max_size_mb=4)
            pil_image = await read_image_file(image)

            # Handle mask parameter (currently unsupported)
            if mask is not None:
                logger.warning(
                    "Mask parameter provided but masking/inpainting is not currently supported. "
                    "The mask will be ignored and full-image editing will be performed."
                )

            # Validate n parameter
            if n is not None and (n < 1 or n > 10):
                raise ValueError("n must be between 1 and 10")

            # Validate response_format
            if response_format and response_format != "b64_json":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported response_format: {response_format}. Only 'b64_json' is supported.",
                )

            # Create generator for reproducibility if seed is provided
            generator = None
            if seed is not None:
                from vllm_omni.utils.platform_utils import detect_device_type

                device = detect_device_type()
                generator = torch.Generator(device=device).manual_seed(seed)

            # Build edit parameters using model profile
            gen_params = build_edit_params(
                prompt=prompt,
                image=pil_image,
                profile=model_profile,
                negative_prompt=negative_prompt,
                n=n or 1,
                size=size,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                true_cfg_scale=true_cfg_scale,
                generator=generator,
            )

            # Log edit request
            logger.info(
                f"[{model_name}] Editing image - "
                f"prompt: '{prompt[:50]}...', "
                f"input size: {pil_image.size}, "
                f"output size: {gen_params['width']}x{gen_params['height']}, "
                f"n: {n}, steps: {gen_params['num_inference_steps']}, seed: {seed}"
            )

            # Generate edited images using the diffusion engine
            images = omni_instance.generate(**gen_params)

            logger.info(f"Successfully generated {len(images)} edited image(s)")

            # Encode images to base64
            image_data = [
                ImageData(b64_json=encode_image_base64(img), revised_prompt=None) for img in images
            ]

            return ImageGenerationResponse(
                created=int(time.time()),
                data=image_data,
            )

        except ValueError as e:
            # Validation errors
            logger.error(f"Validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )
        except Exception as e:
            # Unexpected errors during editing
            logger.error(f"Image editing failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Image editing failed: {str(e)}",
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

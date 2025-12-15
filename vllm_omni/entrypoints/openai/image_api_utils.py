# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shared helper utilities for OpenAI-compatible image generation API.

This module provides common helper functions for the image generation endpoint.
All functions work with plain Python types to maintain separation from the
FastAPI HTTP layer.
"""

import base64
import io

import PIL.Image
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.image_model_profiles import DiffusionModelProfile
from vllm_omni.entrypoints.openai.protocol.images import ImageGenerationRequest

logger = init_logger(__name__)


def parse_size(size_str: str) -> tuple[int, int]:
    """Parse size string to width and height tuple.

    Args:
        size_str: Size in format "WIDTHxHEIGHT" (e.g., "1024x1024")

    Returns:
        Tuple of (width, height)

    Raises:
        ValueError: If size format is invalid
    """
    if not size_str or not isinstance(size_str, str):
        raise ValueError(
            f"Size must be a non-empty string in format 'WIDTHxHEIGHT' (e.g., '1024x1024'), got: {size_str}"
        )

    parts = size_str.split("x")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid size format: '{size_str}'. Expected format: 'WIDTHxHEIGHT' (e.g., '1024x1024'). "
            f"Did you mean to use 'x' as separator?"
        )

    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid size format: '{size_str}'. Width and height must be integers.")

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid size: {width}x{height}. Width and height must be positive integers.")

    return width, height


def encode_image_base64(image: PIL.Image.Image) -> str:
    """Encode PIL Image to base64 PNG string.

    Args:
        image: PIL Image object

    Returns:
        Base64-encoded PNG image as string
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def build_generation_params(
    request: ImageGenerationRequest,
    profile: DiffusionModelProfile,
    width: int,
    height: int,
) -> dict:
    """Build generation kwargs for AsyncOmniDiffusion.generate().

    Args:
        request: Image generation request
        profile: Model profile with defaults and constraints
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Dictionary of kwargs to pass to diffusion_engine.generate()
    """
    gen_params = {
        "prompt": request.prompt,
        "height": height,
        "width": width,
        "num_outputs_per_prompt": request.n,
    }

    num_steps = request.num_inference_steps or profile.default_num_inference_steps
    if num_steps > profile.max_num_inference_steps:
        raise ValueError(
            f"num_inference_steps={num_steps} exceeds maximum for "
            f"{profile.model_name} (max={profile.max_num_inference_steps})"
        )
    gen_params["num_inference_steps"] = num_steps

    if request.negative_prompt and profile.supports_negative_prompt:
        gen_params["negative_prompt"] = request.negative_prompt

    if profile.supports_guidance_scale:
        if profile.force_guidance_scale is not None:
            gen_params["guidance_scale"] = profile.force_guidance_scale
            if request.guidance_scale is not None and request.guidance_scale != profile.force_guidance_scale:
                logger.warning(
                    f"Ignoring guidance_scale={request.guidance_scale}, "
                    f"{profile.model_name} requires guidance_scale={profile.force_guidance_scale}"
                )
        elif request.guidance_scale is not None:
            gen_params["guidance_scale"] = request.guidance_scale
        elif profile.default_guidance_scale is not None:
            gen_params["guidance_scale"] = profile.default_guidance_scale

    if profile.supports_true_cfg_scale:
        cfg = request.true_cfg_scale or profile.default_true_cfg_scale
        if cfg is not None:
            gen_params["true_cfg_scale"] = cfg
    elif request.true_cfg_scale is not None:
        logger.warning(
            f"Ignoring true_cfg_scale={request.true_cfg_scale}, {profile.model_name} doesn't support this parameter"
        )

    if request.seed is not None:
        gen_params["seed"] = request.seed

    return gen_params

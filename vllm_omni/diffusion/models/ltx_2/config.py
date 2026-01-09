# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration utilities for LTX-2 model."""

import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)


def validate_ltx2_dimensions(height: int, width: int, num_frames: int) -> None:
    """Validate LTX-2 dimension constraints.

    LTX-2 requires:
    - height and width divisible by 32 (VAE spatial downsampling factor)
    - num_frames must be 8K + 1 (VAE temporal downsampling constraint)

    Args:
        height: Video height in pixels
        width: Video width in pixels
        num_frames: Number of frames

    Raises:
        ValueError: If dimensions don't meet LTX-2 requirements
    """
    if height % 32 != 0:
        raise ValueError(
            f"LTX-2 requires height divisible by 32, got {height}. "
            f"Try {(height // 32) * 32} or {((height // 32) + 1) * 32}."
        )

    if width % 32 != 0:
        raise ValueError(
            f"LTX-2 requires width divisible by 32, got {width}. "
            f"Try {(width // 32) * 32} or {((width // 32) + 1) * 32}."
        )

    if (num_frames - 1) % 8 != 0:
        k = (num_frames - 1) // 8
        suggested_low = 8 * k + 1
        suggested_high = 8 * (k + 1) + 1
        raise ValueError(
            f"LTX-2 requires num_frames = 8K + 1, got {num_frames}. "
            f"Try {suggested_low} or {suggested_high}."
        )

    logger.info(
        f"LTX-2 dimension validation passed: "
        f"{width}x{height}, {num_frames} frames"
    )


def load_ltx2_model_config(checkpoint_path: str) -> Dict:
    """Load model configuration from checkpoint metadata.

    Args:
        checkpoint_path: Path to LTX-2 checkpoint file

    Returns:
        Dictionary containing model configuration

    Note:
        This is a placeholder for PoC. Full implementation will parse
        safetensors metadata or accompanying config.json files.
    """
    # TODO: Implement actual config loading from checkpoint
    logger.warning(
        f"load_ltx2_model_config not fully implemented (PoC phase). "
        f"Returning default config for checkpoint: {checkpoint_path}"
    )

    return {
        "model_type": "ltx-video",
        "version": "2.0",
        "transformer_dim": 2048,
        "num_layers": 32,
        "attention_heads": 32,
    }


def get_default_ltx2_params() -> Dict:
    """Return default generation parameters for LTX-2.

    Returns:
        Dictionary of default parameters
    """
    return {
        "num_inference_steps": 40,
        "guidance_scale": 4.0,
        "fps": 24.0,
        "height": 512,
        "width": 768,
        "num_frames": 121,  # 8*15 + 1 = 121 frames (~5 seconds at 24fps)
    }


def resolve_ltx2_model_paths(model_path: str) -> Dict[str, str]:
    """Resolve paths to LTX-2 model components.

    Args:
        model_path: Base model path (directory or HuggingFace model ID)

    Returns:
        Dictionary with paths to checkpoint, gemma, etc.

    Example directory structure:
        model_path/
        ├── checkpoint.safetensors (or ltx-video-2b-v1.0.safetensors)
        └── gemma/                (text encoder)
    """
    paths = {}

    if os.path.isdir(model_path):
        # Local directory - look for standard files
        checkpoint_candidates = [
            "checkpoint.safetensors",
            "ltx-video-2b-v1.0.safetensors",
            "model.safetensors",
        ]

        for candidate in checkpoint_candidates:
            checkpoint_path = os.path.join(model_path, candidate)
            if os.path.exists(checkpoint_path):
                paths["checkpoint"] = checkpoint_path
                break

        gemma_path = os.path.join(model_path, "gemma")
        if os.path.exists(gemma_path):
            paths["gemma"] = gemma_path

    else:
        # Assume HuggingFace model ID
        logger.info(
            f"Model path '{model_path}' is not a local directory. "
            f"Assuming HuggingFace model ID (download not implemented in PoC)."
        )
        paths["checkpoint"] = model_path
        paths["gemma"] = model_path

    return paths

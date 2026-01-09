# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Component wrapper classes for LTX-2 pipeline.

These classes isolate LTX-2 specific types from vLLM-Omni core,
providing a clean interface for the diffusion components.
"""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class LTX2Modality:
    """Wrapper for ltx_core.model.transformer.Modality.

    Represents a single modality (video or audio) in the LTX-2 pipeline.

    Attributes:
        latent: Latent tensor tokens, shape (B, T, D)
        timesteps: Timestep embeddings, shape (B, T)
        positions: Positional indices, shape (B, 3, T) for video
        context: Text conditioning embeddings
        enabled: Whether this modality is active
        context_mask: Optional attention mask for context
    """
    latent: torch.Tensor
    timesteps: torch.Tensor
    positions: torch.Tensor
    context: torch.Tensor
    enabled: bool = True
    context_mask: torch.Tensor | None = None


@dataclass
class LTX2PipelineComponents:
    """Container for LTX-2 diffusion components.

    Holds the scheduler, noiser, stepper, guider, and patchifiers
    used during the diffusion process.

    Attributes:
        scheduler: LTX2Scheduler for sigma schedule generation
        noiser: GaussianNoiser for adding noise to latents
        stepper: EulerDiffusionStep for denoising step implementation
        guider: CFGGuider for classifier-free guidance
        video_patchifier: Video patchification utility
        audio_patchifier: Audio patchification utility
        dtype: Data type for computations
        device: Device for tensor operations
    """
    scheduler: Any  # LTX2Scheduler
    noiser: Any  # GaussianNoiser
    stepper: Any  # EulerDiffusionStep
    guider: Any  # CFGGuider
    video_patchifier: Any
    audio_patchifier: Any
    dtype: torch.dtype
    device: torch.device

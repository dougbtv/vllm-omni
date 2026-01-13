# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LTX-2 audio-video generation pipeline.

This module implements the LTX-2 (Lightricks Video-2) diffusion pipeline
for text-to-video and image-to-video generation with audio synthesis.

Based on TI2VidOneStagePipeline from the LTX-2 reference implementation.
"""

import logging
from collections.abc import Iterable

import torch
from torch import nn

# Lazy imports for LTX-2 dependencies
try:
    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    from ltx_core.components.guiders import CFGGuider
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
    from ltx_core.model.transformer import Modality, X0Model
    from ltx_core.model.video_vae import decode_video as vae_decode_video
    from ltx_core.text_encoders.gemma import encode_text
    from ltx_core.types import LatentState, VideoPixelShape
    from ltx_pipelines.utils import ModelLedger
    from ltx_pipelines.utils.types import PipelineComponents

    LTX2_AVAILABLE = True
except ImportError as e:
    LTX2_AVAILABLE = False
    LTX2_IMPORT_ERROR = str(e)

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.ltx_2.config import (
    get_default_ltx2_params,
    resolve_ltx2_model_paths,
    validate_ltx2_dimensions,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = logging.getLogger(__name__)

# LTX-2 constants (from ltx_pipelines.utils.constants)
VIDEO_LATENT_CHANNELS = 128
VIDEO_SCALE_FACTORS = {"time": 8, "width": 32, "height": 32}
AUDIO_LATENT_CHANNELS = 64
AUDIO_SAMPLE_RATE = 24000


def ensure_ltx2_deps():
    """Ensure LTX-2 dependencies are available, raise clear error if not."""
    if not LTX2_AVAILABLE:
        raise ImportError(
            f"LTX-2 dependencies not available: {LTX2_IMPORT_ERROR}\n\n"
            f"To install LTX-2 dependencies:\n"
            f"  cd /home/doug/codebase/vllm-omni/references/LTX-2/packages/ltx-core\n"
            f"  pip install -e .\n"
            f"  cd ../ltx-pipelines\n"
            f"  pip install -e .\n\n"
            f"Or if available on PyPI:\n"
            f"  pip install ltx-core ltx-pipelines"
        )


def get_ltx2_post_process_func(od_config: OmniDiffusionConfig):
    """Returns function to convert decoded video tensors to final output format.

    Args:
        od_config: Omni diffusion configuration

    Returns:
        Post-processing function that takes video tensor and returns processed output
    """
    from diffusers.video_processor import VideoProcessor

    # LTX-2 uses 32x downsampling in spatial dimensions
    video_processor = VideoProcessor(vae_scale_factor=32)

    def post_process_func(
        video: torch.Tensor,
        output_type: str = "np",
    ):
        """Post-process video frames.

        Args:
            video: Decoded video tensor
                - From vae_decode_video: [f, h, w, c] uint8 (already converted)
                - From latent decode: [B, C, T, H, W] float (needs conversion)
            output_type: "np" for numpy array, "latent" for raw tensor

        Returns:
            Processed video frames (numpy array or list of PIL images)

        Note:
            LTX-2's vae_decode_video returns uint8 tensors in [f, h, w, c] format,
            already converted to [0, 255] range. No further processing needed.
        """
        if output_type == "latent":
            return video

        # Check if video is already uint8 in [f, h, w, c] format (from vae_decode_video)
        if video.dtype == torch.uint8 and video.ndim == 4:
            # Already converted by vae_decode_video, just convert to numpy
            if output_type == "np":
                return video.cpu().numpy()
            else:
                # For PIL, convert frames to list of PIL images
                import PIL.Image
                frames = video.cpu().numpy()
                return [PIL.Image.fromarray(frame) for frame in frames]

        # Otherwise, use VideoProcessor for float tensors
        return video_processor.postprocess_video(video, output_type=output_type)

    return post_process_func


class LTX2Pipeline(nn.Module):
    """LTX-2 audio-video generation pipeline.

    This pipeline implements single-stage text/image-to-video generation
    with audio synthesis using the LTX-2 model architecture.

    Architecture:
        - Gemma text encoder for prompt processing
        - 19B parameter transformer for diffusion
        - Video VAE for encoding/decoding video latents
        - Audio VAE + vocoder for audio generation
        - Euler sampling with CFG guidance

    Note:
        This is a PoC implementation. The denoising loop is stubbed and
        returns placeholder outputs. Full runtime implementation will be
        added in the next phase.
    """

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        """Initialize LTX-2 pipeline.

        Args:
            od_config: Omni diffusion configuration
            prefix: Weight prefix for loading (unused)

        Raises:
            ImportError: If LTX-2 dependencies are not available
        """
        super().__init__()

        # Check dependencies
        ensure_ltx2_deps()

        self.od_config = od_config
        self.device = get_local_device()
        self.dtype = torch.bfloat16

        # Get default parameters
        self.default_params = get_default_ltx2_params()

        # Resolve model paths
        checkpoint_path = self._resolve_checkpoint_path(od_config.model)
        gemma_root = self._resolve_gemma_path(od_config.model)

        logger.info(f"LTX-2 checkpoint: {checkpoint_path}")
        logger.info(f"Gemma root: {gemma_root}")

        # Initialize ModelLedger (video-only, no audio)
        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=self.device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            loras=[],  # No LoRA support in Phase 2
            fp8transformer=False,  # No FP8 optimization in Phase 2
            spatial_upsampler_path=None,  # No upsampler in Phase 2 (single-stage)
        )

        logger.info("ModelLedger initialized (video-only mode)")

        # Initialize pipeline components for distilled generation
        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=self.device,
        )

        # Initialize diffusion components
        self.noiser = GaussianNoiser(generator=None)  # Generator set per-request
        self.stepper = EulerDiffusionStep()

        # Distilled sigma schedule (predefined for fast inference)
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES
        self.distilled_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(
            dtype=torch.float32, device=self.device
        )

        logger.info(
            f"Pipeline components initialized. "
            f"Using distilled sigma schedule with {len(self.distilled_sigmas)} steps (Phase 2: video-only)"
        )

    @property
    def vae(self):
        """Compatibility property for registry VAE configuration checks.

        LTX2Pipeline uses ModelLedger for dynamic component loading,
        so we return None to skip VAE configuration in the registry.
        VAE memory settings are configured directly in decode_video()
        when the decoder is loaded from ModelLedger.
        """
        return None

    def _resolve_checkpoint_path(self, model_path: str) -> str:
        """Resolve path to LTX-2 checkpoint file."""
        import os

        # If it's a HuggingFace repo ID, try to find it in cache first
        if "/" in model_path and not os.path.exists(model_path):
            # Try common HF cache locations
            cache_candidates = []

            # Check HUGGINGFACE_HUB_CACHE env var
            hf_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
            if hf_cache:
                # New format: models--namespace--model
                cache_candidates.append(os.path.join(hf_cache, f"models--{model_path.replace('/', '--')}"))
                # Old format: namespace_model (with underscore)
                cache_candidates.append(os.path.join(hf_cache, model_path.replace('/', '_')))

            # Check HF_HOME env var
            hf_home = os.environ.get("HF_HOME")
            if hf_home:
                cache_candidates.append(os.path.join(hf_home, "hub", f"models--{model_path.replace('/', '--')}"))
                cache_candidates.append(os.path.join(hf_home, "hub", model_path.replace('/', '_')))

            # Default HF cache location
            home = os.path.expanduser("~")
            cache_candidates.append(os.path.join(home, ".cache", "huggingface", "hub", f"models--{model_path.replace('/', '--')}"))
            cache_candidates.append(os.path.join(home, ".cache", "huggingface", "hub", model_path.replace('/', '_')))

            # Try to find existing cache
            checkpoint_candidates = [
                "ltx-2-19b-distilled.safetensors",
                "checkpoint.safetensors",
                "ltx-video-2b-v1.0.safetensors",
                "model.safetensors",
            ]

            for cache_path in cache_candidates:
                if os.path.isdir(cache_path):
                    # For new format, check if it has snapshots
                    snapshots = os.path.join(cache_path, "snapshots")
                    if os.path.isdir(snapshots):
                        # Get latest snapshot
                        snapshot_dirs = [d for d in os.listdir(snapshots) if os.path.isdir(os.path.join(snapshots, d))]
                        if snapshot_dirs:
                            candidate_path = os.path.join(snapshots, snapshot_dirs[0])
                            # Verify it has checkpoint files
                            has_checkpoint = any(os.path.exists(os.path.join(candidate_path, ckpt)) for ckpt in checkpoint_candidates)
                            if has_checkpoint:
                                model_path = candidate_path
                                logger.info(f"Found cached model at: {model_path}")
                                break
                            else:
                                logger.debug(f"Skipping {candidate_path} - no checkpoint files found")
                    else:
                        # Old format - verify it has checkpoint files
                        has_checkpoint = any(os.path.exists(os.path.join(cache_path, ckpt)) for ckpt in checkpoint_candidates)
                        if has_checkpoint:
                            model_path = cache_path
                            logger.info(f"Found cached model at: {model_path}")
                            break
                        else:
                            logger.debug(f"Skipping {cache_path} - no checkpoint files found")
            else:
                # Not found in cache, try to download
                logger.info(f"Model not found in cache, attempting download: {model_path}")
                from huggingface_hub import snapshot_download
                try:
                    local_dir = snapshot_download(model_path, allow_patterns=["*.safetensors", "*.json", "gemma/**"])
                    model_path = local_dir
                except Exception as e:
                    raise FileNotFoundError(
                        f"Failed to download model from HuggingFace: {model_path}. Error: {e}"
                    ) from e

        # Check if model_path is a directory
        if os.path.isdir(model_path):
            # Look for common checkpoint filenames
            candidates = [
                "ltx-2-19b-distilled.safetensors",  # Distilled checkpoint
                "checkpoint.safetensors",
                "ltx-video-2b-v1.0.safetensors",
                "model.safetensors",
            ]
            for candidate in candidates:
                checkpoint = os.path.join(model_path, candidate)
                if os.path.exists(checkpoint):
                    logger.info(f"Found checkpoint: {checkpoint}")
                    return checkpoint
            raise FileNotFoundError(
                f"No LTX-2 checkpoint found in {model_path}. "
                f"Looked for: {', '.join(candidates)}"
            )
        elif os.path.isfile(model_path):
            return model_path
        else:
            raise FileNotFoundError(f"Model path not found: {model_path}")

    def _resolve_gemma_path(self, model_path: str) -> str:
        """Resolve path to Gemma text encoder."""
        import os

        # If it's a HuggingFace repo ID, try to find it in cache first
        if "/" in model_path and not os.path.exists(model_path):
            # Try common HF cache locations
            cache_candidates = []

            # Check HUGGINGFACE_HUB_CACHE env var
            hf_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
            if hf_cache:
                cache_candidates.append(os.path.join(hf_cache, f"models--{model_path.replace('/', '--')}"))
                cache_candidates.append(os.path.join(hf_cache, model_path.replace('/', '_')))

            # Check HF_HOME env var
            hf_home = os.environ.get("HF_HOME")
            if hf_home:
                cache_candidates.append(os.path.join(hf_home, "hub", f"models--{model_path.replace('/', '--')}"))
                cache_candidates.append(os.path.join(hf_home, "hub", model_path.replace('/', '_')))

            # Default HF cache location
            home = os.path.expanduser("~")
            cache_candidates.append(os.path.join(home, ".cache", "huggingface", "hub", f"models--{model_path.replace('/', '--')}"))
            cache_candidates.append(os.path.join(home, ".cache", "huggingface", "hub", model_path.replace('/', '_')))

            # Try to find existing cache
            encoder_dirs = ["text_encoder", "gemma"]

            for cache_path in cache_candidates:
                if os.path.isdir(cache_path):
                    # For new format, check if it has snapshots
                    snapshots = os.path.join(cache_path, "snapshots")
                    if os.path.isdir(snapshots):
                        # Get latest snapshot
                        snapshot_dirs = [d for d in os.listdir(snapshots) if os.path.isdir(os.path.join(snapshots, d))]
                        if snapshot_dirs:
                            candidate_path = os.path.join(snapshots, snapshot_dirs[0])
                            # Verify it has text encoder directory
                            has_encoder = any(os.path.isdir(os.path.join(candidate_path, enc)) for enc in encoder_dirs)
                            if has_encoder:
                                model_path = candidate_path
                                logger.info(f"Found cached model for Gemma at: {model_path}")
                                break
                            else:
                                logger.debug(f"Skipping {candidate_path} - no text encoder directory found")
                    else:
                        # Old format - verify it has text encoder directory
                        has_encoder = any(os.path.isdir(os.path.join(cache_path, enc)) for enc in encoder_dirs)
                        if has_encoder:
                            model_path = cache_path
                            logger.info(f"Found cached model for Gemma at: {model_path}")
                            break
                        else:
                            logger.debug(f"Skipping {cache_path} - no text encoder directory found")
            else:
                # Not found in cache, try to download
                logger.info(f"Model not found in cache for Gemma, attempting download: {model_path}")
                from huggingface_hub import snapshot_download
                try:
                    local_dir = snapshot_download(model_path, allow_patterns=["*.safetensors", "*.json", "gemma/**"])
                    model_path = local_dir
                except Exception as e:
                    raise FileNotFoundError(
                        f"Failed to download model from HuggingFace: {model_path}. Error: {e}"
                    ) from e

        # If model_path is a directory, look for text encoder subdirectory
        if os.path.isdir(model_path):
            # Try both "text_encoder" and "gemma" (different model versions use different names)
            for encoder_dir in ["text_encoder", "gemma"]:
                encoder_path = os.path.join(model_path, encoder_dir)
                if os.path.exists(encoder_path):
                    logger.info(f"Found text encoder at: {encoder_path}")
                    # Return the parent directory as gemma_root (it should contain both text_encoder/ and tokenizer/)
                    return model_path

        # Fall back to model_path itself or raise error
        raise FileNotFoundError(
            f"Gemma text encoder not found. Expected at {model_path}/text_encoder/ or {model_path}/gemma/"
        )

    def forward(
        self,
        req: OmniDiffusionRequest,
    ) -> DiffusionOutput:
        """Execute video generation forward pass.

        Args:
            req: Omni diffusion request containing generation parameters

        Returns:
            DiffusionOutput containing generated video frames

        Raises:
            ValueError: If enable_audio=True (not supported in Phase 2)
        """
        # Phase 2: Enforce video-only
        if req.enable_audio:
            raise ValueError(
                "Audio generation not supported in Phase 2. "
                "Set enable_audio=False for video-only generation."
            )

        # Extract and validate parameters
        prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
        height = req.height if req.height is not None else self.default_params["height"]
        width = req.width if req.width is not None else self.default_params["width"]
        num_frames = (
            req.num_frames if isinstance(req.num_frames, int)
            else req.num_frames[0] if isinstance(req.num_frames, list)
            else self.default_params["num_frames"]
        )

        # Validate dimensions
        validate_ltx2_dimensions(height, width, num_frames)

        batch_size = req.batch_size
        generator = req.generator

        logger.info(
            f"LTX-2 distilled generation: '{prompt[:50]}...', "
            f"{width}x{height}, {num_frames} frames, "
            f"8 denoising steps (distilled)"
        )

        # Step 1: Encode text
        logger.info("Step 1/5: Encoding text prompt...")
        video_context, audio_context = self.encode_text(prompt)

        # Step 2: Prepare latents
        logger.info("Step 2/5: Preparing latents...")
        video_latents, audio_latents = self.prepare_latents(
            batch_size=batch_size,
            num_frames=num_frames,
            height=height,
            width=width,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
            fps=req.fps if hasattr(req, 'fps') and req.fps else 24.0,
        )

        # Step 3: Create and noise latent states using tools
        logger.info("Step 3/5: Creating and noising latent states...")
        from ltx_core.types import LatentState

        # Create properly structured video state using tools
        video_state = self.video_tools.create_initial_state(
            device=self.device,
            dtype=self.dtype,
            initial_latent=video_latents,
        )

        # Create properly structured audio state using tools
        audio_state = self.audio_tools.create_initial_state(
            device=self.device,
            dtype=self.dtype,
            initial_latent=audio_latents,
        )

        # Add noise using GaussianNoiser (callable)
        video_state = self.noiser(video_state, noise_scale=self.distilled_sigmas[0].item())
        # Audio state unused in video-only mode

        # Step 4: Denoising loop
        logger.info("Step 4/5: Running denoising loop (8 steps)...")
        video_state, _ = self.denoise_loop(
            video_state=video_state,
            audio_state=audio_state,
            video_context=video_context,
            audio_context=audio_context,
            sigmas=self.distilled_sigmas,
        )

        # Clear conditioning before unpatchifying (matches upstream pattern)
        video_state = self.video_tools.clear_conditioning(video_state)

        # Unpatchify the video state to get back to (B, C, T, H, W) format
        video_state = self.video_tools.unpatchify(video_state)
        denoised_video_latents = video_state.latent

        # Debug: Check latent statistics
        logger.info(
            f"DEBUG: Denoised latents - shape={denoised_video_latents.shape}, "
            f"min={denoised_video_latents.min().item():.4f}, "
            f"max={denoised_video_latents.max().item():.4f}, "
            f"mean={denoised_video_latents.mean().item():.4f}"
        )

        # Step 5: Decode video
        logger.info("Step 5/5: Decoding video...")
        decoded_video = self.decode_video(denoised_video_latents)

        # Debug: Check decoded video statistics
        logger.info(
            f"DEBUG: Decoded video - shape={decoded_video.shape}, "
            f"dtype={decoded_video.dtype}, "
            f"min={decoded_video.min().item()}, "
            f"max={decoded_video.max().item()}, "
            f"mean={decoded_video.float().mean().item():.4f}"
        )

        # Store in request output field
        req.output = decoded_video

        logger.info("✓ Generation complete!")

        # Return DiffusionOutput
        return DiffusionOutput(
            output=decoded_video,
        )

    def encode_text(
        self,
        prompt: str,
        negative_prompt: str | None = None,
    ):
        """Encode text prompts using Gemma text encoder.

        Args:
            prompt: Positive text prompt
            negative_prompt: Negative prompt (unused in Phase 2 - no CFG)

        Returns:
            Tuple of (video_context, audio_context)
            Note: audio_context is returned but not used in video-only generation

        Note:
            Phase 2 implementation does NOT support CFG (no negative prompt).
            Negative prompt support will be added in Phase 3.
        """
        # Load text encoder
        text_encoder = self.model_ledger.text_encoder()

        try:
            # Encode prompt (returns list of (v_context, a_context) tuples)
            contexts = encode_text(text_encoder, prompts=[prompt])
            video_context, audio_context = contexts[0]

            logger.debug(
                f"Text encoded: video_context shape={video_context.shape}, "
                f"audio_context shape={audio_context.shape}"
            )

            return video_context, audio_context

        finally:
            # Clean up text encoder immediately to save memory
            torch.cuda.synchronize()
            del text_encoder
            from ltx_pipelines.utils.helpers import cleanup_memory
            cleanup_memory()

    def prepare_latents(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
        fps: float = 24.0,
    ):
        """Initialize random latents for video and audio using VideoLatentTools.

        Args:
            batch_size: Batch size
            num_frames: Number of video frames
            height: Video height (pixels)
            width: Video width (pixels)
            device: Torch device
            dtype: Torch dtype
            generator: Random generator for reproducibility
            fps: Frames per second

        Returns:
            Tuple of (video_latents, audio_latents)
            Note: audio_latents is returned but not used in video-only generation
        """
        from ltx_core.tools import VideoLatentTools, AudioLatentTools
        from ltx_core.types import VideoPixelShape, VideoLatentShape, AudioLatentShape, SpatioTemporalScaleFactors

        # Create video pixel shape
        video_pixel_shape = VideoPixelShape(
            batch=batch_size,
            frames=num_frames,
            height=height,
            width=width,
            fps=fps,
        )

        # Get default scale factors (8x temporal, 32x spatial compression)
        scale_factors = SpatioTemporalScaleFactors.default()

        # Create video latent shape
        video_latent_shape = VideoLatentShape.from_pixel_shape(
            shape=video_pixel_shape,
            latent_channels=VIDEO_LATENT_CHANNELS,
            scale_factors=scale_factors,
        )

        # Create VideoLatentTools for proper initialization
        video_tools = VideoLatentTools(
            patchifier=self.pipeline_components.video_patchifier,
            target_shape=video_latent_shape,
            fps=fps,
            scale_factors=scale_factors,
        )

        # For text-to-video, we don't pre-initialize latents
        # VideoLatentTools.create_initial_state will create zeros
        # which will then be noised by GaussianNoiser
        # (This is different from img2vid where initial_latent would be encoded image)
        video_latents = None

        # Create audio latent shape
        audio_latent_shape = AudioLatentShape.from_video_pixel_shape(video_pixel_shape)

        # Create AudioLatentTools
        audio_tools = AudioLatentTools(
            patchifier=self.pipeline_components.audio_patchifier,
            target_shape=audio_latent_shape,
        )

        # For video-only mode, audio latents are unused placeholders
        # create_initial_state will handle initialization
        audio_latents = None

        logger.debug(
            f"Latents initialized: video={video_latent_shape}, "
            f"audio={audio_latent_shape} (audio unused in video-only mode)"
        )

        # Store tools for use in denoise_loop
        self.video_tools = video_tools
        self.audio_tools = audio_tools

        return video_latents, audio_latents

    def denoise_loop(
        self,
        video_state: LatentState,
        audio_state: LatentState,
        video_context: torch.Tensor,
        audio_context: torch.Tensor,
        sigmas: torch.Tensor,
    ):
        """Execute denoising loop with distilled sigma schedule.

        Args:
            video_state: Initial noisy video latent state (patchified)
            audio_state: Initial noisy audio latent state (patchified, unused in video-only)
            video_context: Text conditioning for video
            audio_context: Text conditioning for audio (unused in video-only)
            sigmas: Predefined sigma schedule (DISTILLED_SIGMA_VALUES)

        Returns:
            Tuple of (denoised_video_state, denoised_audio_state)

        Note:
            Phase 2 implementation:
            - Uses simple denoising (no CFG)
            - Video-only (audio modality disabled)
            - Distilled sigma schedule (8 steps)
        """
        from ltx_pipelines.utils.helpers import post_process_latent
        from ltx_core.types import LatentState
        from dataclasses import replace
        from tqdm import tqdm

        # Load transformer
        transformer = self.model_ledger.transformer()

        try:

            # Denoising loop (iterate all sigmas except last)
            for step_idx in tqdm(range(len(sigmas) - 1), desc="Denoising"):
                sigma = sigmas[step_idx]

                # Create video modality input using the video_state directly
                from ltx_pipelines.utils.helpers import modality_from_latent_state
                video_modality = modality_from_latent_state(video_state, video_context, sigma)

                # Audio modality is None for video-only mode (Phase 2)
                audio_modality = None

                # Transformer forward pass
                denoised_video, denoised_audio = transformer(
                    video=video_modality,
                    audio=audio_modality,
                    perturbations=None,
                )

                # Post-process (handle masks and clean latents if present)
                denoised_video = post_process_latent(
                    denoised_video,
                    video_state.denoise_mask,
                    video_state.clean_latent,
                )

                # Euler step to advance noise level
                video_state = replace(
                    video_state,
                    latent=self.stepper.step(
                        video_state.latent, denoised_video, sigmas, step_idx
                    ),
                )

            return video_state, audio_state

        finally:
            # Clean up transformer to save memory
            torch.cuda.synchronize()
            del transformer
            from ltx_pipelines.utils.helpers import cleanup_memory
            cleanup_memory()

    def _create_video_modality(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        sigma: float,
        positions: torch.Tensor,
    ):
        """Create video modality input for transformer."""
        from ltx_pipelines.utils.helpers import modality_from_latent_state
        from ltx_core.types import LatentState

        # Create temporary LatentState for helper function
        temp_state = LatentState(
            latent=latent,
            denoise_mask=torch.ones_like(latent[:, :1]),
            positions=positions,
            clean_latent=torch.zeros_like(latent),
        )

        # Use upstream helper to create modality
        modality = modality_from_latent_state(temp_state, context, sigma)

        return modality

    def _create_audio_modality(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        sigma: float,
        positions: torch.Tensor,
        enabled: bool = False,
    ):
        """Create audio modality input for transformer (disabled in video-only mode)."""
        from ltx_pipelines.utils.helpers import modality_from_latent_state
        from ltx_core.types import LatentState

        # Create temporary LatentState
        temp_state = LatentState(
            latent=latent,
            denoise_mask=torch.ones_like(latent[:, :1]),
            positions=positions,
            clean_latent=torch.zeros_like(latent),
        )

        # Create modality but mark as disabled
        modality = modality_from_latent_state(temp_state, context, sigma)
        modality.enabled = enabled  # Set to False for video-only

        return modality

    def decode_video(
        self,
        video_latents: torch.Tensor,
    ):
        """Decode video latents to pixel space.

        Args:
            video_latents: Latent tensor from denoising loop (B, C, T, H, W)

        Returns:
            Decoded video tensor [f, h, w, c] uint8
        """
        # Load video decoder
        video_decoder = self.model_ledger.video_decoder()

        try:
            # Configure VAE memory optimization settings if available
            # These help reduce VRAM usage for large videos
            if hasattr(video_decoder, "use_slicing"):
                video_decoder.use_slicing = getattr(self.od_config, "vae_use_slicing", True)
                logger.debug(f"VAE slicing enabled: {video_decoder.use_slicing}")
            if hasattr(video_decoder, "use_tiling"):
                video_decoder.use_tiling = getattr(self.od_config, "vae_use_tiling", False)
                logger.debug(f"VAE tiling enabled: {video_decoder.use_tiling}")

            # === FIX 1: Convert to float32 for precision ===
            # Using float32 prevents color banding from bfloat16 precision loss
            video_latents = video_latents.to(torch.float32)

            # === DIAGNOSTIC: Latent Analysis Before Normalization ===
            logger.info("=" * 60)
            logger.info("DIAGNOSTIC: Latent Analysis Before VAE Decode")
            logger.info(f"  Pre-norm Shape: {video_latents.shape}")
            logger.info(f"  Pre-norm Dtype: {video_latents.dtype}")
            logger.info(f"  Pre-norm Min:   {video_latents.min().item():.6f}")
            logger.info(f"  Pre-norm Max:   {video_latents.max().item():.6f}")
            logger.info(f"  Pre-norm Mean:  {video_latents.mean().item():.6f}")
            logger.info(f"  Pre-norm Std:   {video_latents.std().item():.6f}")

            # Check per_channel_statistics status
            stats = video_decoder.per_channel_statistics
            mean_of_means = stats.get_buffer('mean-of-means')
            std_of_means = stats.get_buffer('std-of-means')
            logger.info(f"  Per-channel stats loaded: {mean_of_means.numel() > 0}")
            if mean_of_means.numel() > 0:
                logger.info(f"    mean-of-means: [{mean_of_means.min().item():.4f}, {mean_of_means.max().item():.4f}]")
                logger.info(f"    std-of-means:  [{std_of_means.min().item():.4f}, {std_of_means.max().item():.4f}]")

            # === FIX 2: Normalize latents to match VAE encoder's output space ===
            # The diffusion process outputs approximately normalized latents, but they
            # may not have the exact per-channel statistics that the VAE encoder produced.
            # Re-normalize to ensure they match what the VAE decoder expects.
            normalized_latents = video_decoder.per_channel_statistics.normalize(video_latents)

            logger.info(f"  Post-norm Min:  {normalized_latents.min().item():.6f}")
            logger.info(f"  Post-norm Max:  {normalized_latents.max().item():.6f}")
            logger.info(f"  Post-norm Mean: {normalized_latents.mean().item():.6f}")
            logger.info(f"  Post-norm Std:  {normalized_latents.std().item():.6f}")

            # === FIX 3: Convert back to bfloat16 to match decoder weights ===
            # The VAE decoder weights are in bfloat16, so we need to convert the latents
            # back to bfloat16 to avoid dtype mismatch errors
            normalized_latents = normalized_latents.to(torch.bfloat16)
            logger.info(f"  Final dtype for decode: {normalized_latents.dtype}")
            logger.info("=" * 60)

            # vae_decode_video expects latents in [B, C, T, H, W] format
            # (despite docstring saying [c, f, h, w], code does frames[0] to remove batch dim)
            logger.debug(f"Decoding normalized latents with shape: {normalized_latents.shape}")

            # vae_decode_video is a generator that yields decoded chunks
            # For non-tiled decoding, it yields exactly once with uint8 frames [f, h, w, c]
            # Internally it:
            # 1. Calls video_decoder(latent) with [B, C, T, H, W]
            # 2. video_decoder applies per_channel_statistics.un_normalize()
            # 3. Converts to uint8: (((x + 1) / 2).clamp(0, 1) * 255)
            # 4. Removes batch dim and rearranges: frames[0], "c f h w -> f h w c"
            decoded_video = next(vae_decode_video(normalized_latents, video_decoder))

            logger.debug(f"Video decoded: shape={decoded_video.shape}")

            return decoded_video

        finally:
            # Clean up decoder
            torch.cuda.synchronize()
            del video_decoder
            from ltx_pipelines.utils.helpers import cleanup_memory
            cleanup_memory()

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load model weights.

        Args:
            weights: Iterable of (name, tensor) weight pairs

        Returns:
            Set of loaded weight names

        Note:
            Stubbed in PoC. Full implementation will use AutoWeightsLoader
            pattern from vLLM-Omni and ModelLedger from LTX-2.
        """
        # TODO: Implement weight loading using AutoWeightsLoader
        logger.warning("load_weights called (stub) - no weights loaded in PoC")
        return set()

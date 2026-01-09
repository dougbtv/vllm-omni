# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Text-to-video generation example using LTX-2.

This example demonstrates how to use the LTX-2 (Lightricks Video-2) pipeline
for generating videos with audio from text prompts.

Note:
    This is a PoC example. The pipeline returns placeholder outputs until
    the full runtime implementation is completed.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.utils.platform_utils import detect_device_type


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a video with audio using LTX-2."
    )

    # Model configuration
    parser.add_argument(
        "--model",
        default="Lightricks/LTX-Video",
        help=(
            "Path to LTX-2 model directory containing:\n"
            "  - checkpoint.safetensors (or ltx-video-2b-v1.0.safetensors)\n"
            "  - gemma/ (text encoder directory)\n"
            "Example: /path/to/LTX-Video-2B-v1.0/"
        ),
    )

    # Generation parameters
    parser.add_argument(
        "--prompt",
        default="A dragon breathing fire in a fantasy landscape at sunset",
        help="Text prompt for video generation",
    )
    parser.add_argument(
        "--negative_prompt",
        default="blurry, low quality, distorted",
        help="Negative prompt for guidance",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Dimension parameters
    parser.add_argument(
        "--height",
        type=int,
        default=384,
        help="Video height (must be divisible by 32)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Video width (must be divisible by 32)",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=49,
        help="Number of frames (must be 8K+1, e.g., 49 = 8*6+1)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Frames per second for output video",
    )

    # Sampling parameters
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=40,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.0,
        help="CFG guidance scale",
    )

    # Audio parameters
    parser.add_argument(
        "--enable_audio",
        action="store_true",
        default=False,
        help="Generate audio track (NOT SUPPORTED in Phase 2)",
    )
    parser.add_argument(
        "--no_audio",
        action="store_false",
        dest="enable_audio",
        help="Disable audio generation",
    )

    # Output parameters
    parser.add_argument(
        "--output",
        type=str,
        default="ltx2_output.mp4",
        help="Path to save the generated video",
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Detect device
    device = detect_device_type()
    print(f"Using device: {device}")

    # Create random generator
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Validate audio setting for Phase 2
    if args.enable_audio:
        print("\n⚠️  WARNING: Audio generation not supported in Phase 2")
        print("   Set --no_audio to suppress this warning")
        print("   Disabling audio for this run\n")
        args.enable_audio = False

    print("\n" + "=" * 70)
    print("LTX-2 Video Generation (Phase 2: Distilled, Video-Only)")
    print("=" * 70)
    print(f"Prompt: {args.prompt}")
    print(f"Dimensions: {args.width}x{args.height}, {args.num_frames} frames @ {args.fps} fps")
    print(f"Mode: Distilled (8 steps, video-only)")
    print("=" * 70 + "\n")

    # Initialize Omni with LTX-2 model
    print("Initializing LTX-2 pipeline...")
    try:
        omni = Omni(
            model=args.model,
            # Note: Additional config options can be passed here:
            # vae_use_slicing=True,  # For memory optimization
            # vae_use_tiling=True,   # For large resolutions
        )
        print("✓ Pipeline initialized successfully\n")
    except ImportError as e:
        print(f"\n❌ Error: {e}\n")
        print("To install LTX-2 dependencies:")
        print("  cd /home/doug/codebase/vllm-omni/references/LTX-2/packages/ltx-core")
        print("  pip install -e .")
        print("  cd ../ltx-pipelines")
        print("  pip install -e .\n")
        return
    except Exception as e:
        print(f"\n❌ Error initializing pipeline: {e}\n")
        return

    # Generate video
    print("Generating video...\n")

    try:
        frames = omni.generate(
            args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            fps=args.fps,
            generator=generator,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            enable_audio=args.enable_audio,
            num_outputs_per_prompt=1,
        )

        print("✓ Generation completed\n")

    except Exception as e:
        print(f"\n❌ Error during generation: {e}\n")
        import traceback
        traceback.print_exc()
        return

    # Extract video frames from output
    print("Processing output...")
    if isinstance(frames, list) and len(frames) > 0:
        first_item = frames[0]

        # Check if it's an OmniRequestOutput
        if hasattr(first_item, "final_output_type"):
            if first_item.final_output_type != "image":
                print(
                    f"⚠️  Unexpected output type '{first_item.final_output_type}', "
                    f"expected 'image' for video generation."
                )

            # Pipeline mode: extract from nested request_output
            if hasattr(first_item, "is_pipeline_output") and first_item.is_pipeline_output:
                if isinstance(first_item.request_output, list) and len(first_item.request_output) > 0:
                    inner_output = first_item.request_output[0]
                    if isinstance(inner_output, OmniRequestOutput) and hasattr(inner_output, "images"):
                        frames = inner_output.images[0] if inner_output.images else None
                        if frames is None:
                            print("❌ No video frames found in output.")
                            return
            # Diffusion mode: use direct images field
            elif hasattr(first_item, "images") and first_item.images:
                frames = first_item.images
            else:
                print("❌ No video frames found in OmniRequestOutput.")
                return

    # Save video
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from diffusers.utils import export_to_video
    except ImportError:
        print("❌ diffusers is required for export_to_video.")
        print("Install with: pip install diffusers")
        return

    # Process frames tensor
    if isinstance(frames, torch.Tensor):
        video_tensor = frames.detach().cpu()

        # Handle different tensor shapes
        if video_tensor.dim() == 5:
            # [B, C, F, H, W] or [B, F, H, W, C]
            if video_tensor.shape[1] in (3, 4):
                # [B, C, F, H, W] -> [F, H, W, C]
                video_tensor = video_tensor[0].permute(1, 2, 3, 0)
            else:
                # [B, F, H, W, C] -> [F, H, W, C]
                video_tensor = video_tensor[0]
        elif video_tensor.dim() == 4 and video_tensor.shape[0] in (3, 4):
            # [C, F, H, W] -> [F, H, W, C]
            video_tensor = video_tensor.permute(1, 2, 3, 0)

        # Normalize to [0, 1] if float
        if video_tensor.is_floating_point():
            video_tensor = video_tensor.clamp(-1, 1) * 0.5 + 0.5

        video_array = video_tensor.float().numpy()
    else:
        video_array = frames
        if hasattr(video_array, "shape") and video_array.ndim == 5:
            video_array = video_array[0]

    # Convert to list of frames for export_to_video
    if isinstance(video_array, np.ndarray) and video_array.ndim == 4:
        video_array = list(video_array)

    # Export video
    export_to_video(video_array, str(output_path), fps=args.fps)
    print(f"\n✓ Video saved to: {output_path}")

    print("\n" + "=" * 70)
    print("Generation complete!")
    print("=" * 70)
    print("\nPhase 2 Implementation Notes:")
    print("- Video-only generation (distilled, 8-step denoising)")
    print("- Audio support deferred to Phase 3")
    print("- CFG guidance deferred to Phase 4")
    print("- Two-stage upsampling deferred to Phase 5")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

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
    print(f"Output type: {type(frames)}")
    print(f"Output length: {len(frames) if isinstance(frames, list) else 'N/A'}")

    if isinstance(frames, list) and len(frames) > 0:
        first_item = frames[0]
        print(f"First item type: {type(first_item)}")
        print(f"First item has .images: {hasattr(first_item, 'images')}")

        # Check if it's an OmniRequestOutput
        if hasattr(first_item, "final_output_type"):
            print(f"Output type field: {first_item.final_output_type}")
            if first_item.final_output_type != "image":
                print(
                    f"⚠️  Unexpected output type '{first_item.final_output_type}', "
                    f"expected 'image' for video generation."
                )

            # Pipeline mode: extract from nested request_output
            if hasattr(first_item, "is_pipeline_output") and first_item.is_pipeline_output:
                print("Pipeline output mode detected")
                if isinstance(first_item.request_output, list) and len(first_item.request_output) > 0:
                    inner_output = first_item.request_output[0]
                    print(f"Inner output type: {type(inner_output)}")

                    # Handle both OmniRequestOutput and SimpleNamespace
                    if isinstance(inner_output, OmniRequestOutput):
                        # Try to extract the actual tensor from images field
                        if hasattr(inner_output, "images") and inner_output.images:
                            frames = inner_output.images
                            print(f"Extracted frames from inner.images: {type(frames)}")
                            # If it's a single-element list, unwrap it
                            if isinstance(frames, list) and len(frames) == 1:
                                frames = frames[0]
                                print(f"Unwrapped to: {type(frames)}")
                        else:
                            print("❌ No video frames found in inner output.")
                            print(f"Available attributes: {[a for a in dir(inner_output) if not a.startswith('_')]}")
                            return
                    else:
                        # Handle SimpleNamespace or other types
                        print("Inner output is not OmniRequestOutput, trying attribute extraction...")
                        print(f"Available attributes: {[a for a in dir(inner_output) if not a.startswith('_')]}")

                        # Try common attribute names
                        if hasattr(inner_output, "output"):
                            frames = inner_output.output
                            print(f"Extracted frames from inner.output: {type(frames)}")
                        elif hasattr(inner_output, "images"):
                            frames = inner_output.images
                            print(f"Extracted frames from inner.images: {type(frames)}")
                        else:
                            print("❌ Could not find video frames in inner output.")
                            return

                        # If it's a single-element list, unwrap it
                        if isinstance(frames, list) and len(frames) == 1:
                            frames = frames[0]
                            print(f"Unwrapped to: {type(frames)}")
            # Diffusion mode: use direct images field
            elif hasattr(first_item, "images") and first_item.images is not None:
                print("Diffusion output mode detected")
                print(f"first_item.images type: {type(first_item.images)}")
                frames = first_item.images
                # If it's a single-element list, unwrap it
                if isinstance(frames, list) and len(frames) == 1:
                    frames = frames[0]
                    print(f"Unwrapped to: {type(frames)}")
            else:
                print("❌ No video frames found in OmniRequestOutput.")
                print(f"Available attributes: {[a for a in dir(first_item) if not a.startswith('_')]}")
                return

    # Save video
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Import PyAV for Chrome-compatible video encoding
    try:
        import av
    except ImportError:
        print("❌ PyAV is required for video export.")
        print("Install with: pip install av")
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

    # Ensure video_array is numpy array in [F, H, W, C] format
    if isinstance(video_array, list):
        video_array = np.array(video_array)

    if video_array.ndim != 4:
        print(f"❌ Unexpected video array shape: {video_array.shape}")
        return

    num_frames, height, width, channels = video_array.shape

    # Ensure uint8 format
    if video_array.dtype != np.uint8:
        video_array = np.clip(video_array * 255, 0, 255).astype(np.uint8)

    # Export video using PyAV with Chrome-compatible settings
    print(f"\nExporting video: {num_frames} frames at {width}x{height} @ {args.fps} fps")

    container = av.open(str(output_path), mode="w")
    try:
        # Create video stream with yuv420p for Chrome compatibility
        stream = container.add_stream("libx264", rate=int(args.fps))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"  # Critical for Chrome playback

        # Encode frames
        for i in range(num_frames):
            frame_array = video_array[i]
            # Create AV frame from numpy array (RGB format)
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")

            # Encode and mux
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush encoder
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()

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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example client for vLLM-Omni Image Generation API.

This script demonstrates how to use the OpenAI-compatible text-to-image
generation API to create images from text prompts.

Usage:
    python client.py \\
        --prompt "a cat on a laptop" \\
        --output cat.png

    python client.py \\
        --prompt "a dragon in the sky" \\
        --output dragon.png \\
        --seed 42 \\
        --steps 50 \\
        --n 2
"""

import argparse
import base64
import io
from pathlib import Path

import requests
from PIL import Image


def generate_image(
    prompt: str,
    api_base: str = "http://localhost:8000",
    model: str = "Qwen/Qwen-Image",
    n: int = 1,
    size: str = "1024x1024",
    seed: int = None,
    negative_prompt: str = None,
    num_inference_steps: int = 50,
    true_cfg_scale: float = 4.0,
) -> list[Image.Image]:
    """
    Generate images using the vLLM-Omni API.

    Args:
        prompt: Text description of desired image(s)
        api_base: Base URL of the API server
        model: Model name to use
        n: Number of images to generate
        size: Image size in "WIDTHxHEIGHT" format
        seed: Random seed for reproducibility (optional)
        negative_prompt: Text describing what to avoid (optional)
        num_inference_steps: Number of diffusion steps
        true_cfg_scale: CFG scale for Qwen-Image

    Returns:
        List of PIL Image objects
    """
    url = f"{api_base}/v1/images/generations"

    # Build request payload
    payload = {
        "model": model,
        "prompt": prompt,
        "n": n,
        "size": size,
        "response_format": "b64_json",
    }

    # Add optional parameters
    if seed is not None:
        payload["seed"] = seed
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    if num_inference_steps:
        payload["num_inference_steps"] = num_inference_steps
    if true_cfg_scale:
        payload["true_cfg_scale"] = true_cfg_scale

    print(f"Generating {n} image(s) for prompt: '{prompt}'")
    print(f"Parameters: size={size}, steps={num_inference_steps}, cfg={true_cfg_scale}, seed={seed}")

    # Send request
    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()

    result = response.json()

    # Decode base64 images to PIL Images
    images = []
    for img_data in result["data"]:
        img_bytes = base64.b64decode(img_data["b64_json"])
        img = Image.open(io.BytesIO(img_bytes))
        images.append(img)

    print(f"Successfully generated {len(images)} image(s)")
    return images


def main():
    parser = argparse.ArgumentParser(
        description="Generate images via vLLM-Omni API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a single image
  python client.py --prompt "a cat on a laptop" --output cat.png

  # Generate multiple images with custom parameters
  python client.py \\
    --prompt "a dragon flying over mountains" \\
    --output dragon.png \\
    --n 3 \\
    --seed 42 \\
    --steps 100 \\
    --size 1024x1024

  # Use negative prompt to avoid unwanted elements
  python client.py \\
    --prompt "beautiful landscape" \\
    --negative-prompt "blurry, low quality, distorted" \\
    --output landscape.png
""",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text description of the desired image(s)",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen-Image",
        help="Model name (default: Qwen/Qwen-Image)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output file path (default: output.png)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of images to generate (default: 1)",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1024x1024",
        help="Image size in WIDTHxHEIGHT format (default: 1024x1024)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: random)",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Text describing what to avoid in the image (default: none)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of diffusion steps (default: 50)",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=4.0,
        help="CFG scale (default: 4.0)",
    )

    args = parser.parse_args()

    # Generate images
    try:
        images = generate_image(
            prompt=args.prompt,
            api_base=args.api_base,
            model=args.model,
            n=args.n,
            size=args.size,
            seed=args.seed,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            true_cfg_scale=args.cfg_scale,
        )
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to connect to API server: {e}")
        print(f"Make sure the server is running at {args.api_base}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

    # Save images
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.n == 1:
        images[0].save(output_path)
        print(f"✓ Saved image to: {output_path}")
    else:
        # Save multiple images with numbered filenames
        stem = output_path.stem
        suffix = output_path.suffix or ".png"
        for i, img in enumerate(images):
            save_path = output_path.parent / f"{stem}_{i}{suffix}"
            img.save(save_path)
            print(f"✓ Saved image {i+1}/{len(images)} to: {save_path}")

    return 0


if __name__ == "__main__":
    exit(main())

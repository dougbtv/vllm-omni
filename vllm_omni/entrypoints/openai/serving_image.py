# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CLI entry point for vLLM-Omni image generation server.

This module provides a command-line interface to start the OpenAI-compatible
text-to-image generation API server.

Usage:
    python -m vllm_omni.entrypoints.openai.serving_image \\
        --model Qwen/Qwen-Image \\
        --host 0.0.0.0 \\
        --port 8000
"""

import argparse
import sys

import uvicorn

from vllm.logger import init_logger
from vllm_omni.entrypoints.openai.image_model_profiles import (
    get_model_profile,
    list_supported_models,
)
from vllm_omni.entrypoints.openai.image_server import create_app

logger = init_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="vLLM-Omni Image Generation API Server (OpenAI DALL-E compatible)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with default settings
  python -m vllm_omni.entrypoints.openai.serving_image

  # Specify custom model and port
  python -m vllm_omni.entrypoints.openai.serving_image \\
    --model Qwen/Qwen-Image \\
    --port 8080

  # Development mode with auto-reload
  python -m vllm_omni.entrypoints.openai.serving_image \\
    --reload \\
    --log-level debug
""",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen-Image",
        help="Diffusion model name (e.g., 'Qwen/Qwen-Image', 'Tongyi-MAI/Z-Image-Turbo')",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number to bind the server to (default: 8000)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level (default: info)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development (default: False)",
    )

    return parser.parse_args()


def main():
    """Main entry point for the image generation server."""
    args = parse_args()

    # Validate model early before starting the server
    try:
        profile = get_model_profile(args.model)
    except ValueError as e:
        logger.error(f"Error: {e}")
        logger.error(f"Supported models: {', '.join(list_supported_models())}")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("vLLM-Omni Image Generation Server")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(
        f"Inference Steps: {profile.default_num_inference_steps} "
        f"(max {profile.max_num_inference_steps})"
    )
    logger.info(f"Server URL: http://{args.host}:{args.port}")
    logger.info(f"API Endpoint: http://{args.host}:{args.port}/v1/images/generations")
    logger.info(f"Health Check: http://{args.host}:{args.port}/health")
    logger.info(f"Log Level: {args.log_level}")
    if args.reload:
        logger.warning("Auto-reload enabled (development mode)")
    logger.info("=" * 70)

    # Create FastAPI app with specified model
    app = create_app(model=args.model)

    # Start uvicorn server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()

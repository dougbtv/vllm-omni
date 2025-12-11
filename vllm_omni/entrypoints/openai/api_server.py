import multiprocessing
import multiprocessing.forkserver as forkserver
import os
from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any, Optional

import vllm.envs as envs
from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.datastructures import State
from vllm.config import VllmConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import load_chat_template, resolve_hf_chat_template, resolve_mistral_chat_template
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.api_server import (
    base,
    build_app,
    load_log_config,
    maybe_register_tokenizer_info_endpoint,
    router,
    setup_server,
    validate_json_request,
)
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse
from vllm.entrypoints.openai.serving_models import BaseModelPath, LoRAModulePath, OpenAIServingModels
from vllm.entrypoints.openai.tool_parsers import ToolParserManager

# yapf conflicts with isort for this block
# yapf: disable
# yapf: enable
from vllm.entrypoints.tool_server import DemoToolServer, MCPToolServer, ToolServer
from vllm.entrypoints.utils import load_aware_call, with_cancellation
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import MistralTokenizer
from vllm.utils import decorate_logs

from vllm_omni.diffusion.utils.hf_utils import is_diffusion_model
from vllm_omni.entrypoints.async_diffusion import AsyncOmniDiffusion
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

# Image generation API imports
import base64
import io
import math
import time
from fastapi import File, Form, UploadFile, status
import PIL.Image
import torch
from vllm_omni.entrypoints.openai.protocol.images import (
    ImageData,
    ImageGenerationRequest,
    ImageGenerationResponse,
)
from vllm_omni.entrypoints.openai.image_model_profiles import (
    DiffusionModelProfile,
    get_model_profile,
)

logger = init_logger(__name__)


# ============================================================================
# Image Generation API Helper Functions
# ============================================================================


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
        raise ValueError(
            f"Invalid size format: '{size_str}'. Width and height must be integers."
        )

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid size: {width}x{height}. Width and height must be positive integers."
        )

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


def validate_image_file(file: UploadFile, max_size_mb: int = 4) -> None:
    """Validate uploaded image file.

    Args:
        file: Uploaded file from FastAPI
        max_size_mb: Maximum file size in megabytes

    Raises:
        HTTPException: If file is invalid or too large
    """
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image file is required"
        )

    allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image format: {file.content_type}. Supported: PNG, JPEG, WebP",
        )

    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > max_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large: {file_size/1024/1024:.1f}MB (max: {max_size_mb}MB)",
        )


async def read_image_file(file: UploadFile) -> PIL.Image.Image:
    """Read uploaded file into PIL Image.

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

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read image file: {str(e)}"
        )


def build_generation_params(
    request: ImageGenerationRequest,
    profile: DiffusionModelProfile,
    width: int,
    height: int,
    generator: Optional[torch.Generator],
) -> dict:
    """Build generation kwargs for AsyncOmniDiffusion.generate().

    Args:
        request: Image generation request
        profile: Model profile with defaults and constraints
        width: Image width in pixels
        height: Image height in pixels
        generator: Optional random generator for reproducibility

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
            if (
                request.guidance_scale is not None
                and request.guidance_scale != profile.force_guidance_scale
            ):
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
            f"Ignoring true_cfg_scale={request.true_cfg_scale}, "
            f"{profile.model_name} doesn't support this parameter"
        )

    if request.seed is not None:
        gen_params["seed"] = request.seed

    return gen_params


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
    seed: Optional[int],
) -> dict:
    """Build edit generation kwargs for AsyncOmniDiffusion.generate().

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
        seed: Optional random seed

    Returns:
        Dictionary of kwargs to pass to diffusion_engine.generate()
    """
    # Calculate dimensions
    if size:
        width, height = parse_size(size)
    else:
        # Auto-calculate maintaining ~1024x1024 area
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
        "image": image,  # Pass PIL Image to AsyncOmniDiffusion
        "height": height,
        "width": width,
        "num_outputs_per_prompt": n,
    }

    num_steps = num_inference_steps or profile.default_num_inference_steps
    if num_steps > profile.max_num_inference_steps:
        raise ValueError(
            f"num_inference_steps={num_steps} exceeds maximum for "
            f"{profile.model_name} (max={profile.max_num_inference_steps})"
        )
    gen_params["num_inference_steps"] = num_steps

    if negative_prompt and profile.supports_negative_prompt:
        gen_params["negative_prompt"] = negative_prompt

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

    if profile.supports_true_cfg_scale:
        cfg = true_cfg_scale or profile.default_true_cfg_scale
        if cfg is not None:
            gen_params["true_cfg_scale"] = cfg
    elif true_cfg_scale is not None:
        logger.warning(
            f"Ignoring true_cfg_scale={true_cfg_scale}, "
            f"{profile.model_name} doesn't support this parameter"
        )

    if seed is not None:
        gen_params["seed"] = seed

    return gen_params


# ============================================================================
# Server Entry Points
# ============================================================================


async def omni_run_server(args, **uvicorn_kwargs) -> None:
    """Run a single-worker API server.

    Automatically detects if the model is a diffusion model and routes
    to the appropriate server implementation.
    """

    # Add process-specific prefix to stdout and stderr.
    decorate_logs("APIServer")

    listen_address, sock = setup_server(args)

    # Check if model is a diffusion model
    if is_diffusion_model(args.model):
        logger.info("Detected diffusion model, starting diffusion API server")
        await omni_run_diffusion_server_worker(listen_address, sock, args, **uvicorn_kwargs)
    else:
        await omni_run_server_worker(listen_address, sock, args, **uvicorn_kwargs)


async def omni_run_diffusion_server(args, **uvicorn_kwargs) -> None:
    """Run a diffusion model API server."""

    # Add process-specific prefix to stdout and stderr.
    decorate_logs("DiffusionAPIServer")

    listen_address, sock = setup_server(args)
    await omni_run_diffusion_server_worker(listen_address, sock, args, **uvicorn_kwargs)


async def omni_run_diffusion_server_worker(listen_address, sock, args, **uvicorn_kwargs) -> None:
    """Run a diffusion model API server worker."""

    # Load logging config for uvicorn if specified
    log_config = load_log_config(args.log_config_file)
    if log_config is not None:
        uvicorn_kwargs["log_config"] = log_config

    async with build_async_diffusion(args) as diffusion_engine:
        app = build_app(args)

        await omni_diffusion_init_app_state(diffusion_engine, app.state, args)

        logger.info("Starting vLLM Diffusion API server on %s", listen_address)

        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=getattr(args, "enable_ssl_refresh", False),
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            access_log=not getattr(args, "disable_uvicorn_access_log", False),
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=getattr(args, "ssl_keyfile", None),
            ssl_certfile=getattr(args, "ssl_certfile", None),
            ssl_ca_certs=getattr(args, "ssl_ca_certs", None),
            ssl_cert_reqs=getattr(args, "ssl_cert_reqs", 0),
            h11_max_incomplete_event_size=getattr(args, "h11_max_incomplete_event_size", None),
            h11_max_header_count=getattr(args, "h11_max_header_count", None),
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    try:
        await shutdown_task
    finally:
        sock.close()


async def omni_run_server_worker(listen_address, sock, args, client_config=None, **uvicorn_kwargs) -> None:
    """Run a single API server worker."""

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    # Load logging config for uvicorn if specified
    log_config = load_log_config(args.log_config_file)
    if log_config is not None:
        uvicorn_kwargs["log_config"] = log_config

    async with build_async_omni(
        args,
        client_config=client_config,
    ) as engine_client:
        maybe_register_tokenizer_info_endpoint(args)
        app = build_app(args)

        vllm_config = await engine_client.get_vllm_config()
        await omni_init_app_state(engine_client, vllm_config, app.state, args)

        logger.info(
            "Starting vLLM API server %d on %s",
            vllm_config.parallel_config._api_process_rank,
            listen_address,
        )
        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            # NOTE: When the 'disable_uvicorn_access_log' value is True,
            # no access log will be output.
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            h11_max_incomplete_event_size=args.h11_max_incomplete_event_size,
            h11_max_header_count=args.h11_max_header_count,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    try:
        await shutdown_task
    finally:
        sock.close()


@asynccontextmanager
async def build_async_omni(
    args: Namespace,
    *,
    disable_frontend_multiprocessing: Optional[bool] = None,
    client_config: Optional[dict[str, Any]] = None,
) -> AsyncIterator[EngineClient]:
    """Build an AsyncOmni instance from command-line arguments.

    Creates an async context manager that yields an AsyncOmni instance
    configured from the provided arguments. Handles forkserver setup if
    needed and ensures proper cleanup on exit.

    Args:
        args: Parsed command-line arguments containing model and configuration
        disable_frontend_multiprocessing: Optional flag to disable frontend
            multiprocessing (deprecated in V1)
        client_config: Optional client configuration dictionary

    Yields:
        EngineClient instance (AsyncOmni) ready for use
    """
    if os.getenv("VLLM_WORKER_MULTIPROC_METHOD") == "forkserver":
        # The executor is expected to be mp.
        # Pre-import heavy modules in the forkserver process
        logger.debug("Setup forkserver with pre-imports")
        multiprocessing.set_start_method("forkserver")
        multiprocessing.set_forkserver_preload(["vllm.v1.engine.async_llm"])
        forkserver.ensure_running()
        logger.debug("Forkserver setup complete!")

    # Context manager to handle async_omni lifecycle
    # Ensures everything is shutdown and cleaned up on error/exit
    async with build_async_omni_from_stage_config(
        args,
        disable_frontend_multiprocessing=disable_frontend_multiprocessing,
    ) as async_omni:
        yield async_omni


@asynccontextmanager
async def build_async_diffusion(
    args: Namespace,
    **kwargs: Any,
) -> AsyncIterator[AsyncOmniDiffusion]:
    """Build an AsyncOmniDiffusion instance from command-line arguments.

    Creates an async context manager that yields an AsyncOmniDiffusion
    instance configured from the provided arguments.

    Args:
        args: Parsed command-line arguments containing model and configuration
        **kwargs: Additional keyword arguments passed to AsyncOmniDiffusion

    Yields:
        AsyncOmniDiffusion instance ready for use
    """
    diffusion_engine: Optional[AsyncOmniDiffusion] = None

    try:
        # Build diffusion config from args
        diffusion_kwargs = {
            "model": args.model,
        }

        # Add optional configuration from args
        if hasattr(args, "num_gpus"):
            diffusion_kwargs["num_gpus"] = args.num_gpus

        if hasattr(args, "trust_remote_code"):
            diffusion_kwargs["trust_remote_code"] = args.trust_remote_code

        diffusion_kwargs.update(kwargs)

        logger.info(
            "Building AsyncOmniDiffusion with model=%s, num_gpus=%s",
            args.model,
            diffusion_kwargs.get("num_gpus", 1),
        )

        diffusion_engine = AsyncOmniDiffusion(**diffusion_kwargs)

        yield diffusion_engine
    finally:
        if diffusion_engine:
            diffusion_engine.shutdown()


@asynccontextmanager
async def build_async_omni_from_stage_config(
    args: Namespace,
    *,
    disable_frontend_multiprocessing: bool = False,
    client_config: Optional[dict[str, Any]] = None,
) -> AsyncIterator[EngineClient]:
    """Create AsyncOmni from stage configuration.

    Creates an AsyncOmni instance either in-process or using multiprocess
    RPC. Loads stage configurations from the model or from a specified path.

    Args:
        args: Parsed command-line arguments containing model and stage configs
        disable_frontend_multiprocessing: Flag to disable frontend multiprocessing
            (deprecated in V1)
        client_config: Optional client configuration dictionary

    Yields:
        EngineClient instance (AsyncOmni) ready for use

    Note:
        Stage configurations are loaded from args.stage_configs_path if provided,
        otherwise from the model's default configuration.
    """

    # V1 AsyncLLM.
    assert envs.VLLM_USE_V1

    if disable_frontend_multiprocessing:
        logger.warning(
            "V1 is enabled, but got --disable-frontend-multiprocessing. "
            "To disable frontend multiprocessing, set VLLM_USE_V1=0."
        )

    async_omni: Optional[EngineClient] = None

    try:
        async_omni = AsyncOmni(model=args.model, cli_args=args)

        # # Don't keep the dummy data in memory
        # await async_llm.reset_mm_cache()

        yield async_omni
    finally:
        if async_omni:
            async_omni.shutdown()


async def omni_init_app_state(
    engine_client: EngineClient,
    vllm_config: VllmConfig,
    state: State,
    args: Namespace,
) -> None:
    """Initialize the FastAPI application state for omni API server.

    Sets up the application state with model information, request logger,
    and other server configuration needed for handling API requests.

    Args:
        engine_client: Engine client instance (AsyncOmni)
        vllm_config: vLLM configuration object
        state: FastAPI application state object to initialize
        args: Parsed command-line arguments
    """
    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    if args.enable_log_requests:
        request_logger = RequestLogger(max_log_len=args.max_log_len)
    else:
        request_logger = None

    base_model_paths = [BaseModelPath(name=name, model_path=args.model) for name in served_model_names]
    state.engine_client = engine_client
    state.log_stats = not args.disable_log_stats
    state.vllm_config = vllm_config
    model_config = vllm_config.model_config
    state.log_stats = not args.disable_log_stats

    # For omni models
    state.stage_configs = engine_client.stage_configs

    resolved_chat_template = load_chat_template(args.chat_template)
    if resolved_chat_template is not None:
        # Get the tokenizer to check official template
        tokenizer = await engine_client.get_tokenizer()

        if isinstance(tokenizer, MistralTokenizer):
            # The warning is logged in resolve_mistral_chat_template.
            resolved_chat_template = resolve_mistral_chat_template(chat_template=resolved_chat_template)
        else:
            hf_chat_template = resolve_hf_chat_template(
                tokenizer=tokenizer,
                chat_template=None,
                tools=None,
                model_config=vllm_config.model_config,
            )

            if hf_chat_template != resolved_chat_template:
                logger.warning(
                    "Using supplied chat template: %s\nIt is different from official chat template '%s'. This discrepancy may lead to performance degradation.",  # noqa: E501
                    resolved_chat_template,
                    args.model,
                )

    if args.tool_server == "demo":
        tool_server: Optional[ToolServer] = DemoToolServer()
        assert isinstance(tool_server, DemoToolServer)
        await tool_server.init_and_validate()
    elif args.tool_server:
        tool_server = MCPToolServer()
        await tool_server.add_tool_server(args.tool_server)
    else:
        tool_server = None

    # Merge default_mm_loras into the static lora_modules
    default_mm_loras = vllm_config.lora_config.default_mm_loras if vllm_config.lora_config is not None else {}

    lora_modules = args.lora_modules
    if default_mm_loras:
        default_mm_lora_paths = [
            LoRAModulePath(
                name=modality,
                path=lora_path,
            )
            for modality, lora_path in default_mm_loras.items()
        ]
        if args.lora_modules is None:
            lora_modules = default_mm_lora_paths
        else:
            lora_modules += default_mm_lora_paths

    state.openai_serving_models = OpenAIServingModels(
        engine_client=engine_client,
        model_config=model_config,
        base_model_paths=base_model_paths,
        lora_modules=lora_modules,
    )
    await state.openai_serving_models.init_static_loras()
    state.openai_serving_chat = OmniOpenAIServingChat(
        engine_client,
        model_config,
        state.openai_serving_models,
        args.response_role,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        trust_request_chat_template=args.trust_request_chat_template,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        enable_auto_tools=args.enable_auto_tool_choice,
        exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
        tool_parser=args.tool_call_parser,
        reasoning_parser=args.structured_outputs_config.reasoning_parser,
        enable_prompt_tokens_details=args.enable_prompt_tokens_details,
        enable_force_include_usage=args.enable_force_include_usage,
        enable_log_outputs=args.enable_log_outputs,
        log_error_stack=args.log_error_stack,
    )

    state.enable_server_load_tracking = args.enable_server_load_tracking
    state.server_load_metrics = 0


async def omni_diffusion_init_app_state(
    diffusion_engine: AsyncOmniDiffusion,
    state: State,
    args: Namespace,
) -> None:
    """Initialize the FastAPI application state for diffusion model API server.

    Sets up the application state with diffusion model information and
    chat completion handler for image generation via /v1/chat/completions.

    Args:
        diffusion_engine: AsyncOmniDiffusion engine instance
        state: FastAPI application state object to initialize
        args: Parsed command-line arguments
    """
    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    model_name = served_model_names[0] if served_model_names else args.model

    state.diffusion_engine = diffusion_engine
    state.diffusion_model_name = model_name  # Store for image endpoints
    state.log_stats = not getattr(args, "disable_log_stats", False)

    # Get default parameters from CLI args
    default_seed = getattr(args, "diffusion_seed", None)
    default_num_inference_steps = getattr(args, "num_inference_steps", 50)
    default_guidance_scale = getattr(args, "guidance_scale", 4.0)

    # Initialize chat handler with diffusion engine (uses /v1/chat/completions endpoint)
    state.openai_serving_chat = OmniOpenAIServingChat.for_diffusion(
        diffusion_engine=diffusion_engine,
        model_name=model_name,
        default_seed=default_seed,
        default_num_inference_steps=default_num_inference_steps,
        default_guidance_scale=default_guidance_scale,
    )

    # Set other handlers to None for diffusion-only mode
    state.engine_client = None
    state.vllm_config = None

    state.enable_server_load_tracking = getattr(args, "enable_server_load_tracking", False)
    state.server_load_metrics = 0

    logger.info(
        "Diffusion API server initialized for model: %s (seed=%s, steps=%d, guidance=%.2f)",
        model_name,
        default_seed,
        default_num_inference_steps,
        default_guidance_scale,
    )


def Omnichat(request: Request) -> Optional[OmniOpenAIServingChat]:
    return request.app.state.openai_serving_chat


@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    handler = Omnichat(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(message="The model does not support Chat Completions API")
    try:
        generator = await handler.create_chat_completion(request, raw_request)
    except Exception as e:
        logger.exception("Chat completion failed: %s", e)
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)) from e

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.code if hasattr(generator, "code") else 400
        )

    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


# ============================================================================
# Image Generation API Endpoints
# ============================================================================


@router.post(
    "/v1/images/generations",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"model": ImageGenerationResponse},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.SERVICE_UNAVAILABLE.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def generate_images(request: ImageGenerationRequest, raw_request: Request) -> ImageGenerationResponse:
    """Generate images from text prompts using diffusion models.

    OpenAI DALL-E compatible endpoint for text-to-image generation.

    Args:
        request: Image generation request with prompt and parameters
        raw_request: Raw FastAPI request for accessing app state

    Returns:
        ImageGenerationResponse with generated images as base64 PNG

    Raises:
        HTTPException: For validation errors, missing engine, or generation failures
    """
    # Get diffusion engine from app state
    diffusion_engine: Optional[AsyncOmniDiffusion] = getattr(raw_request.app.state, "diffusion_engine", None)
    if diffusion_engine is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Diffusion engine not initialized. Start server with a diffusion model."
        )

    # Get model profile
    model_name = getattr(raw_request.app.state, "diffusion_model_name", request.model)
    try:
        profile = get_model_profile(model_name)
    except ValueError as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail=str(e))

    try:
        # Parse and validate size
        if request.size:
            width, height = parse_size(request.size.value if hasattr(request.size, 'value') else request.size)
        else:
            width = profile.default_width
            height = profile.default_height

        # Build generation parameters
        gen_params = build_generation_params(
            request=request,
            profile=profile,
            width=width,
            height=height,
            generator=None,  # Seed handled in gen_params
        )

        logger.info(
            f"[{model_name}] Generating {request.n} image(s) - "
            f"prompt: '{request.prompt[:50]}...', size: {width}x{height}, "
            f"steps: {gen_params['num_inference_steps']}, seed: {request.seed}"
        )

        # Generate images using AsyncOmniDiffusion
        result = await diffusion_engine.generate(**gen_params)

        # Extract images from result
        images = result.images if hasattr(result, 'images') else []

        logger.info(f"Successfully generated {len(images)} image(s)")

        # Encode images to base64
        image_data = [
            ImageData(b64_json=encode_image_base64(img), revised_prompt=None)
            for img in images
        ]

        return ImageGenerationResponse(
            created=int(time.time()),
            data=image_data,
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail=str(e))
    except Exception as e:
        logger.exception(f"Image generation failed: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=f"Image generation failed: {str(e)}"
        )


@router.post(
    "/v1/images/edits",
    responses={
        HTTPStatus.OK.value: {"model": ImageGenerationResponse},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.SERVICE_UNAVAILABLE.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def edit_images(
    prompt: str = Form(..., description="Edit instruction"),
    image: UploadFile = File(..., description="Image to edit"),
    model: Optional[str] = Form(None, description="Model to use"),
    mask: Optional[UploadFile] = File(None, description="Mask for inpainting (not currently supported)"),
    n: Optional[int] = Form(1, ge=1, le=10, description="Number of edited images to generate"),
    size: Optional[str] = Form(None, description="Output size (WxH format)"),
    response_format: Optional[str] = Form("b64_json", description="Response format"),
    negative_prompt: Optional[str] = Form(None, description="Negative prompt"),
    num_inference_steps: Optional[int] = Form(None, description="Number of diffusion steps"),
    guidance_scale: Optional[float] = Form(None, description="CFG scale"),
    true_cfg_scale: Optional[float] = Form(None, description="True CFG scale (Qwen-specific)"),
    seed: Optional[int] = Form(None, description="Random seed"),
    user: Optional[str] = Form(None, description="User ID"),
    raw_request: Request = None,
) -> ImageGenerationResponse:
    """Edit images based on text instructions using diffusion models.

    OpenAI DALL-E compatible endpoint for image editing.

    Args:
        prompt: Text instruction for how to edit the image
        image: Input image file (PNG, JPEG, WebP)
        model: Optional model name (uses server's model if not specified)
        mask: Optional mask for inpainting (currently ignored with warning)
        n: Number of edited images to generate
        size: Optional output size (auto-calculated from input if omitted)
        response_format: Response format (only "b64_json" supported)
        negative_prompt: Optional negative prompt
        num_inference_steps: Optional number of diffusion steps
        guidance_scale: Optional CFG scale
        true_cfg_scale: Optional true CFG scale
        seed: Optional random seed
        user: Optional user ID for tracking
        raw_request: Raw FastAPI request for accessing app state

    Returns:
        ImageGenerationResponse with edited images as base64 PNG

    Raises:
        HTTPException: For validation errors, missing engine, or editing failures
    """
    # Get diffusion engine from app state
    diffusion_engine: Optional[AsyncOmniDiffusion] = getattr(raw_request.app.state, "diffusion_engine", None)
    if diffusion_engine is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Diffusion engine not initialized. Start server with a diffusion model."
        )

    # Get model profile
    model_name = getattr(raw_request.app.state, "diffusion_model_name", model)
    try:
        profile = get_model_profile(model_name)
    except ValueError as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail=str(e))

    try:
        # Validate and read image file
        validate_image_file(image, max_size_mb=4)
        pil_image = await read_image_file(image)

        # Handle mask (currently unsupported)
        if mask is not None:
            logger.warning(
                "Mask parameter provided but masking/inpainting not supported. "
                "Mask will be ignored - full-image editing will be performed."
            )

        # Validate response_format
        if response_format and response_format != "b64_json":
            raise ValueError(f"Unsupported response_format: {response_format}. Only 'b64_json' is supported.")

        # Build edit parameters
        gen_params = build_edit_params(
            prompt=prompt,
            image=pil_image,
            profile=profile,
            negative_prompt=negative_prompt,
            n=n or 1,
            size=size,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            seed=seed,
        )

        logger.info(
            f"[{model_name}] Editing image - "
            f"prompt: '{prompt[:50]}...', input: {pil_image.size}, "
            f"output: {gen_params['width']}x{gen_params['height']}, "
            f"n: {n}, steps: {gen_params['num_inference_steps']}, seed: {seed}"
        )

        # Generate edited images using AsyncOmniDiffusion
        result = await diffusion_engine.generate(**gen_params)

        # Extract images from result
        images = result.images if hasattr(result, 'images') else []

        logger.info(f"Successfully generated {len(images)} edited image(s)")

        # Encode images to base64
        image_data = [
            ImageData(b64_json=encode_image_base64(img), revised_prompt=None)
            for img in images
        ]

        return ImageGenerationResponse(
            created=int(time.time()),
            data=image_data,
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail=str(e))
    except Exception as e:
        logger.exception(f"Image editing failed: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=f"Image editing failed: {str(e)}"
        )

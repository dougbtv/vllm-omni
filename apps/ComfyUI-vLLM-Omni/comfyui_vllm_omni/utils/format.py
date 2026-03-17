"""Image/tensor format helpers.

The image generation part is derived from dougbtv/comfyui-vllm-omni by Doug (@dougbtv).
Original source at https://github.com/dougbtv/comfyui-vllm-omni, distributed under the MIT License.
"""

import base64
import mimetypes
from fractions import Fraction
from io import BytesIO

import av
import numpy as np
import torch
from comfy_api.input import AudioInput, VideoInput
from comfy_api.input_impl import VideoFromComponents
from comfy_api.latest._util.video_types import VideoComponents
from comfy_extras import nodes_audio
from PIL import Image

from .logger import get_logger

logger = get_logger(__name__)


def base64_to_image_tensor(base64_str: str, mode: str = "RGB") -> torch.Tensor:
    """
    Convert base64-encoded image to ComfyUI image tensor.

    Args:
        base64_str: Base64-encoded image string
        mode: PIL image mode (default RGB for transparency support)

    Returns:
        torch.Tensor with shape (1, H, W, C) in float32 [0, 1] range

    Raises:
        ValueError: If base64 string is invalid or image cannot be decoded
    """
    if base64_str.startswith("data:image"):
        _, base64_str = base64_str.split(",", 1)

    try:
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_str)
    except Exception as e:
        raise ValueError(f"Invalid base64 string: {e}")

    # Create BytesIO object for PIL
    image_bytesio = BytesIO(image_bytes)

    # Open with PIL and convert to desired mode
    try:
        pil_image = Image.open(image_bytesio)
        pil_image = pil_image.convert(mode)
    except Exception as e:
        raise RuntimeError(f"Failed to open image: {e}")

    image_array = np.asarray(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array)
    return image_tensor


def image_tensor_to_png_bytes(tensor: torch.Tensor, filename: str = "image.png") -> BytesIO:
    """
    Convert ComfyUI image tensor to PNG BytesIO for multipart upload.

    This function converts a ComfyUI IMAGE tensor to a PNG-encoded BytesIO object
    suitable for multipart/form-data upload. The BytesIO object has its .name
    attribute set, which is required by aiohttp for file uploads.

    Args:
        tensor: ComfyUI IMAGE tensor with shape (B, H, W, C), dtype float32, range [0, 1]
        filename: Name attribute to set on BytesIO (default: "image.png")

    Returns:
        BytesIO object containing PNG-encoded image with .name attribute set

    Raises:
        ValueError: If tensor format is invalid (not 4D, wrong dtype, etc.)
    """
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor with shape (B, H, W, C), got {tensor.ndim}D tensor")

    image_tensor = tensor[0]  # Shape: (H, W, C)
    image_np = (image_tensor.cpu().numpy() * 255.0).astype(np.uint8)
    pil_image = Image.fromarray(image_np)

    # Save to BytesIO as image file
    img_bytes = BytesIO()
    # Set name attribute (required for multipart upload and mimetype detection)
    img_bytes.name = filename
    try:
        pil_image.save(img_bytes)
    except Exception as e:
        raise RuntimeError(f"Failed to save image as file: {e}")

    # Reset position to beginning
    img_bytes.seek(0)

    return img_bytes


def image_tensor_to_base64(tensor: torch.Tensor, filename: str = "image.png") -> str:
    """
    Convert ComfyUI image tensor to base64-encoded image string.

    Args:
        tensor: ComfyUI IMAGE tensor with shape (B, H, W, C), dtype float32, range [0, 1]
        filename: Name attribute to set on BytesIO (default: "image.png")
        format: File format of the output image file buffer (default: "PNG")

    Returns:
        Base64-encoded image string

    Raises:
        ValueError: If tensor format is invalid (not 4D, wrong dtype, etc.)
    """
    img_bytes = image_tensor_to_png_bytes(tensor, filename)
    img_bytes.seek(0)
    byte_data = img_bytes.read()
    base64_str = base64.b64encode(byte_data).decode("utf-8")
    mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return f"data:{mime_type};base64,{base64_str}"


def video_to_bytes(video: VideoInput, filename: str = "video.mp4") -> BytesIO:
    output_buffer = BytesIO()
    output_buffer.name = filename
    video.save_to(output_buffer)
    output_buffer.seek(0)
    return output_buffer


def video_to_base64(video: VideoInput, filename: str = "video.mp4") -> str:
    video_buffer = video_to_bytes(video, filename)
    video_buffer.seek(0)
    byte_data = video_buffer.read()
    base64_str = base64.b64encode(byte_data).decode("utf-8")
    mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return f"data:{mime_type};base64,{base64_str}"


def audio_to_bytes(audio: AudioInput, filename: str = "audio.mp3", quality: str = "128k") -> BytesIO:
    waveform = audio["waveform"][0]  # Shape: (C, T)
    sample_rate = audio["sample_rate"]
    format = filename.rsplit(".", maxsplit=1)[1]
    layout = "mono" if waveform.shape[0] == 1 else "stereo"

    output_buffer = BytesIO()
    output_buffer.name = filename
    output_container = av.open(output_buffer, mode="w", format=format)
    if format == "opus":
        out_stream = output_container.add_stream("libopus", rate=sample_rate, layout=layout)
        if quality == "64k":
            out_stream.bit_rate = 64000  # type: ignore # copy from ComfyUI comfy_api/latest/_ui.py
        elif quality == "96k":
            out_stream.bit_rate = 96000  # type: ignore # copy from ComfyUI comfy_api/latest/_ui.py
        elif quality == "128k":
            out_stream.bit_rate = 128000  # type: ignore # copy from ComfyUI comfy_api/latest/_ui.py
        elif quality == "192k":
            out_stream.bit_rate = 192000  # type: ignore # copy from ComfyUI comfy_api/latest/_ui.py
        elif quality == "320k":
            out_stream.bit_rate = 320000  # type: ignore # copy from ComfyUI comfy_api/latest/_ui.py
    elif format == "mp3":
        out_stream = output_container.add_stream("libmp3lame", rate=sample_rate, layout=layout)
        if quality == "V0":
            out_stream.codec_context.qscale = 1  # type: ignore # copy from ComfyUI comfy_api/latest/_ui.py
        elif quality == "128k":
            out_stream.bit_rate = 128000  # type: ignore # copy from ComfyUI comfy_api/latest/_ui.py
        elif quality == "320k":
            out_stream.bit_rate = 320000  # type: ignore # copy from ComfyUI comfy_api/latest/_ui.py
    else:  # format == "flac":
        out_stream = output_container.add_stream("flac", rate=sample_rate, layout=layout)

    frame = av.AudioFrame.from_ndarray(
        waveform.movedim(0, 1).reshape(1, -1).float().numpy(),
        format="flt",
        layout=layout,
    )
    frame.sample_rate = sample_rate
    frame.pts = 0
    output_container.mux(out_stream.encode(frame))  # type: ignore # copy from ComfyUI comfy_api/latest/_ui.py
    # Flush encoder
    output_container.mux(out_stream.encode(None))  # type: ignore # copy from ComfyUI comfy_api/latest/_ui.py
    output_container.close()
    output_buffer.seek(0)

    return output_buffer


def audio_to_base64(audio: AudioInput, filename: str = "audio.mp3", quality: str = "128k") -> str:
    audio_buffer = audio_to_bytes(audio, filename, quality)
    audio_buffer.seek(0)
    byte_data = audio_buffer.read()
    base64_str = base64.b64encode(byte_data).decode("utf-8")
    mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return f"data:{mime_type};base64,{base64_str}"


def bytes_to_audio(audio_bytes: bytes) -> AudioInput:
    """
    Convert audio bytes to ComfyUI audio tensor.

    Args:
        audio_bytes: Audio file bytes
    Returns:
        torch.Tensor with shape (B, C, T) in float32 range [-1, 1]
    """
    audio_buffer = BytesIO(audio_bytes)
    waveform, sample_rate = nodes_audio.load(audio_buffer)  # type: ignore # Although expect string argument, it calls av.open underneath, which supports BytesIO (file-like)
    return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}


def base64_to_audio(base64_str: str) -> AudioInput:
    """
    Convert base64-encoded audio to ComfyUI audio tensor.

    Args:
        base64_str: Base64-encoded audio string
    Returns:
        torch.Tensor with shape (B, C, T) in float32 range [-1, 1]
    """
    if base64_str.startswith("data:audio"):
        _, base64_str = base64_str.split(",", 1)

    try:
        # Decode base64 to bytes
        audio_bytes = base64.b64decode(base64_str)
    except Exception as e:
        raise ValueError(f"Invalid base64 string: {e}")

    return bytes_to_audio(audio_bytes)


def base64_to_video(base64_str: str) -> VideoInput:
    """
    Convert base64-encoded video (MP4) to ComfyUI VideoInput.

    Args:
        base64_str: Base64-encoded video string (MP4 format)

    Returns:
        VideoInput with frames, frame_rate, and optional audio

    Raises:
        ValueError: If base64 string is invalid
        RuntimeError: If video cannot be decoded or missing video stream
    """
    if base64_str.startswith("data:video"):
        _, base64_str = base64_str.split(",", 1)

    try:
        # Decode base64 to bytes
        video_bytes = base64.b64decode(base64_str)
    except Exception as e:
        raise ValueError(f"Invalid base64 string: {e}")

    # Create BytesIO buffer for PyAV
    video_buffer = BytesIO(video_bytes)

    try:
        container = av.open(video_buffer)
    except Exception as e:
        raise RuntimeError(f"Failed to decode MP4 video: {e}")

    # Extract video stream
    video_stream = None
    for stream in container.streams:
        if stream.type == "video":
            video_stream = stream
            break

    if video_stream is None:
        raise RuntimeError("No video stream found in MP4 file")

    # Extract frames
    frames = []
    for frame in container.decode(video_stream):
        # Convert frame to numpy array (RGB format)
        frame_np = frame.to_ndarray(format="rgb24")  # Shape: (H, W, 3)
        # Normalize to [0, 1] and convert to torch tensor
        frame_tensor = torch.from_numpy(frame_np).float() / 255.0
        frames.append(frame_tensor)

    if not frames:
        raise RuntimeError("No frames extracted from video")

    # Stack frames: (num_frames, H, W, 3)
    video_tensor = torch.stack(frames, dim=0)

    # Get frame rate as Fraction
    frame_rate = Fraction(video_stream.average_rate.numerator, video_stream.average_rate.denominator)

    # Extract audio if present
    audio = None
    for stream in container.streams:
        if stream.type == "audio":
            # Decode audio using existing audio loading logic
            video_buffer.seek(0)
            try:
                waveform, sample_rate = nodes_audio.load(video_buffer)  # type: ignore
                audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
            except Exception:
                # Audio extraction failed, leave as None
                logger.debug("Failed to extract audio from video, continuing without audio")
            break

    container.close()

    # Create VideoInput using VideoFromComponents
    video_components = VideoComponents(images=video_tensor, frame_rate=frame_rate, audio=audio)
    return VideoFromComponents(video_components)

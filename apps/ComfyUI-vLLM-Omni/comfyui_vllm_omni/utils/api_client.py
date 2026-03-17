"""
An high-level API client adapter that forwards ComfyUI inputs to vLLM-Omni's REST API,
and transforms the API responses back to ComfyUI formats.

The image generation part is derived from dougbtv/comfyui-vllm-omni by Doug (@dougbtv).
Original source at https://github.com/dougbtv/comfyui-vllm-omni, distributed under the MIT License.
"""

from typing import Any

import aiohttp
import av.error
import torch
from comfy_api.input import AudioInput, VideoInput

from .format import (
    audio_to_base64,
    base64_to_audio,
    base64_to_image_tensor,
    base64_to_video,
    bytes_to_audio,
    image_tensor_to_base64,
    image_tensor_to_png_bytes,
    video_to_base64,
)
from .logger import get_logger, pretty_printer
from .models import lookup_model_spec
from .types import AudioFormat

logger = get_logger(__name__)


class VLLMOmniClient:
    def __init__(self, base_url: str, timeout: float = 300.0):
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def generate_image(
        self,
        *,
        model: str,
        prompt: str,
        width: int,
        height: int,
        negative_prompt: str | None = None,
        sampling_params: dict | None = None,
    ) -> torch.Tensor:
        """Run text-to-image generation via DALLE API"""
        await self._check_model_exist(model)

        size = f"{width}x{height}"
        payload = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "response_format": "b64_json",
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if sampling_params is not None:
            payload.update(sampling_params)
        logger.debug("img gen payload: %s", payload)

        url = self.base_url + "/images/generations"
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if not response.ok:
                        error_text = await response.text()
                        raise (ValueError if response.status < 500 else RuntimeError)(
                            f"vLLM-Omni API returned status {response.status}: {error_text}"
                        )

                    try:
                        data = await response.json()
                    except aiohttp.ContentTypeError as e:
                        raise RuntimeError(f"Invalid JSON response from vLLM-Omni: {e}")
                    if "data" not in data:
                        raise RuntimeError("API response missing 'data' field - expected OpenAI DALL-E format")
                    if not data["data"]:
                        raise RuntimeError("API returned empty data array")

                    image_tensors = []
                    for idx, img in enumerate(data["data"]):
                        if "b64_json" not in img:
                            raise RuntimeError(f"API returned image #{idx} without 'b64_json' field")
                        base64_str = img["b64_json"]
                        tensor = base64_to_image_tensor(base64_str)
                        image_tensors.append(tensor)
                        logger.debug("Image #%d has shape %s", idx, tensor.shape)

                    batch_tensor = torch.stack(image_tensors, dim=0)
                    logger.debug("batch_tensor output has shape: %s", batch_tensor.shape)
                    return batch_tensor

            except aiohttp.ClientError as e:
                raise RuntimeError(f"Network error connecting to vLLM-Omni at {url}: {e}")

    async def edit_image(
        self,
        *,
        model: str,
        prompt: str,
        image: torch.Tensor,
        width: int,
        height: int,
        negative_prompt: str | None = None,
        mask: torch.Tensor | None = None,
        sampling_params: dict | None = None,
    ) -> torch.Tensor:
        """Run image editing via DALLE API"""
        await self._check_model_exist(model)

        size = f"{width}x{height}"
        image_filename = "image.png"  # Required for multipart form
        form = aiohttp.FormData()
        form.add_field("model", model)
        form.add_field(
            "image",
            image_tensor_to_png_bytes(image, image_filename),
            filename=image_filename,
            content_type="image/png",
        )
        form.add_field("prompt", prompt)
        form.add_field("size", size)
        if negative_prompt:
            form.add_field("negative_prompt", negative_prompt)
        if sampling_params is not None:
            for k, v in sampling_params.items():
                form.add_field(k, str(v))
        if mask is not None:
            mask_filename = "mask.png"
            form.add_field(
                "mask",
                image_tensor_to_png_bytes(mask, mask_filename),
                filename=mask_filename,
                content_type="image/png",
            )

        url = self.base_url + "/images/edits"
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.post(url, data=form) as response:
                    if not response.ok:
                        error_text = await response.text()
                        raise (ValueError if response.status < 500 else RuntimeError)(
                            f"vLLM-Omni API returned status {response.status}: {error_text}"
                        )

                    try:
                        data = await response.json()
                    except aiohttp.ContentTypeError as e:
                        raise RuntimeError(f"Invalid JSON response from vLLM-Omni: {e}")

                    if "data" not in data:
                        raise RuntimeError("API response missing 'data' field - expected OpenAI DALL-E format")
                    if not data["data"]:
                        raise RuntimeError("API returned empty data array")

                    image_tensors = []
                    for idx, img in enumerate(data["data"]):
                        if "b64_json" not in img:
                            raise RuntimeError(f"API returned image #{idx} without 'b64_json' field")
                        base64_str = img["b64_json"]
                        tensor = base64_to_image_tensor(base64_str)
                        image_tensors.append(tensor)

                    return torch.stack(image_tensors, dim=0)

            except aiohttp.ClientError as e:
                raise RuntimeError(f"Network error connecting to vLLM-Omni at {url}: {e}")

    async def generate_video(
        self,
        *,
        model: str,
        prompt: str,
        width: int | None = None,
        height: int | None = None,
        num_frames: int | None = None,
        fps: int | None = None,
        seconds: int | None = None,
        negative_prompt: str | None = None,
        image: torch.Tensor | None = None,
        sampling_params: dict | None = None,
    ) -> VideoInput:
        """Run text-to-video or image-to-video generation via async video API

        This method handles the async job flow:
        1. POST /v1/videos to create job
        2. Poll GET /v1/videos/{job_id} until completion
        3. GET /v1/videos/{job_id}/content to download video
        4. Convert to VideoInput
        """
        await self._check_model_exist(model)

        form = aiohttp.FormData()
        form.add_field("model", model)
        form.add_field("prompt", prompt)

        # Add optional dimension parameters
        if width is not None:
            form.add_field("width", str(width))
        if height is not None:
            form.add_field("height", str(height))
        if num_frames is not None:
            form.add_field("num_frames", str(num_frames))
        if fps is not None:
            form.add_field("fps", str(fps))
        if seconds is not None:
            form.add_field("seconds", str(seconds))
        if negative_prompt:
            form.add_field("negative_prompt", negative_prompt)

        # Add sampling parameters
        if sampling_params is not None:
            for k, v in sampling_params.items():
                form.add_field(k, str(v))

        # Add image for I2V mode
        if image is not None:
            image_filename = "input_reference.png"
            form.add_field(
                "input_reference",
                image_tensor_to_png_bytes(image, image_filename),
                filename=image_filename,
                content_type="image/png",
            )

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            # Step 1: Create video generation job
            create_url = self.base_url + "/videos"
            try:
                async with session.post(create_url, data=form) as response:
                    if not response.ok:
                        error_text = await response.text()
                        raise (ValueError if response.status < 500 else RuntimeError)(
                            f"vLLM-Omni API returned status {response.status}: {error_text}"
                        )

                    try:
                        job_data = await response.json()
                    except aiohttp.ContentTypeError as e:
                        raise RuntimeError(f"Invalid JSON response from vLLM-Omni: {e}")

                    if "id" not in job_data:
                        raise RuntimeError(f"API response missing 'id' field - expected async job format. Got: {list(job_data.keys())}")

                    job_id = job_data["id"]
                    logger.info(f"Video generation job created: {job_id}")

            except aiohttp.ClientError as e:
                raise RuntimeError(f"Network error connecting to vLLM-Omni at {create_url}: {e}")

            # Step 2: Poll for job completion
            status_url = self.base_url + f"/videos/{job_id}"
            poll_interval = 2.0  # seconds
            max_wait_time = 600  # 10 minutes max
            start_time = __import__('time').time()

            while True:
                if __import__('time').time() - start_time > max_wait_time:
                    raise RuntimeError(f"Video generation timed out after {max_wait_time}s")

                try:
                    async with session.get(status_url) as response:
                        if not response.ok:
                            error_text = await response.text()
                            raise RuntimeError(f"Failed to check job status: {response.status}: {error_text}")

                        status_data = await response.json()
                        status = status_data.get("status")
                        progress = status_data.get("progress", 0)

                        logger.debug(f"Job {job_id} status: {status}, progress: {progress}%")

                        if status == "failed":
                            error = status_data.get("error", "Unknown error")
                            raise RuntimeError(f"Video generation failed: {error}")

                        if status == "completed":
                            logger.info(f"Job {job_id} completed successfully")
                            break

                        # Still in progress, wait before polling again
                        await __import__('asyncio').sleep(poll_interval)

                except aiohttp.ClientError as e:
                    raise RuntimeError(f"Network error polling job status: {e}")

            # Step 3: Download video content
            content_url = self.base_url + f"/videos/{job_id}/content"
            try:
                async with session.get(content_url) as response:
                    if not response.ok:
                        error_text = await response.text()
                        raise RuntimeError(f"Failed to download video: {response.status}: {error_text}")

                    video_bytes = await response.read()
                    logger.info(f"Downloaded video: {len(video_bytes)} bytes")

                    # Step 4: Convert MP4 bytes to VideoInput
                    from io import BytesIO
                    video_buffer = BytesIO(video_bytes)

                    # Decode using PyAV (similar to base64_to_video but from bytes)
                    import av
                    from fractions import Fraction
                    from comfy_extras import nodes_audio
                    from comfy_api.input_impl import VideoFromComponents
                    from comfy_api.latest._util.video_types import VideoComponents

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
                        frame_np = frame.to_ndarray(format="rgb24")
                        frame_tensor = torch.from_numpy(frame_np).float() / 255.0
                        frames.append(frame_tensor)

                    if not frames:
                        raise RuntimeError("No frames extracted from video")

                    video_tensor = torch.stack(frames, dim=0)
                    frame_rate = Fraction(video_stream.average_rate.numerator, video_stream.average_rate.denominator)

                    # Extract audio if present
                    audio = None
                    for stream in container.streams:
                        if stream.type == "audio":
                            video_buffer.seek(0)
                            try:
                                waveform, sample_rate = nodes_audio.load(video_buffer)  # type: ignore
                                audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
                            except Exception:
                                logger.debug("Failed to extract audio from video, continuing without audio")
                            break

                    container.close()

                    video_components = VideoComponents(images=video_tensor, frame_rate=frame_rate, audio=audio)
                    return VideoFromComponents(video_components)

            except aiohttp.ClientError as e:
                raise RuntimeError(f"Network error downloading video: {e}")

    async def generate_image_chat_completion(
        self,
        *,
        model: str,
        prompt: str,
        negative_prompt: str | None = None,
        image: torch.Tensor | None = None,
        audio: AudioInput | None = None,
        video: VideoInput | None = None,
        sampling_params: dict | list[dict] | None = None,
    ) -> torch.Tensor:
        payload = VLLMOmniClient._prepare_chat_completion_messages(
            model=model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            audio=audio,
            video=video,
            sampling_params=sampling_params,
            modalities=["image"],
        )
        choices = await self._generate_base_chat_completion(model, payload)

        image_tensors = []
        for idx, img_content in enumerate(choices[0]["message"]["content"]):
            base64_str = img_content.get("image_url", {}).get("url", "")
            if not base64_str:
                raise RuntimeError(f"API returned image #{idx} without image url")
            tensor = base64_to_image_tensor(base64_str)
            image_tensors.append(tensor)

        return torch.stack(image_tensors, dim=0)

    async def generate_understanding_chat_completion(
        self,
        *,
        model: str,
        prompt: str,
        image: torch.Tensor | None = None,
        audio: AudioInput | None = None,
        video: VideoInput | None = None,
        sampling_params: dict | list[dict] | None = None,
        modalities: list[str] = ["text", "audio"],
        **extra_body,
    ) -> tuple[str | None, AudioInput | None]:
        # Response may contain two choices: one with text, one with audio
        payload = VLLMOmniClient._prepare_chat_completion_messages(
            model=model,
            prompt=prompt,
            negative_prompt=None,
            image=image,
            audio=audio,
            video=video,
            sampling_params=sampling_params,
            modalities=modalities,
            **extra_body,
        )

        choices = await self._generate_base_chat_completion(model, payload)
        text_response = None
        audio_base64 = None
        for choice in choices:
            try:
                text_response = choice["message"]["content"]
            except (KeyError, TypeError):
                # Either this case (text response) or the audio response case will be hit. Checking None's later.
                pass
            try:
                audio_base64 = choice["message"]["audio"]["data"]
            except (KeyError, TypeError):
                # Either this case (text response) or the audio response case will be hit. Checking None's later.
                pass
        if audio_base64 is None and text_response is None:
            raise RuntimeError(
                "API response missing both '.message.audio' and 'message.content' fields."
                f"The choices object is {choices}"
            )
        if audio_base64 is not None:
            audio = base64_to_audio(audio_base64)
            logger.debug(
                "audio sample rate %d, audio shape %s, duration in second %f",
                audio["sample_rate"],
                audio["waveform"].shape,
                audio["waveform"].shape[2] / audio["sample_rate"],
            )
        else:
            audio = None
        return text_response, audio

    async def generate_speech(
        self,
        *,
        model: str,
        input: str,
        voice: str,
        response_format: AudioFormat,
        speed: float,
        **extra_params,
    ) -> AudioInput:
        await self._check_model_exist(model)

        ref_audio: AudioInput | None = extra_params.pop("ref_audio", None)

        payload = {
            "model": model,
            "input": input,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
            **extra_params,
        }

        if ref_audio is not None:
            audio_base64 = audio_to_base64(ref_audio)
            payload["ref_audio"] = audio_base64

        logger.debug("Omni TTS payload: %s", pretty_printer.pformat(payload))

        url = self.base_url + "/audio/speech"
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if not response.ok:
                        error_text = await response.text()
                        raise (ValueError if response.status < 500 else RuntimeError)(
                            f"vLLM-Omni API returned status {response.status}: {error_text}"
                        )

                    try:
                        audio_bytes = await response.read()
                    except aiohttp.ContentTypeError as e:
                        raise RuntimeError(f"Invalid JSON response from vLLM-Omni: {e}")

                    try:
                        audio = bytes_to_audio(audio_bytes)
                    except av.error.InvalidDataError as e:
                        raise ValueError(
                            f"Invalid audio data received from vLLM-Omni: {e}"
                            "Check if you have input unsupported arguments (such as 'voice')"
                        )
                    return audio

            except aiohttp.ClientError as e:
                raise RuntimeError(f"Network error connecting to vLLM-Omni at {url}: {e}")

    async def _generate_base_chat_completion(self, model: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        logger.debug("Omni payload: %s", pretty_printer.pformat(payload))
        await self._check_model_exist(model)

        url = self.base_url + "/chat/completions"
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if not response.ok:
                        error_text = await response.text()
                        raise (ValueError if response.status < 500 else RuntimeError)(
                            f"vLLM-Omni API returned status {response.status}: {error_text}"
                        )

                    try:
                        data = await response.json()
                    except aiohttp.ContentTypeError as e:
                        raise RuntimeError(f"Invalid JSON response from vLLM-Omni: {e}")

                    logger.debug(
                        "chat completion response: %s",
                        pretty_printer.pformat(data),
                    )

                    try:
                        return data["choices"]
                    except (KeyError, TypeError):
                        raise RuntimeError("Invalid JSON response from vLLM-Omni: missing 'choices' field")

            except aiohttp.ClientError as e:
                raise RuntimeError(f"Network error connecting to vLLM-Omni at {self.base_url}: {e}")

    async def _check_model_exist(self, model: str):
        url = self.base_url + "/models"
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.get(
                    url,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if not response.ok:
                        error_text = await response.text()
                        raise (ValueError if response.status < 500 else RuntimeError)(
                            f"vLLM-Omni API returned status {response.status} "
                            f"when getting hosted model list: {error_text}"
                        )

                    try:
                        data = await response.json()
                    except aiohttp.ContentTypeError as e:
                        raise RuntimeError(f"Invalid JSON response when getting hosted model list from vLLM-Omni: {e}")

            except aiohttp.ClientError as e:
                raise RuntimeError(f"Network error connecting to vLLM-Omni at {self.base_url}: {e}")
        try:
            model_list = data["data"]
            model_found = next((True for m in model_list if m["id"] == model), False)
        except (KeyError, TypeError):
            raise RuntimeError(f"Invalid JSON response of the hosted model list: {data}")

        if not model_found:
            raise ValueError(f"Model {model} not served at {self.base_url}.")

    @staticmethod
    def _prepare_chat_completion_messages(
        *,
        model: str,
        prompt: str,
        negative_prompt: str | None,
        image: torch.Tensor | None = None,
        audio: AudioInput | None = None,
        video: VideoInput | None = None,
        sampling_params: dict | list[dict] | None = None,
        modalities: list[str] | None = None,  # diffusion don't have this field
        **extra_body,
    ):
        message_content: list[dict] = [{"type": "text", "text": prompt}]
        if image is not None:
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_tensor_to_base64(image)},
                }
            )
        if audio is not None:
            message_content.append({"type": "audio_url", "audio_url": {"url": audio_to_base64(audio)}})
        if video is not None:
            message_content.append({"type": "video_url", "video_url": {"url": video_to_base64(video)}})
        messages = [{"role": "user", "content": message_content}]

        payload: dict[str, Any] = {"messages": messages, "model": model}
        if modalities:
            payload["modalities"] = modalities

        combined_extra_body: dict[str, Any] = {}
        if sampling_params is not None:
            spec, _ = lookup_model_spec(model)
            is_single_sampling_param = isinstance(sampling_params, dict) or len(sampling_params) == 1

            if (spec is None and is_single_sampling_param) or (spec is not None and spec["stages"] == ["diffusion"]):
                # Diffusion format: extra_body directly contains sampling params.
                # Validation has already taken care of matching sampling params' types and length. Safe to take [0].
                # * Use this mode if the model is a simple one-stage diffusion model.
                # * Fallback to this mode if model is not registered and a single sampling param is provided.
                sampling_params = sampling_params if isinstance(sampling_params, dict) else sampling_params[0]
                combined_extra_body: dict[str, Any] = sampling_params.copy()
                if "n" in combined_extra_body:
                    combined_extra_body["num_outputs_per_prompt"] = combined_extra_body.pop("n")
            else:
                # AR format: the payload has a sampling_params_list field, containing a list.
                sampling_params_list = sampling_params if isinstance(sampling_params, list) else [sampling_params]
                payload["sampling_params_list"] = sampling_params_list

        if negative_prompt:
            combined_extra_body["negative_prompt"] = negative_prompt

        if extra_body:
            combined_extra_body.update(extra_body)

        # Add extra_body only if it has any content.
        if combined_extra_body:
            payload["extra_body"] = combined_extra_body

        # Place to inject any model-specific payload adjustment
        spec, _ = lookup_model_spec(model)
        if spec:
            preprocessor = spec.get("payload_preprocessor", None)
            if preprocessor is not None:
                payload = preprocessor(payload)

        return payload

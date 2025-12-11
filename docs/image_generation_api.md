# Image Generation API

vLLM-Omni provides an OpenAI DALL-E compatible API for text-to-image generation using diffusion models.

## Supported Models

The image generation API supports multiple diffusion models:

**Image Generation:**
- **Qwen/Qwen-Image**: Alibaba's Qwen-Image model with true CFG support
- **Tongyi-MAI/Z-Image-Turbo**: Fast Z-Image Turbo model optimized for ~9 inference steps

**Image Editing:**
- **Qwen/Qwen-Image-Edit**: Full-image editing based on text prompts

Each server instance runs a single model (specified at startup via `--model`).

## Quick Start

### 1. Start the Server

**With Qwen-Image:**
```bash
vllm serve Qwen/Qwen-Image --omni \
  --host 0.0.0.0 \
  --port 8000
```

**With Z-Image Turbo:**
```bash
vllm serve Tongyi-MAI/Z-Image-Turbo --omni \
  --host 0.0.0.0 \
  --port 8000
```

The server will:
- Load the specified diffusion model
- Start listening on `http://0.0.0.0:8000`
- Expose the `/v1/images/generations` and `/v1/images/edits` endpoints
- Also expose `/v1/chat/completions` for chat-based image generation

### 2. Generate Images

Using the example client:

```bash
python examples/online_serving/image_generation/client.py \
  --prompt "a cat on a laptop" \
  --output cat.png
```

Using curl:

```bash
curl -X POST http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful sunset over mountains",
    "size": "1024x1024",
    "n": 1,
    "seed": 42
  }' | jq -r '.data[0].b64_json' | base64 -d > sunset.png
```

## Image Editing API

vLLM-Omni supports image editing via the `/v1/images/edits` endpoint, compatible with OpenAI's DALL-E image editing API.

### Endpoint

```
POST /v1/images/edits
Content-Type: multipart/form-data
```

### Quick Start

**Start edit server:**
```bash
vllm serve Qwen/Qwen-Image-Edit --omni \
  --port 8000
```

**Edit image:**
```bash
curl -X POST http://localhost:8000/v1/images/edits \
  -F "image=@input.png" \
  -F "prompt=make the sky blue and add clouds" \
  -F "n=1" \
  -F "seed=42" \
  | jq -r '.data[0].b64_json' | base64 -d > edited.png
```

### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image` | file | **Yes** | Image to edit (PNG, JPEG, WebP, max 4MB) |
| `prompt` | string | **Yes** | Text instruction for how to edit the image |
| `model` | string | No | Model name (must match server if specified) |
| `mask` | file | No | **Not currently supported** - ignored with warning |
| `n` | integer | No | Number of edited images (1-10, default: 1) |
| `size` | string | No | Output size (WxH). Auto-calculated if omitted |
| `response_format` | string | No | `b64_json` only (default) |
| `negative_prompt` | string | No | What to avoid in edited image |
| `num_inference_steps` | integer | No | Steps (default: 50, max: 200) |
| `guidance_scale` | float | No | CFG scale (default: 1.0) |
| `true_cfg_scale` | float | No | True CFG scale (default: 4.0) |
| `seed` | integer | No | Random seed for reproducibility |

### Automatic Size Calculation

If `size` is omitted, output dimensions are calculated from the input image's aspect ratio while maintaining ~1,024,024 pixels:

- Input: 800x600 (4:3) → Output: ~1152x864
- Input: 1920x1080 (16:9) → Output: ~1344x768
- Input: 1080x1920 (9:16) → Output: ~768x1344

**Note:** Dimensions are always multiples of 32 (VAE constraint).

### Examples

#### Python Client

```python
import requests
import base64
from PIL import Image
import io

with open("input.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/v1/images/edits",
        files={"image": f},
        data={
            "prompt": "add a sunset in the background",
            "num_inference_steps": 50,
            "seed": 42,
        }
    )

img_data = response.json()["data"][0]["b64_json"]
img_bytes = base64.b64decode(img_data)
img = Image.open(io.BytesIO(img_bytes))
img.save("edited.png")
```

#### Multiple Variations

```bash
curl -X POST http://localhost:8000/v1/images/edits \
  -F "image=@landscape.jpg" \
  -F "prompt=change season to winter with snow" \
  -F "n=4" \
  -F "seed=123"
```

#### With Explicit Size

```bash
curl -X POST http://localhost:8000/v1/images/edits \
  -F "image=@portrait.png" \
  -F "prompt=professional headshot style" \
  -F "size=1024x1024"
```

### Current Limitations

- **No masking/inpainting**: The `mask` parameter is accepted but ignored
- **One model per server**: Specify edit model at startup
- **Base64 only**: URL response format not supported

## API Reference

### Endpoint

```
POST /v1/images/generations
```

### Request Format

```json
{
  "prompt": "text description",
  "model": "Qwen/Qwen-Image",
  "n": 1,
  "size": "1024x1024",
  "response_format": "b64_json",
  "negative_prompt": "text to avoid",
  "num_inference_steps": 50,
  "true_cfg_scale": 4.0,
  "seed": 42
}
```

### Request Parameters

#### OpenAI Standard Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | **required** | Text description of the desired image |
| `model` | string | `null` (uses server's model) | Model to use (optional, must match server if specified) |
| `n` | integer | `1` | Number of images to generate (1-10) |
| `size` | string | `"1024x1024"` | Image dimensions (WxH format) |
| `response_format` | string | `"b64_json"` | Response format (`b64_json` only) |
| `user` | string | `null` | User identifier for tracking |

#### vllm-omni Extension Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `negative_prompt` | string | `null` | Text describing what to avoid |
| `num_inference_steps` | integer | model-dependent | Number of diffusion steps (1-200, see table below) |
| `guidance_scale` | float | model-dependent | Classifier-free guidance scale (0.0-20.0) |
| `true_cfg_scale` | float | model-dependent | True CFG scale for Qwen-Image (0.0-20.0, ignored by Z-Image) |
| `seed` | integer | `null` | Random seed for reproducibility |

#### Parameter Support by Model

Different models have different defaults and constraints:

| Parameter | Qwen/Qwen-Image | Tongyi-MAI/Z-Image-Turbo |
|-----------|-----------------|---------------------------|
| `num_inference_steps` | Default: 50, Max: 200 | Default: 9, Max: 16 |
| `guidance_scale` | Default: 1.0 (user configurable) | **Forced to 0.0** (user input ignored) |
| `true_cfg_scale` | Default: 4.0 (used by model) | **Ignored** (Qwen-specific parameter) |
| `negative_prompt` | ✅ Supported | ✅ Supported |

**Important notes:**
- Z-Image Turbo is distilled for `guidance_scale=0.0` and will **always** use this value regardless of user input
- Z-Image Turbo will **ignore** `true_cfg_scale` (Qwen-specific parameter)
- If `num_inference_steps` exceeds the model's max, the request will be rejected with a 400 error

### Response Format

```json
{
  "created": 1701234567,
  "data": [
    {
      "b64_json": "<base64-encoded PNG>",
      "url": null,
      "revised_prompt": null
    }
  ]
}
```

### Supported Image Sizes

- `256x256`
- `512x512`
- `1024x1024`
- `1792x1024`
- `1024x1792`

Custom sizes are also supported (use WxH format like `800x600`).

## Error Responses

### 400 Bad Request

Invalid parameters or unsupported format:

```json
{
  "detail": "Invalid size format: 'invalid'. Expected format: 'WIDTHxHEIGHT'"
}
```

### 422 Unprocessable Entity

Validation errors (Pydantic):

```json
{
  "detail": [
    {
      "loc": ["body", "prompt"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 500 Internal Server Error

Generation failures:

```json
{
  "detail": "Image generation failed: CUDA out of memory"
}
```

### 503 Service Unavailable

Model not loaded:

```json
{
  "detail": "Model not loaded. Server may still be initializing."
}
```

## Examples

### Python Client

```python
import requests
import base64
from PIL import Image
import io

# Generate image
response = requests.post(
    "http://localhost:8000/v1/images/generations",
    json={
        "prompt": "a dragon flying over mountains",
        "size": "1024x1024",
        "num_inference_steps": 50,
        "seed": 42,
    }
)

# Decode and save
img_data = response.json()["data"][0]["b64_json"]
img_bytes = base64.b64decode(img_data)
img = Image.open(io.BytesIO(img_bytes))
img.save("dragon.png")
```

### Using OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # No auth required for PoC
)

response = client.images.generate(
    model="Qwen/Qwen-Image",
    prompt="a white siamese cat",
    n=1,
    size="1024x1024",
    response_format="b64_json"
)

# Note: Extension parameters (seed, steps) require direct HTTP requests
```

### Z-Image Turbo Example

Z-Image Turbo is optimized for fast generation with ~9 inference steps:

```bash
# Start Z-Image server
vllm serve Tongyi-MAI/Z-Image-Turbo --omni \
  --port 8000

# Generate image (in another terminal)
curl -X POST http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful mountain landscape at sunset",
    "size": "1024x1024",
    "num_inference_steps": 9,
    "seed": 42
  }' | jq -r '.data[0].b64_json' | base64 -d > landscape.png
```

**Z-Image Turbo notes:**
- Default steps: 9 (recommended for best quality/speed trade-off)
- Maximum steps: 16 (requests with more steps will be rejected)
- `guidance_scale` is always forced to 0.0 (Turbo is distilled for CFG=0)
- `true_cfg_scale` is ignored (Qwen-specific parameter)

### Multiple Images

```bash
python examples/online_serving/image_generation/client.py \
  --prompt "a futuristic city" \
  --output city.png \
  --n 4 \
  --seed 123
```

This will generate 4 images saved as:
- `city_0.png`
- `city_1.png`
- `city_2.png`
- `city_3.png`

### With Negative Prompt

```bash
python examples/online_serving/image_generation/client.py \
  --prompt "a beautiful landscape" \
  --negative-prompt "blurry, low quality, distorted, ugly" \
  --output landscape.png \
  --steps 100
```

## Server Configuration

### Command Line Options

```bash
vllm serve --help
```

Key options for diffusion models:
- `--omni`: Enable omni mode for diffusion models (required)
- `--host`: Host address (default: `0.0.0.0`)
- `--port`: Port number (default: `8000`)
- `--uvicorn-log-level`: Server logging level (default: `info`)

### Development Mode

```bash
vllm serve <model> --omni \
  --uvicorn-log-level debug
```

## Performance Tips

### Memory Optimization

The server automatically enables:
- VAE slicing: Reduces memory usage for large images
- VAE tiling: Enables processing of very large images

### Reducing Generation Time

- Use fewer steps: `--steps 25` (lower quality but faster)
- Generate smaller images: `--size 512x512`

### CUDA Out of Memory

If you encounter OOM errors:
1. Reduce image size: `512x512` instead of `1024x1024`
2. Reduce number of steps: `25` instead of `50`
3. Generate one image at a time: `n=1`

## Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "model": "Qwen/Qwen-Image",
  "ready": true
}
```

### Logs

The server logs:
- Request parameters (prompt, size, steps)
- Generation time
- Errors with stack traces

Example log:
```
INFO: Generating 1 image(s) - prompt: 'a cat on a laptop...' size: 1024x1024, steps: 50, cfg: 4.0, seed: 42
INFO: Successfully generated 1 image(s)
```

## Troubleshooting

### Server Won't Start

**Error**: `Failed to load model`

Check:
- Model name is correct: `Qwen/Qwen-Image`
- HuggingFace cache is accessible
- Sufficient disk space for model download

### Connection Refused

**Error**: `Connection refused`

Check:
- Server is running: `curl http://localhost:8000/health`
- Correct host/port: default is `0.0.0.0:8000`
- Firewall settings

### Invalid Size

**Error**: `Invalid size format`

Use correct format: `WIDTHxHEIGHT`
- Valid: `1024x1024`, `512x768`
- Invalid: `1024`, `1024x`

### URL Response Format Not Supported

**Error**: `'url' response format is not supported`

Currently only `b64_json` is supported. Images are returned as base64-encoded PNG data.

## Integration with ComfyUI

The ComfyUI custom node (separate project) integrates with this API:

1. Start this server on a known URL
2. Install ComfyUI custom node
3. Configure node to point to server URL
4. Use in ComfyUI workflows

See the ComfyUI integration documentation for details.

## Limitations (PoC)

This is a proof-of-concept implementation with some limitations:

1. **Synchronous only**: Requests block during generation
2. **Single model**: Cannot switch models without restart
3. **Base64 only**: URL response format not implemented
4. **No batching**: Processes one request at a time
5. **No authentication**: Open access

## Future Enhancements

Planned improvements:
- Async request handling for concurrency
- Multiple model support
- URL response format with image hosting
- Request queuing and load balancing
- Integration with main `vllm serve` command
- Authentication and rate limiting
- Prometheus metrics endpoint

## Testing

### Run Tests

```bash
# Unit tests (no GPU required)
pytest tests/entrypoints/openai/test_image_server.py -v

# Specific test
pytest tests/entrypoints/openai/test_image_server.py::test_parse_size_valid -v
```

### Manual Testing

```bash
# Start server
vllm serve Qwen/Qwen-Image --omni --port 8000

# Test generation
curl http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test image"}'

# Test generation
curl -X POST http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test", "size": "512x512"}' \
  | jq
```

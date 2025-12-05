# Image Generation API Example

This directory contains example code for using the vLLM-Omni text-to-image generation API.

## Prerequisites

1. Start the image generation server:

```bash
python -m vllm_omni.entrypoints.openai.serving_image \
  --model Qwen/Qwen-Image \
  --port 8000
```

2. Install required dependencies (if not already installed):

```bash
pip install requests pillow
```

## Usage

### Basic Example

Generate a single image:

```bash
python client.py --prompt "a cat on a laptop" --output cat.png
```

### Advanced Examples

Generate multiple images with custom parameters:

```bash
python client.py \
  --prompt "a dragon flying over mountains" \
  --output dragon.png \
  --n 3 \
  --seed 42 \
  --steps 100 \
  --size 1024x1024
```

Use negative prompt to avoid unwanted elements:

```bash
python client.py \
  --prompt "beautiful landscape" \
  --negative-prompt "blurry, low quality, distorted" \
  --output landscape.png
```

### Using curl

You can also call the API directly with curl:

```bash
curl -X POST http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cat on a laptop",
    "size": "1024x1024",
    "num_inference_steps": 50,
    "true_cfg_scale": 4.0,
    "seed": 42
  }' | jq -r '.data[0].b64_json' | base64 -d > output.png
```

## Parameters

- `--prompt`: Text description of the desired image (required)
- `--output`: Output file path (default: output.png)
- `--n`: Number of images to generate (default: 1)
- `--size`: Image dimensions in WIDTHxHEIGHT format (default: 1024x1024)
- `--seed`: Random seed for reproducibility (default: random)
- `--negative-prompt`: Text describing what to avoid
- `--steps`: Number of diffusion steps (default: 50)
- `--cfg-scale`: Classifier-free guidance scale (default: 4.0)
- `--api-base`: API server URL (default: http://localhost:8000)
- `--model`: Model name (default: Qwen/Qwen-Image)

## OpenAI SDK Compatibility

The API is compatible with the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # vllm-omni doesn't require auth for PoC
)

response = client.images.generate(
    model="Qwen/Qwen-Image",
    prompt="a white siamese cat",
    n=1,
    size="1024x1024",
    response_format="b64_json"
)

# Note: Extra parameters (seed, steps, cfg) require direct HTTP requests for now
```

## Troubleshooting

### Server Not Running

```
Error: Failed to connect to API server
```

Make sure the server is running at the specified URL:

```bash
# Check if server is responding
curl http://localhost:8000/health
```

### Out of Memory

If you get CUDA out of memory errors, try:
- Reducing image size: `--size 512x512`
- Reducing number of steps: `--steps 25`
- Generating fewer images: `--n 1`

### Invalid Size Format

The `--size` parameter must be in `WIDTHxHEIGHT` format (e.g., `1024x1024`, `512x768`).

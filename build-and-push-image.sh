#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./build-and-push-image.sh --rev <tag> [--remote dougbtv@a100-07] [--gpu 0]
                            [--prompt "A majestic mountain landscape"]
                            [--output z_image_output.png]
                            [--skip-build]

Example:
  HF_TOKEN=hf_... ./build-and-push-image.sh --rev zimagebase-rev-b \
    --prompt "A majestic mountain landscape at sunset"

  # Skip build/push, just run (useful for quick iteration):
  HF_TOKEN=hf_... ./build-and-push-image.sh --rev zimagebase-rev-b --skip-build

Notes:
- Builds and pushes: quay.io/dosmith/vllm-omni:<rev>
- Use --skip-build to skip build/push and just run the container remotely
- Remote uses podman explicitly (avoids docker-vs-podman alias issues over ssh)
- Streams Python output to your terminal.
- Default model: Tongyi-MAI/Z-Image (Z-Image Base with CFG support)
EOF
}

REV=""
SKIP_BUILD=false

# ============================================================================
# DEFAULTS - Customize these for your environment
# ============================================================================
# Remote host for testing (can override with --remote)
REMOTE="${REMOTE:-dougbtv@a100-07}"

# GPU device to use (can override with --gpu)
GPU="${GPU:-0}"

# Generation parameters (can override with --prompt/--output)
PROMPT="${PROMPT:-An antique Western-style wide photograph of a sacred balancing stone pillar rising from a vast desert, with a futuristic DeLorean-style stainless steel time-travel car hovering just above the stone, styled like an 1800s tintype with sepia tones, film grain, dust specks, plate scratches, soft vignetting, and aged photographic texture, contrasted by subtle electric-blue energy arcs, glowing flux lines, and neon reflections on the vehicle, golden-hour cinematic lighting, dramatic scale, surreal yet believable alternate-history atmosphere, no people, no text, no modern city, no cartoon or anime style}"
OUTPUT="${OUTPUT:-z_image_output.png}"

# Container image settings (customize for your registry)
IMAGE_REPO="${IMAGE_REPO:-quay.io/dosmith/vllm-omni}"
DOCKERFILE="${DOCKERFILE:-docker/Dockerfile.ltx2}"
CONTAINER_WORKDIR="/workspace/vllm-omni"

# Remote paths (customize for your setup)
REMOTE_HF_CACHE="${REMOTE_HF_CACHE:-/mnt/nvme-data/engine/dougbtv/hub_cache}"
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR:-/home/dougbtv/mp4s}"

# Model to use
MODEL="${MODEL:-Tongyi-MAI/Z-Image}"
# ============================================================================

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rev) REV="${2:-}"; shift 2 ;;
    --remote) REMOTE="${2:-}"; shift 2 ;;
    --gpu) GPU="${2:-}"; shift 2 ;;
    --prompt) PROMPT="${2:-}"; shift 2 ;;
    --output) OUTPUT="${2:-}"; shift 2 ;;
    --skip-build) SKIP_BUILD=true; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$REV" ]]; then
  echo "ERROR: --rev is required"
  usage
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN env var not set."
  echo "Run like: HF_TOKEN=hf_... $0 --rev $REV"
  exit 1
fi

IMAGE="${IMAGE_REPO}:${REV}"

if [[ "$SKIP_BUILD" == "false" ]]; then
  echo "==> Building ${IMAGE}"
  podman build -t "${IMAGE}" -f "${DOCKERFILE}" .

  echo "==> Pushing ${IMAGE}"
  podman push "${IMAGE}"

  echo "==> Remote pull (using podman explicitly to avoid non-interactive docker alias issues)"
  ssh -o BatchMode=yes "${REMOTE}" "podman pull '${IMAGE}'"
else
  echo "==> Skipping build/push (--skip-build enabled)"
  echo "==> Using existing image: ${IMAGE}"
fi

echo "==> Remote run (streaming output)"
# - We use podman explicitly
# - We run python directly (no need for interactive bash)
# - We keep your GPU and HF cache mounts/env
ssh -t "${REMOTE}" bash -lc "set -euo pipefail
  podman run --rm \
    --device nvidia.com/gpu=${GPU} \
    --security-opt=label=disable \
    --userns=keep-id \
    --security-opt label=level:s0 \
    -e NVIDIA_VISIBLE_DEVICES=${GPU} \
    -e CUDA_VISIBLE_DEVICES=${GPU} \
    -e HF_TOKEN='${HF_TOKEN}' \
    -e HF_HOME=/tmp/hf \
    -e HUGGINGFACE_HUB_CACHE=/hf/hub \
    -v '${REMOTE_HF_CACHE}:/hf/hub' \
    -v '${REMOTE_OUTPUT_DIR}:/output' \
    -w '${CONTAINER_WORKDIR}' \
    '${IMAGE}' \
    python examples/offline_inference/text_to_image/text_to_image.py \
      --model '${MODEL}' \
      --prompt '${PROMPT}' \
      --negative_prompt 'blurry, low quality, distorted' \
      --height 1280 \
      --width 720 \
      --num_inference_steps 50 \
      --guidance_scale 4.0 \
      --seed 42 \
      --output '/output/${OUTPUT}'
"

echo ""
echo "==> Done! Image saved to: ${REMOTE_OUTPUT_DIR}/${OUTPUT}"
echo "    (Accessible on remote host at: ssh ${REMOTE} 'ls -lh ${REMOTE_OUTPUT_DIR}/${OUTPUT}')"

#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./build-and-push.sh --rev <tag-revision-d> [--remote dougbtv@a100-07] [--gpu 0]
                      [--prompt "A cat playing with a ball"]
                      [--output test_output.mp4]
                      [--skip-build]

Example:
  HF_TOKEN=hf_... ./build-and-push.sh --rev ad --remote dougbtv@a100-07 \
    --prompt "A cat playing with a ball" --output test_output.mp4

  # Skip build/push, just run (useful for quick iteration):
  HF_TOKEN=hf_... ./build-and-push.sh --rev ad --skip-build

Notes:
- Builds and pushes: quay.io/dosmith/vllm-omni:ltx2-rev-<rev>
- Use --skip-build to skip build/push and just run the container remotely
- Remote uses podman explicitly (avoids docker-vs-podman alias issues over ssh)
- Streams Python output to your terminal.
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
PROMPT="${PROMPT:-A cat playing with a ball}"
OUTPUT="${OUTPUT:-test_output.mp4}"

# Container image settings (customize for your registry)
IMAGE_REPO="${IMAGE_REPO:-quay.io/dosmith/vllm-omni}"
DOCKERFILE="${DOCKERFILE:-docker/Dockerfile.ltx2}"
CONTAINER_WORKDIR="/workspace/vllm-omni"

# Remote paths (customize for your setup)
REMOTE_HF_CACHE="${REMOTE_HF_CACHE:-/mnt/nvme-data/engine/dougbtv/hub_cache}"
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR:-/home/dougbtv/mp4s}"

# Model to use
MODEL="${MODEL:-Lightricks/LTX-2}"
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

IMAGE="${IMAGE_REPO}:ltx2-rev-${REV}"

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
    python examples/offline_inference/text_to_video_ltx2.py \
      --model '${MODEL}' \
      --prompt \"${PROMPT}\" \
      --height 1024 --width 1024 --num_frames 241 \
      --output '/output/${OUTPUT}'
"

echo "==> Done. Remote output file: ${REMOTE_OUTPUT_DIR}/${OUTPUT}"
echo "    (Accessible on remote host at the path shown above)"

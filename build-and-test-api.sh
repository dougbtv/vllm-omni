#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Build, Deploy, and Test LTX-2 OpenAI Video API
# ==============================================================================
# This script:
# 1. Builds and pushes the vllm-omni:ltx2 image
# 2. Deploys it as an API server on the remote GPU host
# 3. Tests the /v1/videos endpoint
# 4. Validates video generation
#
# Usage:
#   HF_TOKEN=hf_... ./build-and-test-api.sh --rev ae
#   HF_TOKEN=hf_... ./build-and-test-api.sh --rev ae --skip-build
#   HF_TOKEN=hf_... ./build-and-test-api.sh --rev ae --stop-only
# ==============================================================================

usage() {
  cat <<'EOF'
Usage:
  ./build-and-test-api.sh --rev <tag-revision> [OPTIONS]

Options:
  --rev <str>         Tag revision (e.g., "ae") - REQUIRED
  --remote <host>     Remote host (default: dougbtv@a100-07)
  --gpu <id>          GPU device ID (default: 0)
  --skip-build        Skip build/push, use existing image
  --stop-only         Stop running container and exit
  --keep-running      Don't stop container after testing
  -h, --help          Show this help

Examples:
  # Full build + test:
  HF_TOKEN=hf_... ./build-and-test-api.sh --rev ae

  # Skip build (use existing image):
  HF_TOKEN=hf_... ./build-and-test-api.sh --rev ae --skip-build

  # Stop running container:
  ./build-and-test-api.sh --rev ae --stop-only
EOF
}

# ==============================================================================
# Configuration
# ==============================================================================
REV=""
SKIP_BUILD=false
STOP_ONLY=false
KEEP_RUNNING=false

REMOTE="${REMOTE:-dougbtv@h100-02}"
GPU="${GPU:-7}"
PORT="8000"

IMAGE_REPO="${IMAGE_REPO:-quay.io/dosmith/vllm-omni}"
DOCKERFILE="${DOCKERFILE:-docker/Dockerfile.ltx2}"
CONTAINER_NAME="ltx2-api"

# Remote paths
REMOTE_HF_CACHE="${REMOTE_HF_CACHE:-/mnt/nfs-preprod-1/dougbtv/hub_cache}"
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR:-/home/dougbtv/mp4s}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:-/workspace/vllm-omni}"

# Model (HF model ID)
MODEL="Lightricks/LTX-2"

# Test parameters
TEST_PROMPT="A majestic dragon soaring through clouds at sunset"
TEST_SECONDS=2
TEST_FPS=24
TEST_SIZE="768x512"

# ==============================================================================
# Parse arguments
# ==============================================================================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rev) REV="${2:-}"; shift 2 ;;
    --remote) REMOTE="${2:-}"; shift 2 ;;
    --gpu) GPU="${2:-}"; shift 2 ;;
    --skip-build) SKIP_BUILD=true; shift 1 ;;
    --stop-only) STOP_ONLY=true; shift 1 ;;
    --keep-running) KEEP_RUNNING=true; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$REV" ]]; then
  echo "ERROR: --rev is required"
  usage
  exit 1
fi

IMAGE="${IMAGE_REPO}:ltx2-rev-${REV}"

# ==============================================================================
# Stop-only mode
# ==============================================================================
if [[ "$STOP_ONLY" == "true" ]]; then
  echo "==> Stopping and removing container on remote"
  ssh "${REMOTE}" "podman stop ${CONTAINER_NAME} 2>/dev/null || true; podman rm ${CONTAINER_NAME} 2>/dev/null || true"
  echo "==> Done"
  exit 0
fi

# ==============================================================================
# Check HF_TOKEN
# ==============================================================================
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN env var not set."
  echo "Run like: HF_TOKEN=hf_... $0 --rev $REV"
  exit 1
fi

# ==============================================================================
# Build and push image
# ==============================================================================
if [[ "$SKIP_BUILD" == "false" ]]; then
  echo "==> Building ${IMAGE}"
  podman build -t "${IMAGE}" -f "${DOCKERFILE}" .

  echo "==> Pushing ${IMAGE}"
  podman push "${IMAGE}"

  echo "==> Pulling on remote"
  ssh -o BatchMode=yes "${REMOTE}" "podman pull '${IMAGE}'"
else
  echo "==> Skipping build/push (using existing image: ${IMAGE})"
fi

# ==============================================================================
# Stop existing container (if running)
# ==============================================================================
echo "==> Stopping any existing ${CONTAINER_NAME} container"
ssh "${REMOTE}" "podman stop ${CONTAINER_NAME} 2>/dev/null || true"
sleep 2

# ==============================================================================
# Start API server on remote
# ==============================================================================
echo "==> Starting LTX-2 API server on ${REMOTE}"
CONTAINER_ID=$(ssh "${REMOTE}" "podman run -d --name ${CONTAINER_NAME} \
  --device nvidia.com/gpu=${GPU} \
  --security-opt=label=disable \
  --userns=keep-id \
  --security-opt label=level:s0 \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HF_TOKEN='${HF_TOKEN}' \
  -e HF_HOME=/tmp/hf \
  -e HUGGINGFACE_HUB_CACHE=/tmp/hf/hub \
  -e TRANSFORMERS_CACHE=/tmp/hf/hub \
  -v '${REMOTE_OUTPUT_DIR}:/output' \
  -p ${PORT}:${PORT} \
  -w '${REMOTE_WORKDIR}' \
  '${IMAGE}' \
  vllm serve \
    '${MODEL}' \
    --omni \
    --port ${PORT}")

echo "==> Container started: ${CONTAINER_ID}"

echo "==> Waiting for API server to be ready (checking /health endpoint)..."
echo "    (This may take 5-10 minutes for model download + loading...)"
for i in {1..300}; do
  # Check if container is still running
  if ! ssh "${REMOTE}" "podman ps -q --filter name=${CONTAINER_NAME}" | grep -q .; then
    echo "ERROR: Container exited unexpectedly!"
    echo "==> Container logs:"
    ssh "${REMOTE}" "podman logs ${CONTAINER_NAME} 2>&1 | tail -100"
    echo ""
    echo "==> Cleaning up..."
    ssh "${REMOTE}" "podman rm -f ${CONTAINER_NAME} 2>/dev/null || true"
    exit 1
  fi

  if ssh "${REMOTE}" "curl -s http://127.0.0.1:${PORT}/health >/dev/null 2>&1"; then
    echo "✓ API server is ready!"
    break
  fi

  if [[ $i -eq 300 ]]; then
    echo "ERROR: API server did not become ready in 5 minutes (but container is still running)"
    echo "==> Container logs:"
    ssh "${REMOTE}" "podman logs ${CONTAINER_NAME} 2>&1 | tail -100"
    echo ""
    echo "==> Stopping container..."
    ssh "${REMOTE}" "podman stop ${CONTAINER_NAME}"
    exit 1
  fi
  # Print progress every 10 seconds
  if [[ $((i % 10)) -eq 0 ]]; then
    echo "  Waiting... ($i/300 - $((i / 60))m $((i % 60))s)"
  fi
  sleep 1
done

# ==============================================================================
# Test video generation API
# ==============================================================================
echo ""
echo "==> Testing /v1/videos endpoint"
echo "    Prompt: ${TEST_PROMPT}"
echo "    Duration: ${TEST_SECONDS}s @ ${TEST_FPS}fps (size: ${TEST_SIZE})"
echo ""

TEST_FILE="test_ltx2_api_$(date +%s).json"
VIDEO_FILE="test_ltx2_output_$(date +%s).mp4"

ssh "${REMOTE}" bash -c "set -euo pipefail
  echo '==> Sending video generation request...'
  podman exec ${CONTAINER_NAME} curl -s -X POST http://localhost:${PORT}/v1/videos \
    -F 'prompt=${TEST_PROMPT}' \
    -F 'seconds=${TEST_SECONDS}' \
    -F 'size=${TEST_SIZE}' \
    > '${TEST_FILE}'

  echo '==> Checking API response...'
  if ! jq -e '.data[0].b64_json' '${TEST_FILE}' >/dev/null 2>&1; then
    echo 'ERROR: API response missing b64_json field'
    echo 'Response:'
    cat '${TEST_FILE}'
    exit 1
  fi

  echo '==> Decoding base64 video...'
  jq -r '.data[0].b64_json' '${TEST_FILE}' | base64 -d > '${VIDEO_FILE}'

  echo '==> Validating video file...'
  FILE_SIZE=\$(stat -c%s '${VIDEO_FILE}')
  FILE_TYPE=\$(file -b '${VIDEO_FILE}')

  echo \"    File size: \${FILE_SIZE} bytes\"
  echo \"    File type: \${FILE_TYPE}\"

  if [[ \$FILE_SIZE -lt 1000 ]]; then
    echo 'ERROR: Video file too small (< 1KB), likely invalid'
    exit 1
  fi

  if ! echo \"\${FILE_TYPE}\" | grep -qi 'mp4\|iso.*media\|mpeg'; then
    echo 'WARNING: File type does not look like MP4, but continuing...'
  fi

  echo '✓ Video file appears valid!'
  echo ''
  echo 'Files created:'
  echo \"  - ${TEST_FILE} (API response)\"
  echo \"  - ${VIDEO_FILE} (decoded video)\"
"

# ==============================================================================
# Check logs for LTX2-specific messages
# ==============================================================================
echo ""
echo "==> Checking server logs for LTX2 parameter adjustments..."
ssh "${REMOTE}" "podman logs ${CONTAINER_NAME} 2>&1 | grep -i 'ltx2' | tail -10 || echo '(No LTX2 log messages found)'"

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "========================================================================"
echo "✓ SUCCESS! LTX-2 API test completed"
echo "========================================================================"
echo ""
echo "Remote files:"
ssh "${REMOTE}" "ls -lh ${TEST_FILE} ${VIDEO_FILE}"
echo ""
echo "To download the video locally:"
echo "  scp ${REMOTE}:~/${VIDEO_FILE} ."
echo ""
echo "To view container logs:"
echo "  ssh ${REMOTE} podman logs -f ${CONTAINER_NAME}"
echo ""

if [[ "$KEEP_RUNNING" == "true" ]]; then
  echo "Container is still running (--keep-running enabled)"
  echo "API endpoint: http://${REMOTE}:${PORT}/v1/videos"
  echo ""
  echo "To stop later:"
  echo "  $0 --rev ${REV} --stop-only"
else
  echo "==> Stopping and removing container"
  ssh "${REMOTE}" "podman stop ${CONTAINER_NAME} && podman rm ${CONTAINER_NAME}"
  echo "✓ Container stopped and removed"
fi

echo ""
echo "Done! 🎉"

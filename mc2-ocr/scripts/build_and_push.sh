#!/usr/bin/env bash
# Unattended: build the MC2 image locally, check the grader's hard gates, push to
# Docker Hub with retries (for unstable connections). Needs `docker login` done first.
# Usage: bash scripts/build_and_push.sh [image:tag]   (default awaisbari0408/jin-mc2:v1)
set -uo pipefail
cd "$(dirname "$0")/.."
IMG=${1:-awaisbari0408/jin-mc2:v1}
BASE=rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
say() { echo "[$(date +%H:%M:%S)] $*"; }
fail() { say "FAILED: $*"; exit 1; }

[ -f models/Qwen3-VL-8B-Instruct/model-00004-of-00004.safetensors ] || fail "weights missing - run scripts/fetch_model.sh"
rm -rf models/Qwen3-VL-8B-Instruct/.cache

say "build (classic builder: streams the weights sequentially, much faster on a hard drive)"
DOCKER_BUILDKIT=0 docker build -t "$IMG" . || fail "docker build"

say "check: torch is still the ROCm build"
docker run --rm --entrypoint python3 "$IMG" -c "import torch; v=torch.__version__; print(v); assert '+rocm' in v" || fail "torch is not ROCm"
say "check: size under 60 GiB uncompressed"
size=$(docker image inspect "$IMG" --format '{{.Size}}'); say "$((size/1024/1024/1024)) GiB"
[ "$size" -lt $((60*1024*1024*1024)) ] || fail "image too large"
say "check: built on the mandated base (layer identity)"
base=$(docker image inspect "$BASE" --format '{{json .RootFS.Layers}}' | tr -d '[]"' | tr ',' '\n')
ours=$(docker image inspect "$IMG" --format '{{json .RootFS.Layers}}' | tr -d '[]"' | tr ',' '\n' | head -n "$(echo "$base" | wc -l)")
[ "$base" = "$ours" ] || fail "base layers differ"
say "check: weights and code inside the image"
docker run --rm --entrypoint sh "$IMG" -c "ls -la /models/Qwen3-VL-8B-Instruct/*.safetensors /app/app.py /app/server.py" || fail "files missing"

for i in $(seq 1 20); do
  say "push attempt $i"
  docker push "$IMG" && { say "PUSHED $(docker image inspect "$IMG" --format '{{index .RepoDigests 0}}')"; say "ALL DONE"; exit 0; }
  sleep 60
done
fail "push failed 20 times"

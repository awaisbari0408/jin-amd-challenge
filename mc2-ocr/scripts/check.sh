#!/usr/bin/env bash
# Run on a machine with an AMD GPU and Docker. Mirrors the grader's hard gates.
# Usage: scripts/check.sh <image:tag>
set -euo pipefail
IMG=${1:?usage: scripts/check.sh image:tag}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
NAME=mc2-check

echo "== 1. torch must still be the ROCm build"
docker run --rm --entrypoint python3 "$IMG" -c "import torch; v=torch.__version__; print(v); assert '+rocm' in v"

echo "== 2. uncompressed image size (limit 60 GiB)"
docker image inspect "$IMG" --format '{{.Size}}' | awk '{printf "%.1f GiB\n", $1/1073741824; if ($1 > 60*1073741824) {print "OVER LIMIT"; exit 1}}'

echo "== 3. start container and wait for ready (limit 10 min)"
docker rm -f $NAME >/dev/null 2>&1 || true
docker run -d --name $NAME --device=/dev/kfd --device=/dev/dri --group-add video \
  --cap-drop DAC_OVERRIDE -v "$ROOT/samples:/app/input:ro" "$IMG" >/dev/null
start=$(date +%s)
until docker exec $NAME test -f /tmp/ocr.ready; do
  sleep 5; (( $(date +%s) - start > 600 )) && { echo "STARTUP OVER 10 MIN"; docker logs --tail 50 $NAME; exit 1; }
done
echo "ready after $(( $(date +%s) - start ))s"

echo "== 4. run every sample (limit 30s each, 10 min total)"
for f in "$ROOT"/samples/image_*; do
  b=$(basename "$f"); t0=$(date +%s%N)
  timeout 30 docker exec $NAME python3 /app/app.py --input-image /app/input/$b || echo "TIMEOUT/ERROR on $b"
  printf '%-14s %d ms\n' "$b" $(( ($(date +%s%N) - t0) / 1000000 ))
done
docker exec $NAME sh -c 'command -v amd-smi >/dev/null && amd-smi metric --mem | grep -i USED_VRAM || rocm-smi --showmeminfo vram' || true

echo "== 5. score"
rm -rf "$ROOT/.out" && docker cp $NAME:/app/output "$ROOT/.out"
python3 "$ROOT/scripts/score.py" "$ROOT/.out"
docker logs --tail 15 $NAME
docker rm -f $NAME >/dev/null

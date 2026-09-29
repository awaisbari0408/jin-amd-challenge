#!/usr/bin/env bash
# GPU test WITHOUT Docker, for the notebooks.amd.com/hackathon JupyterLab terminal.
# Usage (from the mc2-ocr folder):  bash scripts/notebook_test.sh
# Optional: MODEL_ID=Qwen/Qwen3-VL-4B-Instruct bash scripts/notebook_test.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
export OCR_SOCKET=/tmp/ocr.sock OCR_READY_FILE=/tmp/ocr.ready
# Keep model downloads on persistent storage so they survive session restarts.
for d in /persistent /workspace; do [ -d "$d" ] && [ -w "$d" ] && { export HF_HOME="$d/hf"; break; }; done
echo "HF_HOME=${HF_HOME:-default}  MODEL_ID=${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"

python3 -c "import torch, torchvision; print(f'torch=={torch.__version__}\ntorchvision=={torchvision.__version__}')" > /tmp/constraints.txt
pip install -q -c /tmp/constraints.txt -r requirements.txt
python3 -c "import torch; assert '+rocm' in torch.__version__, torch.__version__; print('torch OK', torch.__version__)"

pkill -f "app/server.py" 2>/dev/null || true; rm -f /tmp/ocr.ready
( cd app && python3 server.py ) > /tmp/ocr-server.log 2>&1 &
echo "server starting (first run downloads the model; log: /tmp/ocr-server.log)"
start=$(date +%s)
until [ -f /tmp/ocr.ready ]; do
  sleep 5; kill -0 $! 2>/dev/null || { echo "server died:"; tail -30 /tmp/ocr-server.log; exit 1; }
  printf '.'; done
echo; echo "ready after $(( $(date +%s) - start ))s  (grader limit incl. startup: 600s)"

rm -rf .out && mkdir -p .out
for f in samples/image_*; do
  t0=$(date +%s%N); python3 app/app.py --input-image "$f" --output-dir .out
  printf '%-14s %d ms\n' "$(basename "$f")" $(( ($(date +%s%N) - t0) / 1000000 ))
done
command -v amd-smi >/dev/null && amd-smi metric --mem 2>/dev/null | grep -i USED_VRAM || rocm-smi --showmeminfo vram || true
python3 scripts/score.py .out
echo "Server still running (pid $!). Stop it with: pkill -f app/server.py"

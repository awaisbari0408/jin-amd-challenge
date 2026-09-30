#!/usr/bin/env bash
# GPU test WITHOUT Docker, for the notebooks.amd.com/hackathon JupyterLab terminal.
# Usage (from the mc3-rag folder):  bash scripts/notebook_test.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
export RAG_SOCKET=/tmp/rag.sock RAG_READY_FILE=/tmp/rag.ready
export MC3_INDEX_DIR="$ROOT/.index" MC3_OUTPUT_DIR="$ROOT/.out"
for d in /persistent /workspace; do [ -d "$d" ] && [ -w "$d" ] && { export HF_HOME="$d/hf"; break; }; done
echo "HF_HOME=${HF_HOME:-default} (the MC2 test already downloaded the model here)"

python3 -c "import torch, torchvision; print(f'torch=={torch.__version__}\ntorchvision=={torchvision.__version__}')" > /tmp/constraints.txt
pip install -q -c /tmp/constraints.txt -r requirements.txt -r requirements-rag.txt
python3 -c "import torch; assert '+rocm' in torch.__version__, torch.__version__; print('torch OK', torch.__version__)"

# A copy of the sample corpus with the two cases a zip cannot carry.
rm -rf /tmp/mc3-corpus && cp -r kit/mc3-corpus /tmp/mc3-corpus
mkdir -p /tmp/mc3-corpus/archive && chmod 000 /tmp/mc3-corpus/vendor/internal_audit.txt

pkill -f "mc3-rag/app/server.py" 2>/dev/null || pkill -f "app/server.py" 2>/dev/null || true
rm -f "$RAG_READY_FILE"
( cd app && exec python3 "$ROOT/app/server.py" ) > /tmp/rag-server.log 2>&1 &
SERVER=$!
start=$(date +%s)
until [ -f "$RAG_READY_FILE" ]; do
  sleep 5; kill -0 $SERVER 2>/dev/null || { echo "server died:"; tail -30 /tmp/rag-server.log; exit 1; }
  printf '.'; done
echo; echo "model ready after $(( $(date +%s) - start ))s"

t0=$(date +%s); python3 app/app.py --index /tmp/mc3-corpus
echo "startup incl. index: $(( $(date +%s) - start ))s  (limit 600s; index pass alone $(( $(date +%s) - t0 ))s)"

rm -rf .out && mkdir -p .out
python3 - <<'PY'
import json, subprocess, time
qs = json.load(open("kit/sample-questions.json"))["queries"]
for q in qs:
    t = time.time()
    subprocess.run(["python3", "app/app.py", "--corpus", "/tmp/mc3-corpus", "--query-id", f"query_{q['n']:02d}", "--query", q["query"]], check=True)
    print(f"query_{q['n']:02d}  {time.time()-t:5.1f}s  (limit 30s)")
PY
command -v amd-smi >/dev/null && amd-smi metric --mem 2>/dev/null | grep -i USED_VRAM || rocm-smi --showmeminfo vram || true
python3 scripts/score.py .out
echo "Server log: /tmp/rag-server.log   Stop the server: kill $SERVER"

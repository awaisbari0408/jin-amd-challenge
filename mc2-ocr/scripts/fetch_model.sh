#!/usr/bin/env bash
# Download the model weights into ./models before `docker build`. Resumable: if the
# connection drops, it retries and continues where it stopped.
set -uo pipefail
cd "$(dirname "$0")/.."
command -v hf >/dev/null || pip3 install --user -q "huggingface_hub[hf_xet]"
HF=$(command -v hf || echo "$HOME/.local/bin/hf")
until "$HF" download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct; do
  echo "--- connection dropped, resuming in 15s ---"; sleep 15
done
echo "Model ready in models/Qwen3-VL-8B-Instruct"

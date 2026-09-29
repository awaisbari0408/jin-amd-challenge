# The FINAL stage must start from the mandated base (checked by layer identity).
# Never squash or flatten this image.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

ENV HF_HOME=/models \
    MODEL_ID=Qwen/Qwen3-VL-8B-Instruct \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt /app/requirements.txt

# Pin the base image's torch stack so pip cannot swap in a CUDA build
# (the organisers' "trap 2"), then fail the build if it happened anyway.
RUN python3 -c "import torch, torchvision; print(f'torch=={torch.__version__}\ntorchvision=={torchvision.__version__}')" > /tmp/constraints.txt \
 && cat /tmp/constraints.txt \
 && pip install --no-cache-dir -c /tmp/constraints.txt -r /app/requirements.txt \
 && python3 -c "import torch; v=torch.__version__; print('torch', v); assert '+rocm' in v, v"

# Optional: bake the weights into the image (faster, reliable startup; ~17 GB larger).
# docker build --build-arg BAKE_MODEL=1 ...
ARG BAKE_MODEL=0
RUN if [ "$BAKE_MODEL" = "1" ]; then \
      python3 -c "from huggingface_hub import snapshot_download as s; import os; s(os.environ['MODEL_ID'])"; \
    fi

COPY app/ /app/
RUN mkdir -p /app/output /app/input

# Healthy only once the model is loaded and warmed up.
HEALTHCHECK --interval=10s --timeout=3s --start-period=600s --retries=3 \
  CMD test -f /tmp/ocr.ready

# The resident server keeps the container running and holds the model.
CMD ["python3", "/app/server.py"]

# The FINAL stage must start from the mandated base (checked by layer identity).
# Never squash or flatten this image.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt /app/requirements.txt

# Pin the base image's torch stack so pip cannot swap in a CUDA build
# (the organisers' "trap 2"), then fail the build if it happened anyway.
RUN python3 -c "import torch, torchvision; print(f'torch=={torch.__version__}\ntorchvision=={torchvision.__version__}')" > /tmp/constraints.txt \
 && cat /tmp/constraints.txt \
 && pip install --no-cache-dir -c /tmp/constraints.txt -r /app/requirements.txt \
 && python3 -c "import torch; v=torch.__version__; print('torch', v); assert '+rocm' in v, v"

# Model weights are baked in (see scripts/fetch_model.sh), so startup never depends on
# network speed. One layer per weight shard: a failed push only retries that shard.
ARG M=models/Qwen3-VL-8B-Instruct
COPY ${M}/*.json ${M}/*.txt /models/Qwen3-VL-8B-Instruct/
COPY ${M}/model-00001-of-00004.safetensors /models/Qwen3-VL-8B-Instruct/
COPY ${M}/model-00002-of-00004.safetensors /models/Qwen3-VL-8B-Instruct/
COPY ${M}/model-00003-of-00004.safetensors /models/Qwen3-VL-8B-Instruct/
COPY ${M}/model-00004-of-00004.safetensors /models/Qwen3-VL-8B-Instruct/

COPY app/ /app/
RUN mkdir -p /app/output /app/input

# Healthy only once the model is loaded and warmed up.
HEALTHCHECK --interval=10s --timeout=3s --start-period=600s --retries=3 \
  CMD test -f /tmp/ocr.ready

# The resident server keeps the container running and holds the model.
CMD ["python3", "/app/server.py"]

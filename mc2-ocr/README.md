# J-I-N — Mini Challenge 2 (OCR) on AMD ROCm

Reads licence plates and road signs with **Qwen3-VL-8B-Instruct** (BF16, Hugging Face
Transformers) on the mandated `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`
base image.

## How it works

```
container CMD ──► app/server.py   loads the model ONCE, warms up the GPU, listens on /tmp/ocr.sock
grader ─exec──► app/app.py        stdlib-only client: sends the image path, writes
                                  /app/output/<name>_output.json  (always, even on failure)
```

Loading the model per call would blow the 30 s per-image limit, so the model lives in a
resident process and `app.py` is a thin client. This is the same design used by the
strongest published MC2 entries.

- `app/ocr_rules.py`: the prompt, plus clean-up of model output (drops US state banners and
  slogans only when what's left looks like a plate; keeps Chinese province characters;
  joins multi-line signs with single spaces; normalizes full-width characters).
- `Dockerfile`: pins the base image's torch/torchvision with a pip constraints file, so pip
  cannot swap in a CUDA build (the organisers' "trap 2"), and the build fails if it
  happens anyway. No squashing. Healthcheck turns healthy only after warm-up.

## Status

| Check | Status |
|---|---|
| Answer-cleaning rules on all 10 sample answers | ✅ passing (CPU) |
| `app.py` contract: file naming, JSON shape, PNG/JPEG/TIFF, dead-server fallback | ✅ passing (CPU) |
| Same tests inside the real base image (Python 3.14) | ✅ passing, offline |
| Base image torch is `2.13.0+rocm10.0.0` | ✅ confirmed |
| Model accuracy on the 10 samples | ⏳ needs a GPU — `scripts/notebook_test.sh` |
| Docker build + all hard gates | ⏳ needs a build + GPU — `scripts/check.sh` |

## Next steps

1. **GPU test in the hackathon notebook** (no Docker needed). Upload this folder, then in a
   JupyterLab terminal:
   ```bash
   bash scripts/notebook_test.sh
   ```
   It installs deps, starts the server, runs all 10 samples, and prints timing, VRAM and a
   score out of 200. The first run downloads the model (~17 GB) on AMD's network, not yours.
2. **Build the image.** Model weights are downloaded at container start by default. To bake
   them in instead (slower build, larger image, more reliable startup):
   `docker build --build-arg BAKE_MODEL=1 -t <you>/jin-mc2:v1 .`
3. **Push to a public registry**, then put the image reference in the **"Mini Challenge 2
   Image"** field of the J-I-N submission. Don't commit the image reference to a public repo.

## Open questions for the organisers

- Does the grader wait for the container healthcheck, or start calling `app.py` right after
  the container starts? This decides whether the weights must be baked in.

## Files

```
app/app.py            grader entry point (thin client)
app/server.py         resident model server
app/ocr_rules.py      prompt + output clean-up + grader normalization
Dockerfile            final stage FROM the mandated base
samples/              the 10 official samples, in the formats the spec lists, + expected.json
scripts/score.py      scores an output folder with the grader's normalization
scripts/notebook_test.sh   GPU test without Docker
scripts/check.sh      full hard-gate check for a built image on an AMD GPU machine
tests/test_contract.py     CPU-only tests
```

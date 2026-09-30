"""Resident OCR server: loads the vision-language model once at container start, then
answers requests from app.py over a unix socket. Started by the container's CMD."""
import json
import os
import socket
import sys
import threading
import time
import traceback

from PIL import Image

from ocr_rules import PROMPT, clean

SOCKET_PATH = os.environ.get("OCR_SOCKET", "/tmp/ocr.sock")
READY_FILE = os.environ.get("OCR_READY_FILE", "/tmp/ocr.ready")
HUB_MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
BAKED_MODEL_DIR = os.environ.get("BAKED_MODEL_DIR", "/models/Qwen3-VL-8B-Instruct")
# Prefer weights baked into the image; fall back to the Hugging Face Hub (notebook tests).
if os.path.isfile(os.path.join(BAKED_MODEL_DIR, "config.json")):
    MODEL_ID = BAKED_MODEL_DIR
    os.environ["HF_HUB_OFFLINE"] = "1"  # never touch the network at grading time
else:
    MODEL_ID = HUB_MODEL_ID
MAX_PIXELS = int(os.environ.get("OCR_MAX_PIXELS", str(1280 * 1280)))
MIN_SIDE = int(os.environ.get("OCR_MIN_SIDE", "448"))
MAX_NEW_TOKENS = int(os.environ.get("OCR_MAX_NEW_TOKENS", "48"))


def log(msg):
    print(f"[ocr-server {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def prepare(path):
    """Open PNG/JPEG/TIFF (first frame), convert to RGB, keep size within budget."""
    img = Image.open(path)
    img.seek(0)
    img = img.convert("RGB")
    w, h = img.size
    if w * h > MAX_PIXELS:
        scale = (MAX_PIXELS / (w * h)) ** 0.5
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    elif min(w, h) < MIN_SIDE:
        scale = MIN_SIDE / min(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


class Reader:
    def __init__(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        log(f"torch {torch.__version__}, GPU available: {torch.cuda.is_available()}")
        log(f"loading {MODEL_ID}")
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
        ).eval()
        log(f"model loaded in {time.time() - t0:.1f}s")
        self.lock = threading.Lock()  # one GPU, one request at a time

    def read(self, path):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": prepare(path)},
            {"type": "text", "text": PROMPT},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        with self.lock, self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        raw = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        text = clean(raw)
        return {"text": text, "confidence": 0.9 if text else 0.0}


def warm_up(reader):
    path = "/tmp/warmup.png"
    Image.new("RGB", (640, 320), "white").save(path)
    t0 = time.time()
    reader.read(path)
    log(f"warm-up inference took {time.time() - t0:.1f}s")


def handle(conn, reader):
    with conn:
        try:
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = conn.recv(65536)
                if not chunk:
                    break
                buf += chunk
            req = json.loads(buf.decode() or "{}")
            t0 = time.time()
            result = reader.read(req["image"])
            log(f"{os.path.basename(req['image'])} -> {result['text']!r} ({time.time() - t0:.2f}s)")
        except Exception:
            log(traceback.format_exc())
            result = {"text": "", "confidence": 0.0}
        conn.sendall((json.dumps(result, ensure_ascii=False) + "\n").encode())


def main():
    for p in (SOCKET_PATH, READY_FILE):
        if os.path.exists(p):
            os.remove(p)
    reader = Reader()
    warm_up(reader)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o777)
    srv.listen(16)
    open(READY_FILE, "w").write("ready\n")
    log(f"ready on {SOCKET_PATH}")
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=handle, args=(conn, reader), daemon=True).start()


if __name__ == "__main__":
    main()

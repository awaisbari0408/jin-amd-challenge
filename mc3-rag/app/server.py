"""Resident model server for MC3, started by the container CMD.

The harness runs app.py as a NEW PROCESS for the index pass and for every question,
so the model lives here, loaded once, and app.py talks to it over a unix socket.
One model (Qwen3-VL-8B-Instruct) does both jobs: reading images during indexing,
and answering questions from retrieved text.
"""
import json
import os
import socket
import sys
import threading
import time
import traceback

SOCKET_PATH = os.environ.get("RAG_SOCKET", "/tmp/rag.sock")
READY_FILE = os.environ.get("RAG_READY_FILE", "/tmp/rag.ready")
INDEX_FILE = os.path.join(os.environ.get("MC3_INDEX_DIR", "/app/index"), "index.json")
HUB_MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
BAKED_MODEL_DIR = os.environ.get("BAKED_MODEL_DIR", "/models/Qwen3-VL-8B-Instruct")
if os.path.isfile(os.path.join(BAKED_MODEL_DIR, "config.json")):
    MODEL_ID = BAKED_MODEL_DIR
    os.environ["HF_HUB_OFFLINE"] = "1"  # the graded container has no network at all
else:
    MODEL_ID = HUB_MODEL_ID

OCR_PROMPT = """Transcribe ALL text visible in this image, exactly as printed.
For labels, forms, tables and diagrams, write one line per item and keep each label
together with its value, e.g. "BOARD REVISION: REV-C2" or "B14: THERM_ALERT#"
(for a connector or pinout, pair each pin with the signal drawn next to it).
Output only the transcription."""


def log(msg):
    print(f"[rag-server {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


class Model:
    def __init__(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        log(f"torch {torch.__version__}, GPU available: {torch.cuda.is_available()}")
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
        ).eval()
        log(f"loaded {MODEL_ID} in {time.time() - t0:.1f}s")
        self.lock = threading.Lock()

    def generate(self, content, max_new_tokens):
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        with self.lock, self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    def text(self, prompt):
        return self.generate([{"type": "text", "text": prompt}], max_new_tokens=160)

    def read_image(self, path):
        from PIL import Image
        img = Image.open(path)
        img.seek(0)
        img = img.convert("RGB")
        w, h = img.size
        if w * h > 1600 * 1600:  # keep large scans within budget
            s = (1600 * 1600 / (w * h)) ** 0.5
            img = img.resize((int(w * s), int(h * s)))
        return self.generate([{"type": "image", "image": img}, {"type": "text", "text": OCR_PROMPT}],
                             max_new_tokens=400)


class Answerer:
    """Keeps the QA object for the current index; rebuilds if the index file changes."""

    def __init__(self, model):
        self.model, self.qa, self.mtime = model, None, None

    def get(self):
        from rag import QA
        m = os.path.getmtime(INDEX_FILE)
        if self.qa is None or m != self.mtime:
            with open(INDEX_FILE, encoding="utf-8") as f:
                records = json.load(f)["records"]
            self.qa, self.mtime = QA(records, self.model.text), m
            log(f"loaded index: {len(records)} records")
        return self.qa


def handle(conn, model, answerer):
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
            if req.get("op") == "ocr":
                result = {"text": model.read_image(req["path"])}
            elif req.get("op") == "query":
                ans, cites, conf = answerer.get().answer(req["query"])
                result = {"answer": ans, "citations": cites, "confidence": conf}
                log(f"{req.get('query_id')}: {ans!r} {cites} ({time.time() - t0:.1f}s)")
            else:
                result = {"ok": True}
        except Exception:
            log(traceback.format_exc())
            result = {"error": "server error"}
        conn.sendall((json.dumps(result, ensure_ascii=False) + "\n").encode())


def main():
    for p in (SOCKET_PATH, READY_FILE):
        if os.path.exists(p):
            os.remove(p)
    model = Model()
    t0 = time.time()
    model.text("Reply with OK.")  # warm up kernels before reporting ready
    log(f"warm-up {time.time() - t0:.1f}s")
    answerer = Answerer(model)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o777)
    srv.listen(16)
    open(READY_FILE, "w").write("ready\n")
    log(f"ready on {SOCKET_PATH}")
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=handle, args=(conn, model, answerer), daemon=True).start()


if __name__ == "__main__":
    main()

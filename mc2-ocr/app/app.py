"""Grader entry point: python3 /app/app.py --input-image <path>

A thin, standard-library-only client. The model lives in server.py (started by the
container CMD), so this process starts in milliseconds and never loads weights.
It always writes an output file, even if the server fails, so one bad image cannot
break the run."""
import argparse
import json
import os
import socket
import sys
import time

SOCKET_PATH = os.environ.get("OCR_SOCKET", "/tmp/ocr.sock")
BUDGET_S = float(os.environ.get("OCR_CLIENT_BUDGET", "27"))  # grader allows 30s per image


def ask_server(image_path, deadline):
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("no answer from OCR server within budget")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(remaining)
                s.connect(SOCKET_PATH)
                s.sendall((json.dumps({"image": image_path}) + "\n").encode())
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = s.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                return json.loads(buf.decode())
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(0.25)  # server still starting


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-image", required=True)
    ap.add_argument("--output-dir", default="/app/output")
    args = ap.parse_args()

    deadline = time.time() + BUDGET_S
    result = {"text": "", "confidence": 0.0}
    try:
        answer = ask_server(os.path.abspath(args.input_image), deadline)
        result = {"text": str(answer.get("text", "")),
                  "confidence": float(answer.get("confidence", 0.0))}
    except Exception as e:
        print(f"app.py: {e}", file=sys.stderr)

    stem = os.path.splitext(os.path.basename(args.input_image))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{stem}_output.json")
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    os.replace(tmp, out_path)


if __name__ == "__main__":
    main()

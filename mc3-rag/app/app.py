"""Grader entry point for MC3. Two invocations, both as NEW processes:

    python3 /app/app.py --index /app/corpus
    python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."

The model lives in server.py (the container CMD); this script never loads weights.
A query ALWAYS writes a well-formed /app/output/<query-id>_output.json, falling back
to a refusal if anything goes wrong, so one failure cannot score the run zero."""
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

SOCKET_PATH = os.environ.get("RAG_SOCKET", "/tmp/rag.sock")
OUTPUT_DIR = Path(os.environ.get("MC3_OUTPUT_DIR", "/app/output"))
INDEX_DIR = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))
QUERY_BUDGET_S = float(os.environ.get("RAG_QUERY_BUDGET", "27"))  # harness allows 30
SERVER_WAIT_S = float(os.environ.get("RAG_SERVER_WAIT", "480"))   # startup budget is 600


def ask(req, deadline):
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("model server did not answer in time")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(remaining)
                s.connect(SOCKET_PATH)
                s.sendall((json.dumps(req) + "\n").encode())
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = s.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                return json.loads(buf.decode())
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(0.5)  # server still loading the model


def do_index(corpus):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import indexer

    start = time.time()
    server_deadline = start + SERVER_WAIT_S

    def read_image(path):
        r = ask({"op": "ocr", "path": os.path.abspath(path)}, max(time.time() + 90, server_deadline))
        if "text" not in r:
            raise RuntimeError(r.get("error", "no transcription"))
        return r["text"]

    records, report = indexer.build(str(corpus), read_image)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_DIR / "index.json.tmp"
    tmp.write_text(json.dumps({"records": records, "report": report}, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, INDEX_DIR / "index.json")
    for r in report:
        print(f"  {r['file']:50} {r['note']}  ({r['chars']} chars)")
    print(f"indexed {len(records)} records from {len(report)} files in {time.time() - start:.1f}s")


def do_query(query_id, query):
    result = {"answer": "", "citations": [], "confidence": 0.0}
    try:
        r = ask({"op": "query", "query": query, "query_id": query_id}, time.time() + QUERY_BUDGET_S)
        if "answer" in r:
            result = {"answer": str(r["answer"]), "citations": [str(c) for c in r.get("citations", [])],
                      "confidence": float(r.get("confidence", 0.0))}
    except Exception as e:  # noqa: BLE001 - always write a well-formed refusal
        print(f"app.py: {e}", file=sys.stderr)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{query_id}_output.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path)
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--query-id")
    ap.add_argument("--query")
    args = ap.parse_args()
    if args.index is not None:
        do_index(args.index)
        return 0
    if args.corpus is None or args.query is None or not args.query_id:
        ap.error("a query needs --corpus, --query-id and --query")
    do_query(args.query_id, args.query)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

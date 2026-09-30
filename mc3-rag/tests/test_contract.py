"""End-to-end CPU test: the real app.py (as new processes, like the harness) talking to
the real server socket code, with only the model replaced by a scripted stand-in.
Checks output paths, JSON shape, index persistence across processes, and scoring."""
import json, os, socket, subprocess, sys, tempfile, threading, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
TMP = tempfile.mkdtemp()
os.environ.update(RAG_SOCKET=os.path.join(TMP, "rag.sock"), MC3_INDEX_DIR=os.path.join(TMP, "index"),
                  MC3_OUTPUT_DIR=os.path.join(TMP, "output"))
sys.path.insert(0, os.path.join(ROOT, "app")); sys.path.insert(0, HERE)
import server
from test_pipeline import IMG, scripted

class FakeModel:
    text = staticmethod(scripted("precise"))
    def read_image(self, path): return IMG[os.path.basename(path)]

class Contract(unittest.TestCase):
    def test_index_then_ten_queries_score_200(self):
        srv = socket.socket(socket.AF_UNIX); srv.bind(os.environ["RAG_SOCKET"]); srv.listen(8)
        model = FakeModel(); answerer = server.Answerer(model)
        def loop():
            while True:
                c, _ = srv.accept()
                threading.Thread(target=server.handle, args=(c, model, answerer), daemon=True).start()
        threading.Thread(target=loop, daemon=True).start()
        app = os.path.join(ROOT, "app", "app.py"); corpus = os.path.join(ROOT, "kit", "mc3-corpus")
        r = subprocess.run([sys.executable, app, "--index", corpus], capture_output=True, text=True, env=os.environ)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("image transcription", r.stdout)
        qs = json.load(open(os.path.join(ROOT, "kit", "sample-questions.json")))["queries"]
        for q in qs:
            subprocess.run([sys.executable, app, "--corpus", corpus, "--query-id", f"query_{q['n']:02d}",
                            "--query", q["query"]], check=True, env=os.environ)
            out = json.load(open(os.path.join(os.environ["MC3_OUTPUT_DIR"], f"query_{q['n']:02d}_output.json")))
            self.assertEqual(set(out), {"answer", "citations", "confidence"})
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "score.py"), os.environ["MC3_OUTPUT_DIR"]],
                           capture_output=True, text=True)
        self.assertIn("Score: 200/200", r.stdout, r.stdout)

    def test_dead_server_still_writes_a_refusal(self):
        env = dict(os.environ, RAG_SOCKET=os.path.join(TMP, "none.sock"), RAG_QUERY_BUDGET="2")
        subprocess.run([sys.executable, os.path.join(ROOT, "app", "app.py"), "--corpus", "/x",
                        "--query-id", "query_99", "--query", "anything"], check=True, env=env)
        out = json.load(open(os.path.join(os.environ["MC3_OUTPUT_DIR"], "query_99_output.json")))
        self.assertEqual(out, {"answer": "", "citations": [], "confidence": 0.0})

if __name__ == "__main__":
    unittest.main(verbosity=2)

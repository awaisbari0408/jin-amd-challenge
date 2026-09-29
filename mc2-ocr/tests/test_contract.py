"""CPU-only tests: answer cleaning rules, and the app.py <-> server contract using a
fake server (no model, no GPU). Run: python3 tests/test_contract.py"""
import json, os, socket, subprocess, sys, tempfile, threading, time, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "..", "app")
sys.path.insert(0, APP)
from ocr_rules import clean, normalize


class Rules(unittest.TestCase):
    def check(self, raw, want):
        self.assertEqual(normalize(clean(raw)), normalize(want), f"{raw!r} -> {clean(raw)!r}")

    def test_banners_dropped_from_us_plates(self):
        self.check("CALIFORNIA\n7ABC123", "7ABC123")
        self.check("NEW YORK JHT 2951 EXCELSIOR", "JHT 2951")
        self.check("TEXAS 5XYZ891 THE LONE STAR STATE", "5XYZ891")

    def test_chinese_plate_kept_whole(self):
        self.check("京A·12345", "京 A·12345")
        self.check("沪Ｂ·８８８８８", "沪 B·88888")  # full-width chars normalized

    def test_signs_untouched(self):
        self.check("SPEED\nLIMIT\n65", "SPEED LIMIT 65")
        self.check("ROAD\nWORK\nAHEAD", "ROAD WORK AHEAD")
        self.check("STOP", "STOP")
        self.check("35", "35")

    def test_model_chatter_removed(self):
        self.check("Text: STOP", "STOP")
        self.check("<think>hmm</think>\n\"7ABC123\"", "7ABC123")
        self.check("```\nSTOP\n```", "STOP")


class Contract(unittest.TestCase):
    def run_app(self, sock, image, outdir):
        env = dict(os.environ, OCR_SOCKET=sock, OCR_CLIENT_BUDGET="3")
        t0 = time.time()
        subprocess.run([sys.executable, os.path.join(APP, "app.py"), "--input-image", image,
                        "--output-dir", outdir], env=env, check=True, capture_output=True)
        return time.time() - t0

    def test_writes_named_json_and_survives_dead_server(self):
        with tempfile.TemporaryDirectory() as d:
            sock, out = os.path.join(d, "s.sock"), os.path.join(d, "output")
            srv = socket.socket(socket.AF_UNIX); srv.bind(sock); srv.listen(4)
            def serve():
                for _ in range(3):
                    c, _ = srv.accept()
                    req = json.loads(c.recv(65536).decode())
                    c.sendall((json.dumps({"text": os.path.basename(req["image"]), "confidence": 0.5}) + "\n").encode())
                    c.close()
            threading.Thread(target=serve, daemon=True).start()
            for img in ("image_01.png", "image_03.jpg", "image_07.tif"):
                took = self.run_app(sock, os.path.join(d, img), out)
                self.assertLess(took, 2.0)
                res = json.load(open(os.path.join(out, img.rsplit(".", 1)[0] + "_output.json")))
                self.assertEqual(res, {"text": img, "confidence": 0.5})
            # No server at all: must still write a well-formed empty answer, within budget.
            took = self.run_app(os.path.join(d, "missing.sock"), os.path.join(d, "image_99.png"), out)
            self.assertLess(took, 5.0)
            self.assertEqual(json.load(open(os.path.join(out, "image_99_output.json"))),
                             {"text": "", "confidence": 0.0})


if __name__ == "__main__":
    unittest.main(verbosity=2)

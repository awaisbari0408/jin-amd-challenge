"""CPU-only tests of indexing and citation logic. The model is replaced by scripted
stand-ins that behave in different realistic ways; the citation SET must come out
right regardless. (Whether the real model picks the right VALUE needs a GPU.)"""
import json, os, re, shutil, stat, sys, tempfile, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "app"))
import indexer
from rag import QA, norm

KIT = os.path.join(ROOT, "kit")
IMG = {"backplane_pinout.png": "TQ-40 backplane connector - pin assignment\nB11: GND\nB12: SDA\nB13: SCL\nB14: THERM_ALERT#\nB15: PRSNT#\nB16: GND",
       "asset_label.jpg": "Orrery Systems\nMODEL: TQ-40\nBOARD REVISION: REV-C2\nSN: OS4-118823-77"}
QS = json.load(open(os.path.join(KIT, "sample-questions.json")))["queries"]


def blocks(prompt):
    ev = prompt.split("Evidence:\n", 1)[1].split("\n\nReply with", 1)[0]
    return [(int(m.group(1)), m.group(2)) for m in re.finditer(r"\[(\d+)\] file: [^\n]+\n(.*?)(?=\n\n\[\d+\] file: |\Z)", ev, re.S)]


def scripted(behaviour):
    def llm(prompt):
        q = re.search(r"Question: (.*)\n", prompt).group(1)
        want = next(x for x in QS if x["query"] == q)
        ans = want["expected_answer"]
        bl = blocks(prompt)
        if not ans:
            return '{"answer": "", "evidence": [], "lookup": ""}'
        hit = [n for n, t in bl if norm(ans) in norm(t)]
        if behaviour == "lookup" and want["n"] == 9 and "ORR-1847" not in "".join(t for _, t in bl if "fixed_in" in t):
            log = [n for n, t in bl if "ORR-1847" in t]
            return json.dumps({"answer": "", "evidence": log[:1], "lookup": "ORR-1847"})
        if not hit:
            return '{"answer": "", "evidence": [], "lookup": ""}'
        ev = [n for n, _ in bl] if behaviour == "select_all" else hit[:1]
        return json.dumps({"answer": ans, "evidence": ev, "lookup": ""})
    return llm


class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, cls.report = indexer.build(os.path.join(KIT, "mc3-corpus"), lambda p: IMG[os.path.basename(p)])

    def run_all(self, behaviour):
        qa = QA(self.records, scripted(behaviour))
        for q in QS:
            ans, cites, _ = qa.answer(q["query"])
            with self.subTest(q=q["n"], behaviour=behaviour):
                self.assertEqual(norm(ans), norm(q["expected_answer"]))
                self.assertEqual(sorted(cites), sorted(q["expected_citations"]))

    def test_model_cites_only_the_right_record(self):
        self.run_all("precise")

    def test_model_selects_every_record_shown(self):
        self.run_all("select_all")

    def test_model_asks_for_a_lookup_on_multi_hop(self):
        self.run_all("lookup")

    def test_hallucinated_value_is_refused(self):
        qa = QA(self.records, lambda p: '{"answer": "6412", "evidence": [1], "lookup": ""}')
        self.assertEqual(qa.answer("What is the unit price of the TQ-40 at 10,000 unit volume?"), ("", [], 0.0))

    def test_withdrawn_datasheet_never_used(self):
        qa = QA(self.records, lambda p: '{"answer": "105", "evidence": [1], "lookup": ""}')
        self.assertEqual(qa.answer("What is the maximum junction temperature of the TQ-40?")[0], "")


class Robustness(unittest.TestCase):
    def test_hostile_corpus_is_indexed_around(self):
        d = tempfile.mkdtemp()
        try:
            c = os.path.join(d, "corpus")
            shutil.copytree(os.path.join(KIT, "mc3-corpus"), c)
            os.makedirs(os.path.join(c, "archive"), exist_ok=True)  # empty dir
            locked = os.path.join(c, "vendor", "internal_audit.txt")
            os.chmod(locked, 0)
            os.makedirs(os.path.join(c, "zzz"), exist_ok=True)       # a readable file AFTER the bad ones
            open(os.path.join(c, "zzz", "late.txt"), "w").write("LATE-MARKER 42")
            records, report = indexer.build(c, lambda p: IMG[os.path.basename(p)])
            notes = {r["file"]: r["note"] for r in report}
            self.assertTrue(notes["vendor/supplier_agreement_ENCRYPTED.pdf"].startswith("skipped"))
            self.assertEqual(notes["vendor/telemetry_capture.dat"], "unknown type")
            if os.geteuid() != 0:
                self.assertTrue(notes["vendor/internal_audit.txt"].startswith("skipped"))
            self.assertIn("zzz/late.txt", {r["file"] for r in records})
            self.assertNotIn("vendor/supplier_agreement_ENCRYPTED.pdf", {r["file"] for r in records})
        finally:
            for root, dirs, files in os.walk(d):
                for f in files:
                    os.chmod(os.path.join(root, f), stat.S_IRUSR | stat.S_IWUSR)
            shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main(verbosity=2)

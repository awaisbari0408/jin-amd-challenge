"""Score /app/output-style results against samples/expected.json using the grader's normalization.
Usage: python3 scripts/score.py <output_dir> [samples_dir]"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from ocr_rules import normalize

out_dir = sys.argv[1]
samples = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "samples")
expected = json.load(open(os.path.join(samples, "expected.json"), encoding="utf-8"))
score = 0
for name, want in sorted(expected.items()):
    path = os.path.join(out_dir, os.path.splitext(name)[0] + "_output.json")
    try:
        got = json.load(open(path, encoding="utf-8")).get("text", "")
    except Exception as e:
        got = f"<missing: {e.__class__.__name__}>"
    ok = normalize(got) == normalize(want)
    score += 20 * ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:14}  want={want!r:20}  got={got!r}")
print(f"\nScore: {score}/200")

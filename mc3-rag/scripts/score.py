"""Score an output folder against kit/sample-questions.json with the grader's rules:
answer after normalization AND the exact citation set, or zero for that question.
Usage: python3 scripts/score.py <output_dir>"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from rag import norm

out = sys.argv[1]
qs = json.load(open(os.path.join(os.path.dirname(__file__), "..", "kit", "sample-questions.json")))["queries"]
total = 0
for q in qs:
    path = os.path.join(out, f"query_{q['n']:02d}_output.json")
    try:
        got = json.load(open(path, encoding="utf-8"))
        ans, cites = got["answer"], got["citations"]
    except Exception as e:
        ans, cites = f"<{e.__class__.__name__}>", None
    cites_n = sorted(c.split("/app/corpus/")[-1] for c in (cites or []))
    ok_a = norm(ans) in {norm(q["expected_answer"])} | {norm(a) for a in q.get("answer_aliases", [])}
    ok_c = cites_n == sorted(q["expected_citations"])
    total += 20 * (ok_a and ok_c)
    print(f"{'PASS' if ok_a and ok_c else 'FAIL'} Q{q['n']:<2} answer={'ok' if ok_a else 'WRONG'} "
          f"citations={'ok' if ok_c else 'WRONG'}  got={ans!r} {cites_n}")
print(f"\nScore: {total}/200")

"""Question answering over the index. The language model is injected as a plain
function `llm(prompt) -> str`, so this whole module is testable without a GPU.

Scoring rules this encodes:
- answer is the value only; citations are an EXACT set (necessity, not relevance)
- refuse with ("", []) when the corpus does not contain the answer
- withdrawn / superseded documents are never sources
"""
import json
import re

from retrieval import BM25, identifiers

_REFERENCE = re.compile(
    r"\b(?:against|ticket|tracked (?:as|in|under)|see|refer(?:ence)? to|ref\.?|incident|issue|bug)\b[^\n]{0,40}", re.I)

PROMPT = """You answer questions using ONLY the evidence records below. The products are
fictional, so anything you remember instead of reading here is wrong.

Question: {query}

Evidence:
{evidence}

Reply with ONE JSON object and nothing else:
{{"answer": "<value>", "evidence": [<record numbers you used>], "lookup": "<identifier or empty>"}}

Rules:
- "answer" is the value only: a number, part number, version, code, pin or quarter. No
  sentence, no explanation. Keep qualifiers that identify the value (Q3 FY27, REV-C2).
- Prefer current values over comments or notes describing old or previous values.
- If a record names an identifier (ticket, error code, part number) and the value asked for
  must be looked up under that identifier elsewhere, set "answer" to "" and put that
  identifier in "lookup".
- If the evidence does not state exactly what is asked, "answer" must be "". A related value
  for a different item, product or quantity is NOT the answer.
"""


def norm(s):
    """The grader's normalization: uppercase, no whitespace, no - . · _"""
    return re.sub(r"[-.·_]", "", re.sub(r"\s+", "", str(s or "").upper()))


def parse_reply(raw):
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S)
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return {"answer": "", "evidence": [], "lookup": ""}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        d = {}
        for key in ("answer", "lookup"):
            mm = re.search(rf'"{key}"\s*:\s*"([^"]*)"', m.group(0))
            d[key] = mm.group(1) if mm else ""
        d["evidence"] = [int(x) for x in re.findall(r"\d+", (re.search(r'"evidence"\s*:\s*\[([^\]]*)\]', m.group(0)) or [None, ""])[1])]
    ev = d.get("evidence") or []
    if not isinstance(ev, list):
        ev = [ev]
    return {"answer": clean_answer(d.get("answer", "")),
            "evidence": [int(x) for x in ev if str(x).strip().isdigit()],
            "lookup": str(d.get("lookup") or "").strip()}


def clean_answer(a):
    a = str(a or "").strip().strip("\"'`").strip()
    a = re.sub(r"^(the\s+)?(answer|value)\s+(is|=)\s*[:]?\s*", "", a, flags=re.I)
    a = a.rstrip(".").strip()
    if a.upper() in {"NONE", "N/A", "NA", "UNKNOWN", "NOT FOUND", "NULL"}:
        return ""
    return a


class QA:
    def __init__(self, records, llm, k=8, max_chars=9000):
        # Superseded / withdrawn documents are never evidence.
        self.records = [r for r in records if not r.get("superseded")]
        self.bm25 = BM25(self.records)
        self.llm, self.k, self.max_chars = llm, k, max_chars
        self._answer = ""

    def _ask(self, query, ids):
        blocks, used = [], 0
        for n, i in enumerate(ids, 1):
            r = self.records[i]
            block = f"[{n}] file: {r['file']}\n{r['text']}"
            if used + len(block) > self.max_chars:
                break
            blocks.append(block)
            used += len(block)
        reply = parse_reply(self.llm(PROMPT.format(query=query, evidence="\n\n".join(blocks))))
        # Map record numbers back to record indexes, ignoring anything out of range.
        reply["records"] = [ids[n - 1] for n in reply["evidence"] if 1 <= n <= len(blocks)]
        reply["shown"] = ids[:len(blocks)]
        return reply

    def _files_with(self, value, record_ids):
        """Files (in the given order) whose records contain the normalized value."""
        v, out = norm(value), []
        for i in record_ids:
            f = self.records[i]["file"]
            if v and v in norm(self.records[i]["text"]) and f not in out:
                out.append(f)
        return out

    def answer(self, query):
        first = self._ask(query, self.bm25.search(query, k=self.k))
        final, chain_files = first, []

        lookups = [first["lookup"]] if first["lookup"] else []
        if not first["answer"] and not lookups:
            # Deterministic fallback for multi-hop: identifiers in the top records that
            # the question itself does not mention.
            # Only identifiers the text presents as a REFERENCE ("logged against ORR-1847",
            # "see ticket ..."), never arbitrary part numbers: that would turn a correct
            # refusal into a guess.
            asked = {i.upper() for i in identifiers(query)}
            for i in first["shown"][:3]:
                for m in _REFERENCE.finditer(self.records[i]["text"]):
                    lookups += [x for x in identifiers(m.group(0)) if x.upper() not in asked]
            lookups = list(dict.fromkeys(lookups))[:3]

        if not first["answer"] and lookups:
            keep = first["records"] or first["shown"][:2]
            extra = []
            for term in lookups:
                extra += [i for i in self.bm25.search(term, k=4, exclude=set(keep) | set(extra))]
            second = self._ask(query, keep + extra)
            if second["answer"]:
                final = second
                # The file(s) that named the identifier are part of the chain.
                for term in lookups:
                    for f in self._files_with(term, keep):
                        if f not in chain_files:
                            chain_files.append(f)

        ans = final["answer"]
        if not ans:
            return "", [], 0.0
        # Grounding: the value must literally appear in a cited file, otherwise refuse.
        pool = final["records"] or final["shown"]
        src = [i for i in pool if norm(ans) in norm(self.records[i]["text"])]
        if not src:
            src = [i for i in final["shown"] if norm(ans) in norm(self.records[i]["text"])][:1]
        if not src:
            return "", [], 0.0
        self._answer = ans
        source_file = self.records[src[0]]["file"]
        src = [i for i in src if self.records[i]["file"] == source_file]
        citations = [source_file]
        seen = list(dict.fromkeys(first["shown"] + final["shown"]))
        for f in chain_files + self._linked_files(query, src, seen):
            if f not in citations:
                citations.append(f)
        return ans, citations, 0.8 if len(citations) == 1 else 0.7

    def _linked_files(self, query, source_ids, candidate_ids):
        """Multi-hop: a file is a necessary second source when it REFERS to an identifier
        ("incident logged against ORR-1847") that also keys the answer's own record, and
        the question does not already give that identifier. If the question names it,
        the answer file alone suffices and citing the other file would be an error."""
        asked = {x.upper() for x in identifiers(query)}
        keys = set()
        for i in source_ids:  # only the line(s) that hold the answer, not the whole chunk
            for line in self.records[i]["text"].splitlines():
                if self._answer and norm(self._answer) in norm(line):
                    keys |= {x.upper() for x in identifiers(line)}
        keys -= asked
        source_file = self.records[source_ids[0]]["file"]
        out = []
        for i in candidate_ids:
            f = self.records[i]["file"]
            if f == source_file or f in out:
                continue
            refs = set()
            for m in _REFERENCE.finditer(self.records[i]["text"]):
                refs |= {x.upper() for x in identifiers(m.group(0))}
            if refs & keys:
                out.append(f)
        return out

"""Lexical retrieval (BM25) over chunk records. No embedding model: identifiers such
as ORR-1847, E7731 and THERM_ALERT# must match exactly, which dense retrievers blur."""
import math
import re
from collections import Counter

_ID = re.compile(r"\b[A-Z][A-Z0-9]*[-_][A-Z0-9][A-Z0-9_#-]*\b|\b[A-Z]{1,4}\d{2,}[A-Z0-9]*\b")
_WORD = re.compile(r"[a-z0-9]+")
STOP = set("""a an the of in on at to for and or is are was were be by with what which who
when where how does do did from that this it as its into than then there their any all
""".split())


def identifiers(text):
    """Ticket ids, error codes, part numbers, pin/signal names: ORR-1847, E7731, B14."""
    return {m.group(0).rstrip("-_") for m in _ID.finditer(text or "")}


def tokens(text):
    text = text or ""
    words = [w for w in _WORD.findall(text.lower()) if w not in STOP]
    # Whole identifiers as extra tokens, so "ORR-1847" also matches as one unit.
    words += ["id:" + i.lower() for i in identifiers(text)]
    return words


class BM25:
    def __init__(self, records, k1=1.4, b=0.6):
        self.records = records
        self.docs = [tokens(r["file"].replace("/", " ").replace("_", " ") + "\n" + r["text"]) for r in records]
        self.tf = [Counter(d) for d in self.docs]
        n = len(self.docs) or 1
        self.avg = sum(len(d) for d in self.docs) / n
        df = Counter(t for d in self.docs for t in set(d))
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.k1, self.b = k1, b

    def search(self, query, k=8, exclude=()):
        q = tokens(query)
        scored = []
        for i, tf in enumerate(self.tf):
            if i in exclude:
                continue
            dl = len(self.docs[i]) or 1
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [i for _, i in scored[:k]]

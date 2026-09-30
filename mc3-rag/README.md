# J-I-N — Mini Challenge 3 (RAG with exact citations) on AMD ROCm

Answers questions about a folder of mixed documents (pdf, docx, xlsx, csv, txt/log, py,
png/jpg) with the **value only** and the **exact set of files** it came from, or refuses
with an empty answer when the corpus does not contain it.

## How it works

```
container CMD ─► app/server.py    Qwen3-VL-8B loaded ONCE; reads images, answers questions
harness ──────► app.py --index    parses every file once, asks the server to read images,
                                  writes /app/index/index.json            (startup budget)
harness ──────► app.py --query    thin client per question → /app/output/<id>_output.json
```

- **Parsing** ([app/parsers.py](app/parsers.py)): every file in its own try/except, so an
  unreadable, encrypted, unknown or empty entry never stops the walk. Type is decided by
  extension only. Spreadsheet, CSV and Word-table rows keep each value next to its column
  header. Documents whose title or filename marks them *withdrawn/superseded* are flagged.
- **Images**: transcribed once during `--index` by the same vision-language model, keeping
  labels paired with values (`B14: THERM_ALERT#`, `BOARD REVISION: REV-C2`).
- **Retrieval** ([app/retrieval.py](app/retrieval.py)): BM25 with identifiers (`ORR-1847`,
  `E7731`, `THERM_ALERT#`) kept as whole tokens. Superseded documents are never shown.
- **Answering** ([app/rag.py](app/rag.py)): the model returns the value, the records it
  used, and an optional identifier to look up (multi-hop). Then code, not the model,
  decides the citations:
  - **Grounding**: the value must literally appear in a cited file, or the answer is refused
    (catches values the model "remembers" but the corpus does not say).
  - **Necessity**: cite the file that holds the value, plus a second file only when it
    *refers to* an identifier that keys the answer's row and that the question itself does
    not give (e.g. a log line "incident logged against ORR-1847").

## Status

| Check | Status |
|---|---|
| Retrieval puts every expected file first for all 10 sample questions | ✅ (CPU) |
| Exact citation sets for all 10 samples under 3 different model behaviours | ✅ (CPU, scripted model) |
| Hallucinated value refused; withdrawn datasheet never used | ✅ (CPU) |
| Hostile corpus: locked, encrypted, unknown type, empty dir, files after them | ✅ (CPU) |
| Real `app.py` processes + real server socket, end to end | ✅ 200/200 with scripted model |
| **Real model on a GPU** | ⏳ `bash scripts/notebook_test.sh` in the hackathon notebook |
| Docker image | ⏳ shares MC2's layers; build after the MC2 image exists |

## Run

- **GPU test** (hackathon notebook, from `/workspace/jin-amd-challenge/mc3-rag`):
  `bash scripts/notebook_test.sh`, which reuses the model the MC2 test already downloaded.
- **CPU tests**: `python3 tests/test_pipeline.py && python3 tests/test_contract.py`
  (needs `pypdf python-docx openpyxl`).
- **Build**: `docker build -t <name>:v1 .` from this folder with the weights in
  `models/` (same files as `../mc2-ocr/models`, hard-linked, no extra disk space).
  Test it the way it is graded: `python3 kit/selfcheck.py <name>:v1 kit/mc3-corpus`,
  with no network.

`kit/` is the organisers' starter kit: the sample corpus, questions and self-check.

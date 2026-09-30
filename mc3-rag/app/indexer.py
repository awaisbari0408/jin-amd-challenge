"""Build the index: parse every readable file once, chunk it, and record what was
skipped. Called by `app.py --index`; image reading is injected so this runs without a GPU."""
import parsers


def build(corpus, read_image=None):
    records, report = [], []
    for rel in parsers.walk(corpus):
        text, note = parsers.parse_file(corpus, rel, read_image)
        report.append({"file": rel, "note": note, "chars": len(text or "")})
        if not text or not text.strip():
            continue
        superseded = parsers.is_superseded(rel, text)
        if note == "image transcription":
            text = "[text printed in this image]\n" + text.strip()
        for piece in parsers.chunk(text):
            records.append({"file": rel, "text": piece, "superseded": superseded})
    return records, report

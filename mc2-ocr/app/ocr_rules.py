"""Prompt and post-processing for MC2. Pure Python so it can be tested without a GPU."""
import re
import unicodedata

PROMPT = """Read the text in this image exactly as printed.

Rules:
- Licence plate: return ONLY the registration number. Leave out state or country names,
  slogans, websites, stickers and frame text.
- Chinese licence plate: the leading province character and letter ARE part of the
  registration number. Keep them, e.g. 京A·12345.
- Road sign: return every word and number printed on the sign, reading top to bottom,
  joined by single spaces, e.g. SPEED LIMIT 65 or ROAD WORK AHEAD. Do not add units or
  words that are not printed.
- Output only the characters. No labels, quotes, punctuation you cannot see, or explanation."""

US_STATES = [
    "ALABAMA", "ALASKA", "ARIZONA", "ARKANSAS", "CALIFORNIA", "COLORADO", "CONNECTICUT",
    "DELAWARE", "FLORIDA", "GEORGIA", "HAWAII", "IDAHO", "ILLINOIS", "INDIANA", "IOWA",
    "KANSAS", "KENTUCKY", "LOUISIANA", "MAINE", "MARYLAND", "MASSACHUSETTS", "MICHIGAN",
    "MINNESOTA", "MISSISSIPPI", "MISSOURI", "MONTANA", "NEBRASKA", "NEVADA", "NEW HAMPSHIRE",
    "NEW JERSEY", "NEW MEXICO", "NEW YORK", "NORTH CAROLINA", "NORTH DAKOTA", "OHIO",
    "OKLAHOMA", "OREGON", "PENNSYLVANIA", "RHODE ISLAND", "SOUTH CAROLINA", "SOUTH DAKOTA",
    "TENNESSEE", "TEXAS", "UTAH", "VERMONT", "VIRGINIA", "WASHINGTON", "WEST VIRGINIA",
    "WISCONSIN", "WYOMING", "DISTRICT OF COLUMBIA",
]
SLOGANS = [
    "THE LONE STAR STATE", "LONE STAR STATE", "SUNSHINE STATE", "THE EMPIRE STATE",
    "EMPIRE STATE", "EXCELSIOR", "GARDEN STATE", "LAND OF LINCOLN", "GRAND CANYON STATE",
    "THE GOLDEN STATE", "GOLDEN STATE", "LIVE FREE OR DIE", "FIRST IN FLIGHT",
    "FAMOUS POTATOES", "GREAT LAKES", "PURE MICHIGAN", "THE KEYSTONE STATE", "KEYSTONE STATE",
    "VISIT", "DMV",
]
# Longest first so "NEW YORK" wins over "YORK"-like partial matches.
_BANNERS = sorted(US_STATES + SLOGANS, key=len, reverse=True)
_LABEL = re.compile(r"^\s*(?:text|answer|plate|output|result|transcription)\s*[:：]\s*", re.I)
_PLATE = re.compile(r"^(?=.*\d)[A-Z0-9 ]{4,10}$")


def clean(raw: str) -> str:
    """Turn model output into the single-line answer the grader expects."""
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S)
    text = text.replace("```", "").strip().strip("\"'`")
    text = _LABEL.sub("", text)
    text = unicodedata.normalize("NFKC", text)  # full-width letters/digits -> ASCII
    text = " ".join(line.strip() for line in text.splitlines() if line.strip())
    text = re.sub(r"\s+", " ", text).strip()
    return strip_us_banner(text)


def strip_us_banner(text: str) -> str:
    """Safety net for US plates: drop state names and slogans, but only when what is
    left looks like a plate number. Road signs are never touched."""
    upper = text.upper()
    remainder = upper
    for banner in _BANNERS:
        remainder = re.sub(rf"\b{re.escape(banner)}\b", " ", remainder)
    remainder = re.sub(r"\s+", " ", remainder).strip()
    if remainder != upper and _PLATE.match(remainder):
        return remainder
    return text


def normalize(text: str) -> str:
    """The grader's normalization, used by our own scoring script."""
    text = (text or "").upper()
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[-.·_]", "", text)

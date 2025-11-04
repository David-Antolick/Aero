import json, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MANIFEST = ROOT / "ingest" / "manifest.json"
SOURCES = ROOT / "ingest" / "sources.json"

def load_source_map():
    """Load filename->id mapping from ingest/sources.json if present."""
    if not SOURCES.exists():
        return {}
    try:
        with open(SOURCES, "r", encoding="utf-8") as f:
            items = json.load(f)
        # map exact filename to given id
        return {it["filename"]: it["id"] for it in items if "filename" in it and "id" in it}
    except Exception:
        # If sources.json is malformed, fall back to empty map
        return {}


def infer_id_from_filename(fname: str) -> str:
    """
    Heuristics to infer a short, interpretable ID from a PDF filename when not in sources.json.

    Rules:
    - 14CFR_PartNN*.pdf -> CFRNN (e.g., 14CFR_Part91.pdf -> CFR91)
    - AIM*.pdf -> AIM
    - AC_00-<digits><letter>* -> AC00<digits><letter> (e.g., AC_00-45H_*.pdf -> AC0045H)
    - Otherwise -> uppercase of stem with non-alnum removed
    """
    stem = fname[:-4] if fname.lower().endswith(".pdf") else fname

    # 14CFR_PartNN
    m = re.match(r"^14CFR_Part(\d+)", stem, flags=re.IGNORECASE)
    if m:
        return f"CFR{m.group(1)}"

    # AIM...
    if stem.upper().startswith("AIM"):
        return "AIM"

    # AC_00-45H... or AC_00-6B...
    m = re.match(r"^AC_00-(\d+)([A-Z])", stem, flags=re.IGNORECASE)
    if m:
        digits = m.group(1)
        letter = m.group(2).upper()
        return f"AC00{digits}{letter}"

    # Fallback: compact uppercase alphanumerics
    compact = re.sub(r"[^0-9A-Za-z]+", "", stem).upper()
    return compact

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    src_map = load_source_map()

    entries = []
    for p in sorted(DATA_DIR.glob("*.pdf")):
        fname = p.name
        sid = src_map.get(fname) or infer_id_from_filename(fname)
        entries.append({"id": sid, "path": str(p.as_posix())})

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    print(f"[OK] Wrote {MANIFEST} with {len(entries)} entries")

if __name__ == "__main__":
    main()

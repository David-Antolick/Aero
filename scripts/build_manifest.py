import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MANIFEST = ROOT / "ingest" / "manifest.json"

# Simple mapping: choose short IDs per file name prefix
ID_MAP = {
    "Pilot_Handbook_of_Aeronautical_Knowledge": "PHAK",
    "Airplane_Flying_Handbook": "AFH",
    "Instrument_Procedures_Handbook": "IPH",
    "AIM_Basic": "AIM",
    "14CFR_Part61": "CFR61",
    "14CFR_Part91": "CFR91"
}

def guess_id(fname):
    for k, v in ID_MAP.items():
        if fname.startswith(k):
            return v
    # fallback: uppercase stem
    return fname.split(".pdf")[0].upper()

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(DATA_DIR.glob("*.pdf")):
        sid = guess_id(p.name)
        out.append({"id": sid, "path": str(p.as_posix())})
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] Wrote {MANIFEST} with {len(out)} entries")

if __name__ == "__main__":
    main()

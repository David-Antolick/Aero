import os, json, pathlib, sys
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from tqdm import tqdm
import math

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SRC = ROOT / "ingest" / "sources.json"

def download(url, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as r:
        # try to get file size from headers (may be missing)
        total_size = int(r.headers.get("Content-Length", 0))
        chunk_size = 8192
        num_bars = math.ceil(total_size / chunk_size) if total_size else None

        with open(out_path, "wb") as f, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {out_path.name}",
            leave=True
        ) as pbar:
            while True:
                chunk = r.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))

def main():
    if not SRC.exists():
        print(f"[ERR] Missing {SRC}")
        sys.exit(1)
    with open(SRC, "r", encoding="utf-8") as f:
        items = json.load(f)

    for it in items:
        url = it["url"]
        fname = it["filename"]
        out_path = DATA_DIR / fname
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[SKIP] {fname} already exists")
            continue
        try:
            print(f"[DL ] {it['id']}: {url} -> {out_path}")
            download(url, out_path)
            print(f"[OK ] {fname} ({out_path.stat().st_size} bytes)")
        except HTTPError as e:
            print(f"[FAIL] HTTP {e.code} for {url}")
        except URLError as e:
            print(f"[FAIL] URL error {e.reason} for {url}")

if __name__ == "__main__":
    main()

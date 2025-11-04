import os, json, pathlib, sys, time
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SRC = ROOT / "ingest" / "sources.json"

def _get_content_length(url: str) -> int:
    try:
        req = Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=30) as r:
            return int(r.headers.get("Content-Length", 0) or 0)
    except Exception:
        return 0


def download(url, out_path: pathlib.Path, max_retries: int = 5):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    expected_size = _get_content_length(url)
    if out_path.exists() and out_path.stat().st_size > 0:
        if expected_size and out_path.stat().st_size >= expected_size:
            print(f"[SKIP] {out_path.name} already complete ({out_path.stat().st_size} bytes)")
            return
        else:
            if expected_size and out_path.stat().st_size < expected_size:
                try:
                    os.replace(out_path, tmp_path)
                except OSError:
                    pass

    resume_pos = tmp_path.stat().st_size if tmp_path.exists() else 0

    total_size = expected_size or 0
    initial = resume_pos

    backoff = 1.0
    attempts = 0
    chunk_size = 8192

    while attempts < max_retries:
        attempts += 1
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            if resume_pos:
                headers["Range"] = f"bytes={resume_pos}-"
            req = Request(url, headers=headers)
            with urlopen(req, timeout=60) as r:
                mode = "ab" if resume_pos else "wb"
                with open(tmp_path, mode) as f, tqdm(
                    total=total_size if total_size else None,
                    initial=initial,
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

            final_size = tmp_path.stat().st_size
            if expected_size and final_size < expected_size:
                resume_pos = final_size
                initial = resume_pos
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue

            os.replace(tmp_path, out_path)
            return
        except (HTTPError, URLError) as e:
            print(f"[RETRY] Attempt {attempts}/{max_retries} for {out_path.name}: {getattr(e, 'reason', getattr(e, 'code', e))}")
        except Exception as e:
            print(f"[RETRY] Attempt {attempts}/{max_retries} for {out_path.name}: {e}")

        if tmp_path.exists():
            resume_pos = tmp_path.stat().st_size
            initial = resume_pos
        time.sleep(backoff)
        backoff = min(backoff * 2, 30)

    print(f"[FAIL] Exhausted retries for {out_path.name}")

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
        try:
            print(f"[DL ] {it['id']}: {url} -> {out_path}")
            download(url, out_path)
            if out_path.exists():
                print(f"[OK ] {fname} ({out_path.stat().st_size} bytes)")
        except HTTPError as e:
            print(f"[FAIL] HTTP {e.code} for {url}")
        except URLError as e:
            print(f"[FAIL] URL error {e.reason} for {url}")

if __name__ == "__main__":
    main()

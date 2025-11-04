# scripts/debug_docling.py
import sys, json
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from transformers import AutoTokenizer

def main(pdf_path, embed_model="BAAI/bge-small-en-v1.5", max_tokens=512, n=8):
    print(f"[INFO] Converting: {pdf_path}")
    conv = DocumentConverter()
    res = conv.convert(pdf_path)
    doc = res.document
    print("[INFO] Converted. Now chunking...")

    tok = AutoTokenizer.from_pretrained(embed_model)
    chunker = HybridChunker(tokenizer=tok, max_tokens=max_tokens, merge_peers=True)

    cnt = 0
    for ch in chunker.chunk(doc):
        cnt += 1
        text = getattr(ch, "text", None)
        text = (text or "").strip().replace("\n", " ")[:140]
        meta = {}
        # probe common locations for page info
        meta["attrs_page"] = {k: getattr(ch, k) for k in ("page","pageno","page_no") if hasattr(ch, k)}
        loc = getattr(ch, "location", None)
        if loc:
            meta["location"] = {k: getattr(loc, k) for k in dir(loc) if k.lower().startswith("page")}
        prov = getattr(ch, "provenance", None)
        if prov:
            # provenance can be a list; peek first item
            try:
                p0 = prov[0]
                meta["prov0"] = {k: getattr(p0, k) for k in dir(p0) if "page" in k.lower()}
            except Exception:
                meta["prov0"] = str(type(prov))
        meta["metadata"] = getattr(ch, "metadata", None)

        print(f"\n--- CHUNK {cnt} ---")
        print("TEXT:", text)
        print("META:", json.dumps(meta, default=str, indent=2))
        if cnt >= n:
            break

    print(f"\n[INFO] Showed first {min(cnt,n)} / total >= {cnt} chunks")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/debug_docling.py /path/to/file.pdf")
        sys.exit(1)
    main(sys.argv[1])

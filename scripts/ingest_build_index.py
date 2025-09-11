import os, json, uuid, yaml, argparse
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer

from rag.chunkers import DoclingHybridChunker

CONFIG_PATH = "configs/rag.yaml"
COLLECTION_NAME = "faa_phase1"


class STEmbedding:
    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    # Chroma requires the param name to be exactly "input"
    def __call__(self, input):
        # Accept str or list[str]
        if isinstance(input, str):
            texts = [input]
        else:
            texts = list(input)

        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.tolist()

    # Optional helpers; Chroma doesn’t require these, but nice to have
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.__call__(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.__call__([text])[0]

    def name(self) -> str:
        # Deterministic name lets Chroma detect conflicts
        return f"sentence-transformers:{self.model_name}"



def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="Only ingest these source IDs (e.g., PHAK AFH)")
    ap.add_argument("--rebuild", action="store_true", help="Drop collection first")
    args = ap.parse_args()

    cfg = load_config(CONFIG_PATH)
    manifest = load_manifest(cfg["manifest_path"])

    # Optional filter by ID
    if args.only:
        only = {s.upper() for s in args.only}
        manifest = [m for m in manifest if m["id"].upper() in only]
        if not manifest:
            print("[ERR] No manifest entries match --only filter")
            return

    # Init embeddings + DB
    emb_model_name = cfg["embed_model"]
    emb_fn = STEmbedding(emb_model_name)

    client = chromadb.PersistentClient(path=cfg["persist_dir"])
    if args.rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"[INFO] Dropped '{COLLECTION_NAME}'")
        except Exception:
            pass

    coll = client.get_or_create_collection(
    COLLECTION_NAME,
    embedding_function=emb_fn 
    )
    # Docling-only chunker, sized to your embed token budget
    chunker = DoclingHybridChunker(
        embed_model_name=emb_model_name,
        max_tokens=int(cfg["chunk_size_tokens"]),
        merge_peers=True
    )

    total_added = 0
    for item in manifest:
        sid, pdf_path = item["id"], item["path"]
        if not os.path.exists(pdf_path):
            print(f"[WARN] Missing PDF: {pdf_path}")
            continue

        print(f"[INFO] {sid}: ingesting with Docling hybrid -> {pdf_path}")
        added = 0
        # Collect in small batches to reduce add() overhead
        buf_ids, buf_docs, buf_meta = [], [], []
        BATCH = 64

        # Iterate docling chunks
        chunks_iter = list(chunker.chunks(pdf_path))
        for text, meta in tqdm(chunks_iter, desc=f"{sid} chunks"):
            page = meta["page"]
            uid = f"{sid}-p{page}-{uuid.uuid4().hex[:8]}"
            buf_ids.append(uid)
            buf_docs.append(text)

            safe_meta = {
                "source_id": sid,
                "page": int(page),
            }
            section_val = meta.get("section")
            if section_val:  # only add if it’s a real string
                safe_meta["section"] = section_val

            buf_meta.append(safe_meta)
            if len(buf_ids) >= BATCH:
                coll.add(ids=buf_ids, documents=buf_docs, metadatas=buf_meta)
                added += len(buf_ids)
                buf_ids, buf_docs, buf_meta = [], [], []

        # flush remainder
        if buf_ids:
            coll.add(ids=buf_ids, documents=buf_docs, metadatas=buf_meta)
            added += len(buf_ids)

        print(f"[OK] {sid}: chunks_added={added}")
        total_added += added

    print(f"[SUCCESS] Total chunks added: {total_added} into {cfg['persist_dir']} (collection '{COLLECTION_NAME}')")

if __name__ == "__main__":
    main()

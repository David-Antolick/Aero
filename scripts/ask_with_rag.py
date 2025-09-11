#!/usr/bin/env python3
"""
Ask the local LLM (OpenAI-compatible) with RAG grounding from the Chroma index.

Success criteria for this script:
1) Proves it can talk to the LLM by returning a response.
2) Provides grounded context with page-number citations from your FAA corpus.

Usage:
  python scripts/ask_with_rag.py --q "What are the requirements for a student pilot to solo?"

Environment (can be set via scripts/.env or shell):
  AERO_LLM_BASE_URL  (default http://10.88.100.175:8000/v1)
  AERO_LLM_API_KEY   (default glm-local)
  AERO_LLM_MODEL     (default glm-4.5-air-awq)
"""
import argparse
import os
import sys
from typing import Any, Dict, List

import yaml as pyyaml
import numpy as np
from dotenv import load_dotenv

import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from openai._exceptions import OpenAIError
import torch

CONFIG_PATH = "configs/rag.yaml"
COLLECTION_NAME = "faa_phase0"


class STEmbedding:
    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, input):
        if isinstance(input, str):
            texts = [input]
        else:
            texts = list(input)
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.__call__([text])[0]

    def name(self) -> str:
        return f"sentence-transformers:{self.model_name}"


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return pyyaml.safe_load(f)


def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def format_citation(meta: Dict[str, Any], citation_fmt: str) -> str:
    return citation_fmt.format(
        source_id=meta.get("source_id", "?"),
        page=meta.get("page", "?")
    )


def main() -> None:
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="User query text")
    ap.add_argument("--k", type=int, default=None, help="Top-k contexts (defaults to config top_k)")
    args = ap.parse_args()

    cfg = load_config(CONFIG_PATH)
    top_k = int(args.k or cfg.get("top_k", 8))
    max_initial = int(cfg.get("max_initial_candidates", max(top_k, 16)))
    citation_fmt = cfg.get("citation_format", "[{source_id} p.{page}]")

    # Retrieval
    emb_name = cfg["embed_model"]
    env_dev = os.getenv("AERO_EMBED_DEVICE", "").lower().strip()
    if env_dev == "cuda":
        device = "cuda"
    else:
        device = "cpu"
    emb_fn = STEmbedding(emb_name, device=device)
    print(f"[INFO] Embedding device: {device}", file=sys.stderr)

    client = chromadb.PersistentClient(path=cfg["persist_dir"])
    coll = client.get_or_create_collection(COLLECTION_NAME, embedding_function=emb_fn)

    # Get initial candidates (server-side ANN) and let the server order them
    res = coll.query(
        query_texts=[args.q],
        n_results=max(max_initial, top_k),
        include=["documents", "metadatas"],
    )

    docs: List[str] = res.get("documents", [[]])[0]
    metas: List[Dict[str, Any]] = res.get("metadatas", [[]])[0]

    if not docs:
        print("[ERR] No retrieval results. Ensure the Chroma index exists and is populated.", file=sys.stderr)
        sys.exit(2)

    # Trim to top_k
    docs = docs[:top_k]
    metas = metas[:top_k]

    # Build grounded context with inline citations
    context_blocks: List[str] = []
    for d, m in zip(docs, metas):
        context_blocks.append(f"{format_citation(m, citation_fmt)}\n{d.strip()}")
    context_text = "\n\n".join(context_blocks)

    # System prompt
    sys_prompt_path = cfg.get("system_prompt_path")
    sys_prompt = read_text(sys_prompt_path) if sys_prompt_path else ""
    if not sys_prompt:
        sys_prompt = "Educational use only. Use only provided sources. Keep answers concise and cite with [ID p.N]."

    instruction = (
        "You are an FAA ground school assistant. Answer the user's question using ONLY the provided context. "
        "Include citations inline using the format [ID p.N]. If information is not in the context, say you don't have it."
    )

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "system", "content": instruction},
        {"role": "system", "content": "Context:\n" + context_text},
        {"role": "user", "content": args.q},
    ]

    # LLM call
    base_url = os.getenv("AERO_LLM_BASE_URL", "http://10.88.100.175:8000/v1")
    api_key = os.getenv("AERO_LLM_API_KEY", "glm-local")
    model = os.getenv("AERO_LLM_MODEL", "glm-4.5-air-awq")

    try:
        oai = OpenAI(base_url=base_url, api_key=api_key)
        resp = oai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
        )
        answer = resp.choices[0].message.content.strip()
        print("\n=== ANSWER ===\n")
        print(answer)
        print("\n=== DEBUG: First context block ===\n")
        print(context_blocks[0])
    except OpenAIError as e:
        print(f"[ERR] LLM request failed: {e}", file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print(f"[ERR] Unexpected error: {e}", file=sys.stderr)
        sys.exit(4)


if __name__ == "__main__":
    main()

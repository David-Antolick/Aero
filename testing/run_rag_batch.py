#!/usr/bin/env python3
"""Batch runner that reuses scripts/ask_with_rag.py to score canned RAG prompts."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from openai._exceptions import OpenAIError

# Ensure repo root is on sys.path so we can import scripts.ask_with_rag
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse helpers + config constants from the interactive script
from scripts.ask_with_rag import (  # type: ignore
    COLLECTION_NAME,
    CONFIG_PATH,
    STEmbedding,
    format_citation,
    load_config,
    read_text,
)

DEFAULT_INPUT = Path("testing/testing.json")
DEFAULT_OUTPUT = Path("testing/results.json")


class RagBatchRunner:
    def __init__(self, cfg_path: str = CONFIG_PATH, collection_name: str = COLLECTION_NAME):
        load_dotenv(Path("scripts/.env"))

        self.cfg = load_config(cfg_path)
        self.collection_name = collection_name

        env_dev = os.getenv("AERO_EMBED_DEVICE", "").strip().lower()
        device = "cuda" if env_dev == "cuda" else "cpu"
        self.embed_fn = STEmbedding(self.cfg["embed_model"], device=device)

        persist_path = self.cfg.get("persist_dir", "storage/chroma")
        self.chroma = chromadb.PersistentClient(path=persist_path)
        self.collection = self.chroma.get_or_create_collection(
            collection_name, embedding_function=self.embed_fn
        )

        self.default_top_k = int(self.cfg.get("top_k", 8))
        self.max_initial = int(self.cfg.get("max_initial_candidates", max(self.default_top_k, 16)))
        self.citation_fmt = self.cfg.get("citation_format", "[{source_id} p.{page}]")

        sys_prompt_path = self.cfg.get("system_prompt_path")
        self.system_prompt = read_text(sys_prompt_path) if sys_prompt_path else ""
        if not self.system_prompt:
            self.system_prompt = (
                "Educational use only. Use only provided sources. Keep answers concise and cite with [ID p.N]."
            )
        self.instruction = (
            "You are an FAA ground school assistant. Answer the user's question using ONLY the provided context. "
            "Include citations inline using the format [ID p.N]. If information is not in the context, say you don't have it."
        )

        base_url = os.getenv("AERO_LLM_BASE_URL", "http://10.88.100.175:8000/v1")
        api_key = os.getenv("AERO_LLM_API_KEY", "glm-local")
        self.model_name = os.getenv("AERO_LLM_MODEL", "glm-4.5-air-awq")
        self.llm = OpenAI(base_url=base_url, api_key=api_key)

    def _retrieve(self, question: str, top_k: int) -> tuple[List[str], List[Dict[str, Any]]]:
        res = self.collection.query(
            query_texts=[question],
            n_results=max(self.max_initial, top_k),
            include=["documents", "metadatas"],
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        if not docs:
            raise RuntimeError("No retrieval results. Ensure the Chroma index exists and is populated.")
        return docs[:top_k], metas[:top_k]

    def ask(self, question: str, top_k: int | None = None) -> Dict[str, Any]:
        k = int(top_k or self.default_top_k)
        docs, metas = self._retrieve(question, k)

        context_blocks = [f"{format_citation(m, self.citation_fmt)}\n{d.strip()}" for d, m in zip(docs, metas)]
        context_text = "\n\n".join(context_blocks)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": self.instruction},
            {"role": "system", "content": "Context:\n" + context_text},
            {"role": "user", "content": question},
        ]

        try:
            resp = self.llm.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=2048,
            )
        except OpenAIError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        answer = resp.choices[0].message.content.strip()
        return {
            "answer": answer,
            "contexts": context_blocks,
            "metadatas": metas,
        }


def load_test_queries(path: Path, limit: int | None) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text())
    queries = payload.get("test_queries", [])
    if limit is not None:
        return queries[:limit]
    return queries


def run_batch(
    input_path: Path,
    output_path: Path,
    delay_s: float,
    limit: int | None,
    top_k: int | None,
) -> None:
    runner = RagBatchRunner()
    queries = load_test_queries(input_path, limit)
    results: List[Dict[str, Any]] = []

    for item in queries:
        qid = item.get("id")
        question = item.get("query")
        print(f"[INFO] Running {qid}: {question}")
        try:
            resp = runner.ask(question, top_k=top_k)
            results.append(
                {
                    "id": qid,
                    "category": item.get("category"),
                    "intent": item.get("intent"),
                    "expected_behavior": item.get("expected_behavior"),
                    "model_answer": resp["answer"],
                    "contexts": resp["contexts"],
                    "metadatas": resp["metadatas"],
                }
            )
        except Exception as exc:  # noqa: BLE001 - capture per-query failures
            print(f"[ERR] {qid} failed: {exc}", file=sys.stderr)
            results.append(
                {
                    "id": qid,
                    "category": item.get("category"),
                    "intent": item.get("intent"),
                    "expected_behavior": item.get("expected_behavior"),
                    "error": str(exc),
                }
            )
        time.sleep(delay_s)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"[INFO] Saved {len(results)} results to {output_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Batch FAA RAG regression tester")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to test query JSON")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination for result JSON")
    ap.add_argument("--delay", type=float, default=0.4, help="Sleep (seconds) between queries")
    ap.add_argument("--limit", type=int, default=None, help="Run only the first N queries")
    ap.add_argument("--top-k", type=int, default=None, help="Override retrieval top-k")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_batch(
        input_path=args.input,
        output_path=args.output,
        delay_s=args.delay,
        limit=args.limit,
        top_k=args.top_k,
    )

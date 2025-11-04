# Aero — FAA-Grounded RAG Assistant

Aero is a retrieval‑augmented assistant focused on aviation study and reference (PPL → IR → CPL → CFI/CFII). It answers questions using content retrieved from FAA publications you ingest locally. Responses aim to be concise, teaching‑oriented, and include page‑level citations.


## Features
- **RAG over FAA sources** using a local Chroma vector store
- **Docling hybrid chunking** with page detection and fallbacks
- **OpenAI‑compatible inference** (points to your local/server LLM endpoint)
- **Inline citations** like `[PHAK p.7]` or `[AIM p. 8-12]`
- **Scripts for end‑to‑end flow**: fetch PDFs → build manifest → ingest/index → ask with RAG
- **Config‑driven** via `configs/rag.yaml` and prompt in `prompts/system_faa_grounded.txt`


## Repository layout
- `configs/rag.yaml` — Retrieval and embedding configuration (chunk sizes, model, top‑k, persist dir, manifest path, etc.)
- `ingest/manifest.json` — Auto‑generated mapping of source IDs to local PDF paths
- `ingest/sources.json` — Source list to download (URL, filename, ID)
- `prompts/system_faa_grounded.txt` — System prompt and guardrails
- `rag/chunkers.py` — `DoclingHybridChunker` used during ingest
- `scripts/` — Operational scripts (see below)
- `data/` — Local folder where FAA PDFs are stored
- `storage/chroma/` — Chroma persistence directory (created after ingest)
- `roadmap.txt` — Longer‑term milestones and quality targets
- `requirements.txt` — Python dependencies


## Prerequisites
- Python 3.10+
- A running OpenAI‑compatible LLM endpoint you can query (local server or remote). Examples include OpenAI‑compatible gateways around local models.
- System libraries needed by `docling` and `pypdf` for PDF parsing (platform‑specific)


## Installation
1) Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2) Install Python packages:

```bash
pip install -U pip
pip install -U -r requirements.txt
# Depending on your environment, you may also need:
pip install docling pypdf
```

If you hit tokenizer/model download prompts (from `transformers`/`sentence-transformers`), ensure your machine has internet access or pre‑cache models.


## Configuration
Key settings live in `configs/rag.yaml`:
- `chunk_size_tokens`, `chunk_overlap_tokens`
- `embed_model` (default: `BAAI/bge-small-en-v1.5`)
- `vector_store` and `persist_dir` (default: `storage/chroma`)
- `top_k`, `mmr_lambda`, `max_initial_candidates`
- `citation_format`
- `system_prompt_path` (default: `prompts/system_faa_grounded.txt`)
- `manifest_path` (default: `ingest/manifest.json`)

Environment variables (can be set in your shell or via a `.env` file):
- `AERO_LLM_BASE_URL` (default `http://10.88.100.175:8000/v1`)
- `AERO_LLM_API_KEY` (default `glm-local`)
- `AERO_LLM_MODEL` (default `glm-4.5-air-awq`)
- `AERO_EMBED_DEVICE` (`cuda` or `cpu`, default auto/`cpu`)
- `AERO_LLM_TIMEOUT_S`, `AERO_LLM_RETRIES`, `AERO_LLM_RETRY_DELAY_S` (used by readiness check)

Tip: you can create `scripts/.env` (git‑ignored) to store local values used by scripts.


## End‑to‑end workflow
Below is the typical sequence to get to grounded Q&A.

1) Fetch FAA PDFs

```bash
python scripts/fetch_faa_pdfs.py
```
- Reads `ingest/sources.json`
- Downloads PDFs into `data/`

2) Build a manifest of local PDFs

```bash
python scripts/build_manifest.py
```
- Scans `data/*.pdf`
- Writes `ingest/manifest.json` with `{"id": <short_id>, "path": <pdf_path>}` entries
- Uses `ingest/sources.json` for filename→ID mapping; otherwise infers IDs from filenames (e.g., `14CFR_Part91.pdf` → `CFR91`)

3) Ingest and build the vector index

```bash
# Basic
python scripts/ingest_build_index.py

# Rebuild from scratch (drops collection)
python scripts/ingest_build_index.py --rebuild

# Only ingest specific sources by ID
python scripts/ingest_build_index.py --only PHAK AFH AIM
```
- Uses `rag/DoclingHybridChunker` to produce page‑true chunks
- Embeds with `sentence-transformers` and writes to Chroma at `storage/chroma`

4) Sanity‑check your LLM endpoint

```bash
python scripts/check_llm.py
```
- Lists `/v1/models` and sends a minimal chat completion
- Exits non‑zero if not healthy (useful for CI)

5) Ask questions with RAG grounding

```bash
python scripts/ask_with_rag.py --q "What are the requirements for a student pilot to solo?"

# Optional:
#   --k N   # top‑k contexts (overrides `top_k` in configs/rag.yaml)
```
- Retrieves top‑k contexts, builds a grounded system/context message, and queries the LLM
- Prints the answer and the first context block for debugging


## Scripts reference
- `scripts/fetch_faa_pdfs.py` — Download PDFs from `ingest/sources.json` to `data/`
- `scripts/build_manifest.py` — Generate `ingest/manifest.json` from local PDFs
- `scripts/ingest_build_index.py` — Chunk + embed + upsert into Chroma
- `scripts/ask_with_rag.py` — Query LLM with grounded context and citations
- `scripts/check_llm.py` — Readiness/health check for your OpenAI‑compatible endpoint
- `scripts/debug_docling.py` — Inspect docling chunk/page metadata for a given PDF


## Troubleshooting
- Missing packages like `docling` or `pypdf` during ingest: `pip install docling pypdf`
- GPU not used for embeddings: set `AERO_EMBED_DEVICE=cuda` if CUDA is available; otherwise CPU is used
- No retrieval results in `ask_with_rag.py`:
  - Ensure `storage/chroma` exists and is populated
  - Confirm `configs/rag.yaml` references the correct `persist_dir` and `manifest_path`
- LLM endpoint issues:
  - Verify `AERO_LLM_BASE_URL`, `AERO_LLM_API_KEY`, `AERO_LLM_MODEL`
  - Run `python scripts/check_llm.py` and review logs


## Roadmap and goals
See `roadmap.txt` for phased milestones, acceptance criteria, and quality/safety targets (e.g., hit‑rate goals, latency, grounding discipline, personalization hooks).


## Safety disclaimer
Training aid only—verify with AFM/POH, current charts, AIM/FAR, and your instructor. Do not use this project for real‑time operational decisions.

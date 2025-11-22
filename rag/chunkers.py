# rag/chunkers.py
from typing import Iterator, Tuple, Dict, Any, Optional
from functools import lru_cache
from transformers import AutoTokenizer
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.chunking import HybridChunker
import re
import difflib


class DoclingHybridChunker:
    """
    Docling-based hybrid chunker with page fallback:
    - Try to read page from docling chunk (page/location/provenance).
    - If missing, map by searching the chunk text across per-page text (pypdf cache).
    """

    def __init__(self, embed_model_name: str, max_tokens: int, merge_peers: bool = True):


        self.tok = AutoTokenizer.from_pretrained(embed_model_name)

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        pipeline_options.do_code_enrichment = False

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend,
                )
            }
        )

        self.hybrid = HybridChunker(
            tokenizer=self.tok,
            max_tokens=max_tokens,
            merge_peers=merge_peers
        )

    def _guess_page_from_chunk(self, chunk) -> Optional[int]:
        # Direct attrs
        for attr in ("page", "pageno", "page_no"):
            if hasattr(chunk, attr):
                try:
                    v = int(getattr(chunk, attr))
                    if v > 0:
                        return v
                except Exception:
                    pass
        # Metadata dict
        meta = getattr(chunk, "metadata", None) or {}
        for k in ("page", "pageno", "page_no", "page_num", "pageNumber"):
            if k in meta:
                try:
                    v = int(meta[k])
                    if v > 0:
                        return v
                except Exception:
                    pass
        # Location object
        loc = getattr(chunk, "location", None)
        if loc is not None:
            for k in ("page", "page_no", "pageno", "page_index"):
                if hasattr(loc, k):
                    try:
                        v = int(getattr(loc, k))
                        if k == "page_index":
                            v = v + 1
                        if v > 0:
                            return v
                    except Exception:
                        pass
        # Provenance list (optional)
        prov = getattr(chunk, "provenance", None)
        if prov:
            try:
                p0 = prov[0]
                for k in dir(p0):
                    if "page" in k.lower():
                        try:
                            v = int(getattr(p0, k))
                            if "index" in k.lower():
                                v = v + 1
                            if v > 0:
                                return v
                        except Exception:
                            pass
            except Exception:
                pass
        return None

    @lru_cache(maxsize=8)
    def _page_texts(self, pdf_path: str):
        # Cache (path -> list of page texts) for quick substring search
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        pages = []
        for i in range(len(reader.pages)):
            try:
                t = reader.pages[i].extract_text() or ""
            except Exception:
                t = ""
            pages.append(t.replace("\x00", " "))
        return pages

    def _norm(self, s: str) -> str:
        if not s:
            return ""
        # join hyphenated line breaks, normalize whitespace, lowercase
        s = s.replace("\r", "\n")
        s = re.sub(r"-\s*\n\s*", "", s)          # remove hyphen + newline breaks
        s = re.sub(r"\s+", " ", s)               # collapse whitespace
        return s.strip().lower()

    def _guess_page_by_search(self, pdf_path: str, text: str) -> Optional[int]:
        pages = self._page_texts(pdf_path)
        if not text:
            return None

        probe = self._norm(text)[:600]           # limit work
        if len(probe) < 40:
            return None

        # generate a few windows from the probe
        windows = []
        L = len(probe)
        windows.append(probe[:200])
        if L > 240:
            windows.append(probe[80:280])
        if L > 360:
            windows.append(probe[160:360])

        # exact-substring search on normalized page text
        norm_pages = [self._norm(p) for p in pages]
        for w in windows:
            if len(w) < 40:
                continue
            for idx, pg in enumerate(norm_pages):
                if w in pg:
                    return idx + 1

        # fuzzy fallback: compare first 200–800 chars vs page slices
        best_idx, best_score = None, 0.0
        for idx, pg in enumerate(norm_pages):
            if len(pg) < 200:
                continue
            comp_slice = pg[:2000] if len(pg) > 2000 else pg
            for size in (200, 400, 800):
                sl = probe[:size] if len(probe) >= size else probe
                if len(sl) < 60:
                    continue
                score = difflib.SequenceMatcher(None, sl, comp_slice).ratio()
                if score > best_score:
                    best_idx, best_score = idx, score

        if best_idx is not None and best_score >= 0.28:
            return best_idx + 1
        return None

    def _guess_section(self, chunk) -> Optional[str]:
        for attr in ("heading", "section", "title"):
            val = getattr(chunk, attr, None)
            if isinstance(val, str) and val.strip():
                return val.strip()
        meta = getattr(chunk, "metadata", None) or {}
        for k in ("heading", "section", "title"):
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    def chunks(self, pdf_path: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
        result = self.converter.convert(pdf_path)
        dl_doc = result.document

        for ch in self.hybrid.chunk(dl_doc):
            text = getattr(ch, "text", None) or self.hybrid.contextualize(ch)
            text = (text or "").replace("\x00", " ").strip()
            if not text:
                continue

            page = self._guess_page_from_chunk(ch)
            if page is None:
                page = self._guess_page_by_search(pdf_path, text)
                if page is None:
                    # if still unknown, skip to preserve page-true citations
                    continue

            section = self._guess_section(ch)
            yield text, {"page": int(page), "section": section}

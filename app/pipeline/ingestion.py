"""
Phase 1: Ingestion
Goal:   Load a raw file → clean text → return structured page dicts
Input:  file path (PDF / DOCX / TXT / MD)
Output: list of { text, metadata } dicts
"""

import re
from pathlib import Path

import fitz          # PyMuPDF
import docx

from app.pipeline.observer import log_phase


class DocumentIngester:

    def load(self, file_path: str) -> list[dict]:
        """Entry point: detect format and dispatch to correct parser."""
        path   = Path(file_path)
        suffix = path.suffix.lower()

        with log_phase("ingestion", file=path.name):
            if suffix == ".pdf":
                raw_pages = self._parse_pdf(file_path)
            elif suffix == ".docx":
                raw_pages = self._parse_docx(file_path)
            elif suffix in (".txt", ".md"):
                raw_pages = self._parse_text(file_path)
            else:
                raise ValueError(f"Unsupported file type: {suffix}")

            return [self._clean(page) for page in raw_pages]

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    def _parse_pdf(self, path: str) -> list[dict]:
        pages = []
        doc   = fitz.open(path)
        for i, page in enumerate(doc):
            text = page.get_text("text")
            pages.append({
                "text": text,
                "metadata": {
                    "source":      Path(path).name,
                    "page":        i + 1,
                    "total_pages": len(doc),
                },
            })
        return pages

    def _parse_docx(self, path: str) -> list[dict]:
        doc   = docx.Document(path)
        paras = [p.text for p in doc.paragraphs if p.text.strip()]
        text  = "\n\n".join(paras)
        return [{"text": text, "metadata": {"source": Path(path).name, "page": 1}}]

    def _parse_text(self, path: str) -> list[dict]:
        text = Path(path).read_text(encoding="utf-8")
        return [{"text": text, "metadata": {"source": Path(path).name, "page": 1}}]

    # ------------------------------------------------------------------
    # Cleaner
    # ------------------------------------------------------------------

    def _clean(self, page: dict) -> dict:
        """
        Normalise extracted text.
        Logs raw_chars vs clean_chars so you can see how much noise was removed.
        """
        original = page["text"]

        text = re.sub(r"\n{3,}", "\n\n", original)          # collapse blank lines
        text = re.sub(r" {2,}", " ", text)                   # collapse spaces
        text = re.sub(r"[^\x00-\x7F\uAC00-\uD7A3]", "", text)  # keep ASCII + Korean
        text = text.strip()

        page["text"]                      = text
        page["metadata"]["raw_chars"]     = len(original)
        page["metadata"]["clean_chars"]   = len(text)
        return page

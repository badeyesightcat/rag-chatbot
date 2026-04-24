# =============================================================
# PHASE 1: INGESTION
# Goal: Load a document → clean text → return structured chunks
# Input:  raw file (PDF / DOCX / TXT / MD)
# Output: list of { text, metadata } dicts
# =============================================================

import re
import fitz # PyMuPDF — best PDF parser
import docx

from pathlib import Path
from typing import List, Dict
from app.pipeline.observer import log_phase # Phase 6 logging

class DocumentIngester:

    def load(self, file_path: str) -> List[Dict]: 
        """Entry point: detect format and dispatch to correct parser."""
        path = Path(file_path)
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
            
            # Clean each page and record what changed
            cleaned = [self._clean(page) for page in raw_pages]
            return cleaned
        
    def _parse_pdf(self, path) -> List[Dict]:
        pages = []
        doc = fitz.open(path)
        for i, page in enumerate(doc):
            text = page.get_text("text")
            pages.append({
                "text": text,
                "metadata": {
                    "source": Path(path).name,
                    "page": i + 1,
                    "total_pages": len(doc)
                }
            })
        return pages
    
    def _parse_docx(self, path) -> List[Dict]:
        doc = docx.Document(path)
        # Treat each paragraph as a "page" for consistency
        paras = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text = "\n\n".join(paras)
        return [{ "text": full_text, "metadata": { "source": Path(path).name, "page": 1 }}]
    
    def _parse_text(self, path) -> List[Dict]:
        text = Path(path).read_text(encoding="utf-8")
        return [{ "text": text, "metadata": { "source": Path(path).name, "page": 1 }}]
    
    def _clean(self, page: Dict) -> Dict:
        """
        Text normalization — THIS IS WHERE MOST BUGS HIDE.
        Log before/after so you can see what's being removed.
        """
        original = page["text"]

        text = re.sub(r'\n{3,}', '\n\n', original) # collapse blank lines
        text = re.sub(r' {2,}', ' ', text) # collapse spaces
        text = re.sub(r'[^\x00-\x7F\uAC00-\uD7A3]', '', text) # keep ASCII + Korean
        text = text.strip()

        page["text"] = text
        page["metadata"]["raw_chars"] = len(original) # visible in admin UI
        page["metadata"]["clean_chars"] = len(text)
        return page

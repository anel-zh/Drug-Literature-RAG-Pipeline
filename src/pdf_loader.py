from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")  # ﬁ, ﬂ
    text = " ".join(text.split())
    text = re.sub(r"\s+([,.;:])", r"\1", text)

    return text.strip()


def load_pdf_pages(pdf_path: Path, doc_id: str, doc_type: str) -> List[Dict]:
    """
    Returns pages as dicts:
      {doc_id, doc_type, page (1-based), text}
    """
    reader = PdfReader(str(pdf_path))
    pages: List[Dict] = []

    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = clean_text(raw)
        if text:
            pages.append(
                {
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "page": i,
                    "text": text,
                }
            )

    return pages
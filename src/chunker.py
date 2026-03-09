from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_type: str
    page: int
    text: str
    section_id: str        
    section_label: str     


# Canonical section taxonomy
# FDA canonical IDs
FDA_UNKNOWN = "FDA_UNKNOWN"
PAPER_UNKNOWN = "PAPER_UNKNOWN"

_FDA_SECTION_PATTERNS: List[Tuple[re.Pattern, Tuple[str, str]]] = [
    # Big headers
    (re.compile(r"\bBOXED\s+WARNING\b", re.IGNORECASE), ("FDA_BOXED_WARNING", "BOXED WARNING")),
    (re.compile(r"\bHIGHLIGHTS\s+OF\s+PRESCRIBING\s+INFORMATION\b", re.IGNORECASE), ("FDA_HIGHLIGHTS", "HIGHLIGHTS")),
    (re.compile(r"\bFULL\s+PRESCRIBING\s+INFORMATION\b", re.IGNORECASE), ("FDA_FULL_PI", "FULL PRESCRIBING INFORMATION")),

    # Common numbered sections
    (re.compile(r"\b1\s+INDICATIONS\s+AND\s+USAGE\b", re.IGNORECASE), ("FDA_INDICATIONS", "1 INDICATIONS AND USAGE")),
    (re.compile(r"\b2\s+DOSAGE\s+AND\s+ADMINISTRATION\b", re.IGNORECASE), ("FDA_DOSAGE_ADMIN", "2 DOSAGE AND ADMINISTRATION")),
    (re.compile(r"\b3\s+DOSAGE\s+FORMS\s+AND\s+STRENGTHS\b", re.IGNORECASE), ("FDA_DOSAGE_FORMS", "3 DOSAGE FORMS AND STRENGTHS")),
    (re.compile(r"\b4\s+CONTRAINDICATIONS\b", re.IGNORECASE), ("FDA_CONTRAINDICATIONS", "4 CONTRAINDICATIONS")),
    (re.compile(r"\b5\s+WARNINGS\s+AND\s+PRECAUTIONS\b", re.IGNORECASE), ("FDA_WARNINGS", "5 WARNINGS AND PRECAUTIONS")),
    (re.compile(r"\b6\s+ADVERSE\s+REACTIONS\b", re.IGNORECASE), ("FDA_ADVERSE_REACTIONS", "6 ADVERSE REACTIONS")),
    (re.compile(r"\b7\s+DRUG\s+INTERACTIONS\b", re.IGNORECASE), ("FDA_DRUG_INTERACTIONS", "7 DRUG INTERACTIONS")),
    (re.compile(r"\b8\s+USE\s+IN\s+SPECIFIC\s+POPULATIONS\b", re.IGNORECASE), ("FDA_POPULATIONS", "8 USE IN SPECIFIC POPULATIONS")),
    (re.compile(r"\b10\s+OVERDOSAGE\b", re.IGNORECASE), ("FDA_OVERDOSAGE", "10 OVERDOSAGE")),
    (re.compile(r"\b11\s+DESCRIPTION\b", re.IGNORECASE), ("FDA_DESCRIPTION", "11 DESCRIPTION")),
    (re.compile(r"\b12\s+CLINICAL\s+PHARMACOLOGY\b", re.IGNORECASE), ("FDA_CLIN_PHARM", "12 CLINICAL PHARMACOLOGY")),
    (re.compile(r"\b13\s+NONCLINICAL\s+TOXICOLOGY\b", re.IGNORECASE), ("FDA_TOXICOLOGY", "13 NONCLINICAL TOXICOLOGY")),
    (re.compile(r"\b14\s+CLINICAL\s+STUDIES\b", re.IGNORECASE), ("FDA_CLIN_STUDIES", "14 CLINICAL STUDIES")),
    (re.compile(r"\b16\s+HOW\s+SUPPLIED\s*/?\s*STORAGE\s+AND\s+HANDLING\b", re.IGNORECASE), ("FDA_HOW_SUPPLIED", "16 HOW SUPPLIED / STORAGE")),
    (re.compile(r"\b17\s+PATIENT\s+COUNSELING\s+INFORMATION\b", re.IGNORECASE), ("FDA_PATIENT_COUNSEL", "17 PATIENT COUNSELING")),

    # Subsections 
    (re.compile(r"\b12\.1\s+MECHANISM\s+OF\s+ACTION\b", re.IGNORECASE), ("FDA_MOA", "12.1 MECHANISM OF ACTION")),
    (re.compile(r"\b6\.2\s+IMMUNOGENICITY\b", re.IGNORECASE), ("FDA_IMMUNOGENICITY", "6.2 IMMUNOGENICITY")),

    # Other 
    (re.compile(r"\bMEDICATION\s+GUIDE\b", re.IGNORECASE), ("FDA_MED_GUIDE", "MEDICATION GUIDE")),
    (re.compile(r"\bINSTRUCTIONS\s+FOR\s+USE\b", re.IGNORECASE), ("FDA_IFU", "INSTRUCTIONS FOR USE")),
]

_PAPER_SECTION_PATTERNS: List[Tuple[re.Pattern, Tuple[str, str]]] = [
    (re.compile(r"\bABSTRACT\b", re.IGNORECASE), ("PAPER_ABSTRACT", "ABSTRACT")),
    (re.compile(r"\bINTRODUCTION\b", re.IGNORECASE), ("PAPER_INTRO", "INTRODUCTION")),
    (re.compile(r"\bMETHODS\b|\bMATERIALS\s+AND\s+METHODS\b", re.IGNORECASE), ("PAPER_METHODS", "METHODS")),
    (re.compile(r"\bRESULTS\b", re.IGNORECASE), ("PAPER_RESULTS", "RESULTS")),
    (re.compile(r"\bDISCUSSION\b", re.IGNORECASE), ("PAPER_DISCUSSION", "DISCUSSION")),
    (re.compile(r"\bCONCLUSION(S)?\b", re.IGNORECASE), ("PAPER_CONCLUSIONS", "CONCLUSIONS")),
]


def infer_section_from_page_text(doc_type: str, text: str) -> Optional[Tuple[str, str]]:
    """
    Returns (section_id, section_label) if a header match is found on this page text.
    """
    patterns = _FDA_SECTION_PATTERNS if doc_type == "fda_label" else _PAPER_SECTION_PATTERNS
    for rx, (sec_id, sec_label) in patterns:
        if rx.search(text):
            return sec_id, sec_label
    return None


# Chunking
def _split_with_overlap(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - chunk_overlap
        if start < 0:
            start = 0
    return chunks


def chunk_pages(pages: List[Dict], chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    """
    pages: list of dicts with keys at least:
      - doc_id (str)
      - doc_type (str): 'fda_label' or 'paper'
      - page (int)
      - text (str)
    """
    out: List[Chunk] = []

    # Keep current section per doc_id (carry forward page-to-page)
    current_section: Dict[str, Tuple[str, str]] = {}

    for p in pages:
        doc_id = p["doc_id"]
        doc_type = p.get("doc_type", "paper")
        page_num = int(p["page"])
        text = p.get("text", "") or ""
        if not text.strip():
            continue

        maybe_section = infer_section_from_page_text(doc_type, text)
        if maybe_section:
            current_section[doc_id] = maybe_section

        if doc_id in current_section:
            section_id, section_label = current_section[doc_id]
        else:
            section_id = FDA_UNKNOWN if doc_type == "fda_label" else PAPER_UNKNOWN
            section_label = "UNKNOWN"

        pieces = _split_with_overlap(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, piece in enumerate(pieces):
            chunk_id = f"{doc_id}_p{page_num:04d}_c{i:03d}"
            out.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    doc_type=doc_type,
                    page=page_num,
                    text=piece,
                    section_id=section_id,
                    section_label=section_label,
                )
            )

    return out

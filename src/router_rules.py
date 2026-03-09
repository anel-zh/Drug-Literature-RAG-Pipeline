from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RoutingDecision:
    doc_ids: List[str]
    doc_type: Optional[str]
    section_ids: List[str]  


_WORD_RX_CACHE: dict[str, re.Pattern] = {}


def _word_boundary_rx(term: str) -> re.Pattern:
    term = term.lower().strip()
    if term not in _WORD_RX_CACHE:
        _WORD_RX_CACHE[term] = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    return _WORD_RX_CACHE[term]


def _infer_aliases(doc: dict) -> List[str]:
    aliases = set()

    doc_id = (doc.get("doc_id") or "").lower()
    filename = (doc.get("file") or "").lower()

    if doc_id:
        aliases.add(doc_id)
        aliases.update(doc_id.split("_"))

    if filename:
        filename = filename.replace(".pdf", "")
        aliases.add(filename)
        aliases.update(re.split(r"[_\-\s]+", filename))

    # Remove junk tokens
    stop = {"pdf", "fda", "label", "pmc", "ra"}
    aliases = {a for a in aliases if a and a not in stop and len(a) >= 3}

    return sorted(aliases)


# Canonical section intent map
SECTION_INTENTS: dict[str, List[str]] = {
    "FDA_BOXED_WARNING": ["boxed warning", "black box", "black-box"],
    "FDA_INDICATIONS": ["indication", "indications", "indicated for", "usage"],
    "FDA_IMMUNOGENICITY": ["immunogenicity", "anti-drug", "antibodies", "anti drug", "ada"],
    "FDA_MOA": ["mechanism", "moa", "how it works", "mechanism of action"],
    "FDA_ADVERSE_REACTIONS": ["adverse reactions", "adverse", "side effects", "most common adverse", "ae", "aes"],
    "FDA_CONTRAINDICATIONS": ["contraindication", "contraindications", "should not use", "do not use"],
    "FDA_WARNINGS": ["warnings", "precautions", "serious risk", "risk of", "warning and precaution"],
}


def route_query(query: str, meta: dict) -> RoutingDecision:
    qn = query.lower()

    selected_doc_ids: List[str] = []
    selected_doc_type: Optional[str] = None
    selected_section_ids: List[str] = []

    # 1) Identify specific documents via inferred aliases (word-boundary match)
    for doc in meta.get("docs", []):
        doc_id = doc.get("doc_id")
        if not doc_id:
            continue

        aliases = _infer_aliases(doc)
        matched = False

        # Exact doc_id substring can be OK (doc_ids often contain underscores)
        if doc_id.lower() in qn:
            matched = True
        else:
            # Safer: word-boundary match aliases
            for a in aliases:
                if _word_boundary_rx(a).search(qn):
                    matched = True
                    break

        if matched:
            selected_doc_ids.append(doc_id)

    # If none matched -> empty list => downstream can search across all docs.

    # 2) Identify doc_type (light heuristic)
    if any(_word_boundary_rx(w).search(qn) for w in ["paper", "pmc", "study", "trial"]):
        selected_doc_type = "paper"
    elif any(w in qn for w in ["label", "prescribing", "boxed warning", "indications", "immunogenicity", "moa"]):
        selected_doc_type = "fda_label"

    # 3) Section intent -> canonical section_ids
    for section_id, keywords in SECTION_INTENTS.items():
        if any((k in qn) for k in keywords):
            selected_section_ids.append(section_id)

    # Optional: if section intent implies label, force doc_type
    if selected_section_ids and selected_doc_type is None:
        if any(s.startswith("FDA_") for s in selected_section_ids):
            selected_doc_type = "fda_label"

    return RoutingDecision(
        doc_ids=selected_doc_ids,
        doc_type=selected_doc_type,
        section_ids=selected_section_ids,
    )

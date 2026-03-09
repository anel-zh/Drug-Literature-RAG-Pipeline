from __future__ import annotations

import re
from typing import List

# Keeps things like "12.1", "adalimumab-bwwd", "TNF-α" reasonably intact
_TOKEN_RE = re.compile(
    r"""
    (?:\d+\.\d+(?:\.\d+)*)            # section numbers like 12.1 or 6.2.1
    |(?:[A-Za-z]+(?:[-_][A-Za-z0-9]+)+) # hyphen/underscore terms like adalimumab-bwwd
    |(?:[A-Za-z]+\d+)                 # tokens like SB5
    |(?:\d+%?)                        # numbers, optional %
    |(?:[A-Za-z]+)                    # plain words
    """,
    re.VERBOSE,
)

def tokenize(text: str) -> List[str]:
    if not text:
        return []
    text = text.lower()
    return _TOKEN_RE.findall(text)
"""
date_utils.py
Date-like text detection utilities.
"""
from __future__ import annotations

import re

_DATE_RE_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_DATE_RE_MDY = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_DATE_RE_ISO = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

def is_date_like(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    return bool(_DATE_RE_YEAR.search(t) or _DATE_RE_MDY.search(t) or _DATE_RE_ISO.search(t))

__all__=["is_date_like"]
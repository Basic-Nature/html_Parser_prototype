from __future__ import annotations

import os

from webapp.parser.utils.misc_utils import extract_url_and_label
from webapp.parser.utils.shared_logic import safe_strip


def url_already_listed(urls_file: str, url: str) -> bool:
    if not urls_file or not url or not os.path.exists(urls_file):
        return False
    try:
        with open(urls_file, "r", encoding="utf-8") as f:
            for raw in f:
                s = safe_strip(raw)
                if not s or s.startswith("#"):
                    continue
                existing, _ = extract_url_and_label(s)
                if (existing or s) == url:
                    return True
    except Exception:
        return False
    return False

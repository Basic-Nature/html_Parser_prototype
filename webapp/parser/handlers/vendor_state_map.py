"""
State to vendor mapping for vendor-dispatch handlers.

Entries may be flagged enabled=False when vendor is only a candidate.
"""
from __future__ import annotations

from typing import Dict, List

from webapp.parser.utils.shared_logic import normalize_state_name

VENDOR_STATE_MAP: List[Dict[str, object]] = [
    {
        "state": "georgia",
        "vendor": "clarity",
        "confidence": "medium",
        "notes": "urls.txt: Fulton clarity ENR; verify statewide portal vendor",
        "enabled": True,
    },
    {
        "state": "south_carolina",
        "vendor": "clarity",
        "confidence": "high",
        "notes": "urls.txt: enr-scvotes.org clarity ENR",
        "enabled": True,
    },
    {
        "state": "west_virginia",
        "vendor": "clarity",
        "confidence": "high",
        "notes": "urls.txt: results.enr.clarityelections.com",
        "enabled": True,
    },
    {
        "state": "new_york",
        "vendor": "dominion",
        "confidence": "low",
        "notes": "TODO: enhancedvoting.com domain; confirm vendor",
        "enabled": False,
    },
    {
        "state": "utah",
        "vendor": "dominion",
        "confidence": "low",
        "notes": "TODO: enhancedvoting.com domain; confirm vendor",
        "enabled": False,
    },
]


def get_vendor_for_state(state_key: str) -> str | None:
    """Return vendor name for a normalized state key when enabled."""
    norm = normalize_state_name(state_key)
    if not norm:
        return None
    for entry in VENDOR_STATE_MAP:
        entry_state = normalize_state_name(str(entry.get("state", "")))
        if entry_state == norm and entry.get("enabled", True):
            return str(entry.get("vendor"))
    return None

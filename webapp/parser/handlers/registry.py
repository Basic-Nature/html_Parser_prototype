from __future__ import annotations

import importlib.util
from typing import Dict, Optional

from webapp.parser.Context_Integration.Context_Library.constants import STATE_MODULE_MAP
from webapp.parser.handlers.vendor_state_map import VENDOR_STATE_MAP
from webapp.parser.utils.shared_logic import normalize_county_name, normalize_state_name

DEFAULT_STATE_HANDLER = "webapp.parser.handlers.shared.state_scaffold"
_STATE_HANDLER_OVERRIDES: Dict[str, str] = {}
_COUNTY_HANDLER_OVERRIDES: Dict[str, Dict[str, str]] = {}
_VENDOR_DISPATCH_MODULE = "webapp.parser.handlers.shared.vendor_dispatch"


def register_state_handler(state_key: str, module_path: str) -> None:
    """Register or override a state handler module path."""
    normalized = normalize_state_name(state_key)
    if normalized:
        _STATE_HANDLER_OVERRIDES[normalized] = module_path


def register_county_handler(state_key: str, county_key: str, module_path: str) -> None:
    """Register or override a county handler module path."""
    normalized_state = normalize_state_name(state_key)
    normalized_county = normalize_county_name(county_key)
    if not normalized_state or not normalized_county:
        return
    _COUNTY_HANDLER_OVERRIDES.setdefault(normalized_state, {})[normalized_county] = module_path


def apply_vendor_overrides() -> None:
    """Register vendor dispatch handlers for mapped states."""
    for entry in VENDOR_STATE_MAP:
        if not entry.get("enabled", True):
            continue
        state_key = entry.get("state")
        if not state_key:
            continue
        register_state_handler(str(state_key), _VENDOR_DISPATCH_MODULE)


def _module_exists(module_path: str) -> bool:
    try:
        return importlib.util.find_spec(module_path) is not None
    except Exception:
        return False


def get_state_handler_module_path(state_key: str) -> str:
    """Resolve the state handler module path, falling back to shared scaffold."""
    normalized = normalize_state_name(state_key)
    if not normalized:
        return DEFAULT_STATE_HANDLER

    override = _STATE_HANDLER_OVERRIDES.get(normalized)
    if override and _module_exists(override):
        return override

    module_path = STATE_MODULE_MAP.get(normalized)
    if module_path and _module_exists(module_path):
        return module_path

    return DEFAULT_STATE_HANDLER


def get_county_handler_module_path(state_key: str, county_key: str) -> Optional[str]:
    """Resolve the county handler module path if registered and available."""
    normalized_state = normalize_state_name(state_key)
    normalized_county = normalize_county_name(county_key)
    if not normalized_state or not normalized_county:
        return None

    override = _COUNTY_HANDLER_OVERRIDES.get(normalized_state, {}).get(normalized_county)
    if override and _module_exists(override):
        return override

    return None

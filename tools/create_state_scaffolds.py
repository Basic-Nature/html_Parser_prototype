"""
Create per-state scaffold handler files under `webapp/parser/handlers/states/<state>/<state>.py` when missing.
Each scaffold delegates to the `html_dynamic_fallback.parse` handler so the site can be parsed via the generic fallback.

Run: python tools/create_state_scaffolds.py
"""
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATES_DIR = ROOT / 'webapp' / 'parser' / 'handlers' / 'states'
CONSTANTS = 'webapp.parser.Context_Integration.Context_Library.constants'

TEMPLATE = '''from __future__ import annotations

from typing import Any, Dict

from webapp.parser.handlers.formats.html_dynamic_fallback import parse as dynamic_parse


def parse(page: Any = None, html_context: Dict[str, Any] | None = None, coordinator: Any = None, context: Dict[str, Any] | None = None, session_id: str | None = None, **kwargs):
    """State scaffold handler that delegates to the dynamic HTML fallback.
    This file was auto-generated. Replace with a state-specific implementation when ready.
    """
    ctx = html_context or (context or {})
    return dynamic_parse(page=page, coordinator=coordinator, context=ctx, session_id=session_id, **kwargs)
'''


def main():
    states = []
    # Try a few robust import strategies to reach the constants mapping
    try:
        # 1) import as a module path (module may expose mapping directly)
        const_mod = importlib.import_module(CONSTANTS)
        states = list(getattr(const_mod, 'KNOWN_STATE_TO_COUNTY_MAP', {}).keys())
    except Exception:
        try:
            # 2) import parent module and access attribute
            parent = importlib.import_module('webapp.parser.Context_Integration.Context_Library')
            const_obj = getattr(parent, 'constants', None) or parent
            states = list(getattr(const_obj, 'KNOWN_STATE_TO_COUNTY_MAP', {}).keys())
        except Exception:
            try:
                # 3) direct attribute import style
                from webapp.parser.Context_Integration.Context_Library import (
                    constants as const_attr,  # type: ignore
                )
                states = list(getattr(const_attr, 'KNOWN_STATE_TO_COUNTY_MAP', {}).keys())
            except Exception:
                # Final fallback: try reading constants.py directly from workspace
                try:
                    const_path = ROOT / 'webapp' / 'parser' / 'Context_Integration' / 'Context_Library' / 'constants.py'
                    if const_path.exists():
                        txt = const_path.read_text(encoding='utf-8')
                        ns = {}
                        exec(txt, ns)
                        states = list(ns.get('KNOWN_STATE_TO_COUNTY_MAP', {}).keys())
                except Exception:
                    pass
                if not states:
                    print('Could not import constants; falling back to a small default state list')
                    states = ['alabama','alaska','arizona','arkansas','california','colorado','connecticut','delaware']

    created = []
    for state in states:
        state_dir = STATES_DIR / state
        state_dir.mkdir(parents=True, exist_ok=True)
        target = state_dir / f"{state}.py"
        if not target.exists():
            target.write_text(TEMPLATE, encoding='utf-8')
            created.append(str(target.relative_to(ROOT)))
    if created:
        print('Created scaffolds:')
        for p in created:
            print(' -', p)
    else:
        print('No scaffolds needed; all state handlers present.')

if __name__ == '__main__':
    main()

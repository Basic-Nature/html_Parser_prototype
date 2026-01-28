"""Normalize handler.parse signatures to canonical form (safe, non-typed).

This script scans `webapp/parser/handlers/states/**` and replaces multi-line
`def parse(...):` headers that don't match canonical parameter order with:

def parse(page=None, html_context=None, coordinator=None, context=None, session_id=None, **kwargs):

It preserves the function body unchanged. Run as:

    python tools/normalize_parse_signatures.py

Always review diffs before committing.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
STATES_DIR = ROOT / 'webapp' / 'parser' / 'handlers' / 'states'

CANON_SIG = "def parse(page=None, html_context=None, coordinator=None, context=None, session_id=None, **kwargs):"

SIG_RE = re.compile(r"def\s+parse\s*\((.*?)\)\s*(?:->\s*[^:]+)?\s*:", re.S)

changed = []
for py in STATES_DIR.rglob('*.py'):
    try:
        src = py.read_text(encoding='utf-8')
    except Exception:
        continue
    m = SIG_RE.search(src)
    if not m:
        continue
    args_block = m.group(1)
    # Normalize whitespace and param names
    params = [p.strip() for p in re.split(r",\s*(?![^()]*\))", args_block) if p.strip()]
    # canonical param names set (without types/defaults)
    canonical_names = ['page','html_context','coordinator','context','session_id']
    param_names = [re.split(r"=|:\s*", p)[0].strip() for p in params]
    # check if already canonical (in order) - allow **kwargs anywhere
    filtered = [n for n in param_names if n and not n.startswith('**')]
    if filtered[:len(canonical_names)] == canonical_names:
        continue
    # Replace the entire signature with canonical non-typed signature
    new_src = SIG_RE.sub(CANON_SIG, src, count=1)
    py.write_text(new_src, encoding='utf-8')
    changed.append(str(py.relative_to(ROOT)))

if changed:
    print('Updated signatures in:')
    for p in changed:
        print(' -', p)
else:
    print('No signature changes necessary.')

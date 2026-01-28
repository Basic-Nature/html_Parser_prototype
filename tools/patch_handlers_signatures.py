"""
Utility: inspect handler modules and optionally patch parse() signatures to accept
a flexible signature like (page, html_context=None, coordinator=None, context=None, session_id=None, **kwargs).

This script is conservative: it only prints suggested changes unless run with --apply.
"""
import ast
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HANDLERS_DIR = ROOT / 'webapp' / 'parser' / 'handlers'


def find_parse_defs(path: Path):
    try:
        src = path.read_text(encoding='utf-8')
    except Exception:
        return None
    try:
        tree = ast.parse(src)
    except Exception:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'parse':
            return node
    return None


def main(apply: bool = False):
    found = []
    for p in HANDLERS_DIR.rglob('*.py'):
        node = find_parse_defs(p)
        if node is not None:
            args = [a.arg for a in node.args.args]
            found.append((p.relative_to(ROOT), args))
    if not found:
        print('No handler.parse functions found under handlers/.')
        return 0
    for path, args in found:
        print(f'{path}: parse signature args = {args}')
    if apply:
        print('Apply mode requested, but this script is conservative and does not auto-modify files yet.')
    return 0


if __name__ == '__main__':
    apply_flag = '--apply' in sys.argv
    sys.exit(main(apply_flag))

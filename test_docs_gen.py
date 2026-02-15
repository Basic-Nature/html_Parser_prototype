#!/usr/bin/env python3
"""Quick test for docs generation."""
import traceback
from pathlib import Path

# Monkey-patch the function to show what's happening
from webapp.parser.utils import shared_logic

original_func = shared_logic.generate_project_audit

def patched_audit(*args, **kwargs):
    try:
        root = Path(args[0] if args else ".").resolve()
        print(f"  Scanning modules in {root}/webapp...")
        modules = shared_logic._scan_webapp_modules(root)
        print(f"  Found {len(modules)} modules")
        print("  Indexing defs...")
        def_index = shared_logic._index_defs(modules)
        print(f"  Found {len(def_index)} definitions")
        print("  Resolving targets...")
        edges, inbound = shared_logic._resolve_targets(modules, def_index)
        print(f"  Found {len(edges)} edges")
        print("  Rendering markdown...")
        md = shared_logic._render_audit_md(modules, def_index, edges, inbound, root)
        print(f"  Markdown length: {len(md)} chars")
        out = (root / (args[1] if len(args) > 1 else "docs/DEVELOPMENT/project_audit.md")).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"  Written to {out}")
        return True
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        traceback.print_exc()
        return False

shared_logic.generate_project_audit = patched_audit

from webapp.parser.utils.shared_logic import generate_project_audit

try:
    root = Path(".").resolve()
    print("Testing generate_project_audit...")
    audit_ok = generate_project_audit(root, "docs/DEVELOPMENT/project_audit.md")
    print(f"  audit_ok: {audit_ok}")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()

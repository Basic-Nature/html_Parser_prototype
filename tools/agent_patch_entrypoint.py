"""CLI entrypoint for applying patches with per-principal isolation.

Usage:
  PYTHONPATH=. CERT_PRINCIPAL="<cert_or_user>" \
  python tools/agent_patch_entrypoint.py --json patches.json

Patch file format (JSON):
{
  "files": {"relative/path.txt": "content"},
  "allow_suspicious": false
}

Exit codes:
  0 on success or skipped (already applied)
  2 on suspicious/terminate condition
  3 on errors applying patch
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tools.agent_patch_integration import apply_patch_if_needed


def load_payload(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main():
    parser = argparse.ArgumentParser(description="Apply patch with per-principal isolation")
    parser.add_argument("--json", required=True, help="Path to JSON file containing 'files' mapping")
    parser.add_argument("--principal", default=None, help="Principal/certificate identity (overrides CERT_PRINCIPAL)")
    parser.add_argument("--allow-suspicious", action="store_true", help="Allow static/html/js/css patches")
    args = parser.parse_args()

    payload = load_payload(Path(args.json))
    files = payload.get("files") or {}
    if not isinstance(files, dict) or not files:
        print("No files supplied in payload", file=sys.stderr)
        sys.exit(3)

    principal = args.principal or os.environ.get("CERT_PRINCIPAL") or os.environ.get("USER_PRINCIPAL")
    allow_suspicious = bool(payload.get("allow_suspicious") or args.allow_suspicious)

    res = apply_patch_if_needed(
        files,
        allow_suspicious=allow_suspicious,
        principal=principal,
    )

    print(json.dumps(res, indent=2))

    if res.get("terminate") or res.get("errors"):
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()

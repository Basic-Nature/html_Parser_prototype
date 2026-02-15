#!/usr/bin/env python3
"""Quick test to verify constants.py imports."""
import sys

try:
    print("Importing constants...")
    from webapp.parser.Context_Integration.Context_Library.constants import (
        CONTEST_KEYWORDS,
        DIVISION_TYPES,
        PARTY_CODE_DESCRIPTIONS,
        PARTY_CODE_MAP,
    )
    print(f"✓ PARTY_CODE_MAP: {len(PARTY_CODE_MAP)} items")
    print(f"✓ PARTY_CODE_DESCRIPTIONS: {len(PARTY_CODE_DESCRIPTIONS)} items")
    print(f"✓ DIVISION_TYPES: {len(DIVISION_TYPES)} items")
    print(f"✓ CONTEST_KEYWORDS: {len(CONTEST_KEYWORDS)} items")
    print("✓ All constants imported successfully!")
except Exception as e:
    print(f"ERROR during import: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


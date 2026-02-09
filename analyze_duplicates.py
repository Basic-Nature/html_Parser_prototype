#!/usr/bin/env python3.12
"""Analyze keyword duplicates across all constants categories and vocab files."""

from pathlib import Path

from webapp.parser.Context_Integration.Context_Library.constants import (
    ALLOWED_LABELS,
    BALLOT_MEASURE_TYPES,
    BALLOT_TYPES,
    BUTTON_CLASSES,
    BUTTON_TAGS,
    CONTEST_HEADER_KEYWORDS,
    CONTEST_HEADER_PREFERENCE,
    CONTEST_PANEL_TAGS,
    CONTEST_TITLE_KEYWORDS,
    CONTEST_TITLE_TAGS,
    CONTAINER_EXTRA_KEYWORDS,
    CONTAINER_FALLBACK_SELECTORS,
    ELECTION_ENTITY_LABELS,
    ELECTION_OFFICIAL_KEYWORDS,
    ELECTION_TYPES,
    EXTRA_HEADING_TAGS,
    HEADING_CLASSES,
    HEADING_TAGS,
    HTML_TAGS,
    ICON_CLASSES,
    ICON_TAGS,
    JURISDICTION_KEYWORDS,
    KEYWORD_TAXONOMY,
    LIKELY_ROW_CLASSES,
    LOCATION_KEYWORDS,
    NLP_SKIP_PHRASES,
    PANEL_CLASSES,
    PANEL_TAGS,
    PERCENT_KEYWORDS,
    RESULTS_KEYWORDS,
    STATE_TAGS,
    STRUCTURAL_TAGS,
    TABLE_TAGS,
    TOTAL_KEYWORDS,
    UPDATE_PANEL_KEYWORDS,
    VIEW_BY_PHRASES,
)

VOCAB_ROOT = Path(__file__).resolve().parent / "webapp" / "parser" / "Context_Integration" / "vocab"


def _parse_vocab_file(path: Path) -> set[str]:
    """Parse a vocab file into a set of tokens for duplicate analysis."""
    tokens: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "->" in line:
            left, right = (part.strip() for part in line.split("->", 1))
            if left:
                tokens.add(left)
            if right:
                tokens.add(right)
            continue
        if "|" in line and path.name.endswith("_overrides.txt"):
            left, right = (part.strip() for part in line.split("|", 1))
            if left:
                tokens.add(left)
            if right:
                tokens.add(right)
            continue
        tokens.add(line)
    return tokens


def _collect_vocab_categories() -> dict[str, set[str]]:
    """Load vocab entities/validators into categories keyed by file name."""
    categories: dict[str, set[str]] = {}
    if not VOCAB_ROOT.exists():
        return categories
    for subdir in ("entities", "validators"):
        base = VOCAB_ROOT / subdir
        if not base.exists():
            continue
        for path in sorted(base.glob("*.txt")):
            category = f"VOCAB:{subdir}/{path.name}"
            categories[category] = _parse_vocab_file(path)
    return categories

# Collect all keyword sets/lists  
all_categories = {
    "PANEL_TAGS": PANEL_TAGS,
    "TABLE_TAGS": TABLE_TAGS,
    "STATE_TAGS": STATE_TAGS,
    "HEADING_TAGS": HEADING_TAGS,
    "EXTRA_HEADING_TAGS": EXTRA_HEADING_TAGS,
    "CONTAINER_EXTRA_KEYWORDS": set(CONTAINER_EXTRA_KEYWORDS),
    "CONTAINER_FALLBACK_SELECTORS": set(CONTAINER_FALLBACK_SELECTORS),
    "BUTTON_TAGS": BUTTON_TAGS,
    "ICON_TAGS": ICON_TAGS,
    "PANEL_CLASSES": PANEL_CLASSES,
    "STRUCTURAL_TAGS": STRUCTURAL_TAGS,
    "CONTEST_PANEL_TAGS": CONTEST_PANEL_TAGS,
    "HTML_TAGS": HTML_TAGS,
    "HEADING_CLASSES": HEADING_CLASSES,
    "ICON_CLASSES": ICON_CLASSES,
    "BUTTON_CLASSES": BUTTON_CLASSES,
    "LIKELY_ROW_CLASSES": set(LIKELY_ROW_CLASSES),
    "VIEW_BY_PHRASES": set(VIEW_BY_PHRASES),
    "NLP_SKIP_PHRASES": set(NLP_SKIP_PHRASES),
    "LOCATION_KEYWORDS": set(LOCATION_KEYWORDS),
    "BALLOT_TYPES": set(BALLOT_TYPES),
    "ELECTION_TYPES": set(ELECTION_TYPES),
    "ELECTION_ENTITY_LABELS": set(ELECTION_ENTITY_LABELS),
    "BALLOT_MEASURE_TYPES": set(BALLOT_MEASURE_TYPES),
    "TOTAL_KEYWORDS": set(TOTAL_KEYWORDS),
    "PERCENT_KEYWORDS": set(PERCENT_KEYWORDS),
    "JURISDICTION_KEYWORDS": set(JURISDICTION_KEYWORDS),
    "RESULTS_KEYWORDS": set(RESULTS_KEYWORDS),
    "ELECTION_OFFICIAL_KEYWORDS": set(ELECTION_OFFICIAL_KEYWORDS),
    "UPDATE_PANEL_KEYWORDS": set(UPDATE_PANEL_KEYWORDS),
    "CONTEST_TITLE_TAGS": set(CONTEST_TITLE_TAGS),
    "CONTEST_TITLE_KEYWORDS": set(CONTEST_TITLE_KEYWORDS),
    "CONTEST_HEADER_KEYWORDS": set(CONTEST_HEADER_KEYWORDS),
    "CONTEST_HEADER_PREFERENCE": set(CONTEST_HEADER_PREFERENCE),
    "ALLOWED_LABELS": set(ALLOWED_LABELS),
    "KEYWORD_TAXONOMY_KEYS": set(KEYWORD_TAXONOMY),
}

all_categories.update(_collect_vocab_categories())

# Find duplicates/overlaps
print("=== KEYWORD TAXONOMY & DUPLICATES ===\n")
keyword_to_categories = {}
for category, keywords in all_categories.items():
    for kw in keywords:
        if kw not in keyword_to_categories:
            keyword_to_categories[kw] = []
        keyword_to_categories[kw].append(category)

# Print duplicates
duplicates = {kw: cats for kw, cats in keyword_to_categories.items() if len(cats) > 1}
print(f"Found {len(duplicates)} keywords appearing in multiple categories:\n")
for kw in sorted(duplicates.keys()):
    print(f"  '{kw}': {duplicates[kw]}")

# Statistics
print(f"\n=== STATISTICS ===")
total_unique = len(keyword_to_categories)
total_items = sum(len(v) for v in all_categories.values())
print(f"Total keywords across all categories: {total_items}")
print(f"Total UNIQUE keywords: {total_unique}")
print(f"Redundancy: {100 * (1 - total_unique/total_items):.1f}%")

# Keywords that appear 3+ times
print(f"\n=== HIGH-REDUNDANCY KEYWORDS (3+) ===")
high_redundancy = {kw: cats for kw, cats in keyword_to_categories.items() if len(cats) >= 3}
for kw in sorted(high_redundancy.keys()):
    print(f"  '{kw}': {high_redundancy[kw]}")

# Category sizes
print(f"\n=== CATEGORY SIZES ===")
for cat in sorted(all_categories.keys(), key=lambda x: len(all_categories[x]), reverse=True):
    print(f"  {cat}: {len(all_categories[cat])} items")
    
print(f"\n=== RECOMMENDATIONS FOR CONSOLIDATION ===")
print(f"1. Create KEYWORD_TAXONOMY mapping showing which keywords are in which categories")
print(f"2. Build contextual relationships for keywords that appear in multiple places")
print(f"3. Consider consolidating overlapping sets into a unified 'SEMANTIC_KEYWORDS' map")
print(f"4. Track keyword-to-label mappings for ML pipeline enrichment")

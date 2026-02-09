#!/usr/bin/env python3
"""
Export all remaining constants from constants.py to vocab files.
Complements export_constants_vocab_full.py; targets party, location, HTML, noise, contest, camelot.

Usage:
    python scripts/export_all_constants_vocab.py
"""
import sys
from pathlib import Path

# Add workspace root to path
WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from webapp.parser.Context_Integration.Context_Library import constants


def export_party_canonical():
    """Export _PARTY_CANON_MAP to entities/party_canonical.txt"""
    filepath = WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "entities" / "party_canonical.txt"
    lines = []
    for key, value in sorted(constants._PARTY_CANON_MAP.items()):
        lines.append(key)
    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ Exported {len(lines)} party canonical names to {filepath.name}")


def export_party_codes():
    """Export PARTY_CODE_MAP to validators/party_codes.txt"""
    filepath = WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "validators" / "party_codes.txt"
    lines = []
    for code, canonical in sorted(constants.PARTY_CODE_MAP.items()):
        lines.append(f"{code} -> {canonical}")
    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ Exported {len(lines)} party codes to {filepath.name}")


def export_party_code_descriptions():
    """Export PARTY_CODE_DESCRIPTIONS to validators/party_code_descriptions.txt"""
    filepath = WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "validators" / "party_code_descriptions.txt"
    lines = []
    for code, info in sorted(constants.PARTY_CODE_DESCRIPTIONS.items()):
        desc = info.get("description", "")
        notes = info.get("notes", "")
        # Format: CODE -> Description | Notes
        value = desc
        if notes:
            value = f"{desc} | {notes}"
        lines.append(f"{code} -> {value}")
    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ Exported {len(lines)} party code descriptions to {filepath.name}")


def export_location_synonyms():
    """Export LOCATION_SYNONYM_MAP to validators/location_synonyms.txt"""
    filepath = WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "validators" / "location_synonyms.txt"
    lines = []
    for syn, canonical in sorted(constants.LOCATION_SYNONYM_MAP.items()):
        lines.append(f"{syn} -> {canonical}")
    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ Exported {len(lines)} location synonyms to {filepath.name}")


def export_location_abbreviations():
    """Export LOCATION_ABBREVIATIONS to validators/location_abbreviations.txt as abbr -> comma-separated expansions"""
    filepath = WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "validators" / "location_abbreviations.txt"
    lines = []
    for abbr, expansions in sorted(constants.LOCATION_ABBREVIATIONS.items()):
        # Format: abbr -> expansion1, expansion2, ...
        value = ", ".join(expansions)
        lines.append(f"{abbr} -> {value}")
    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ Exported {len(lines)} location abbreviations to {filepath.name}")


def export_table_builder_priority():
    """Export TABLE_BUILDER_LOCATION_PRIORITY to entities/table_builder_location_priority.txt"""
    filepath = WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "entities" / "table_builder_location_priority.txt"
    lines = list(constants.TABLE_BUILDER_LOCATION_PRIORITY)
    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ Exported {len(lines)} table builder location priorities to {filepath.name}")


def export_table_builder_tokens():
    """Export TABLE_BUILDER_LOCATION_TOKENS to entities/table_builder_location_tokens.txt"""
    filepath = WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "entities" / "table_builder_location_tokens.txt"
    lines = list(constants.TABLE_BUILDER_LOCATION_TOKENS)
    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ Exported {len(lines)} table builder location tokens to {filepath.name}")


def export_table_builder_suffixes():
    """Export TABLE_BUILDER_CANDIDATE_SUFFIXES to entities/table_builder_candidate_suffixes.txt"""
    filepath = WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "entities" / "table_builder_candidate_suffixes.txt"
    lines = list(constants.TABLE_BUILDER_CANDIDATE_SUFFIXES)
    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ Exported {len(lines)} table builder candidate suffixes to {filepath.name}")


def export_html_tags():
    """Export HTML tag sets to separate files"""
    exports = {
        "html_tags.txt": constants.HTML_TAGS,
        "panel_tags.txt": constants.PANEL_TAGS,
        "table_tags.txt": constants.TABLE_TAGS,
        "state_tags.txt": constants.STATE_TAGS,
        "button_tags.txt": constants.BUTTON_TAGS,
        "heading_tags.txt": constants.HEADING_TAGS,
        "extra_heading_tags.txt": constants.EXTRA_HEADING_TAGS,
    }
    for filename, tag_set in exports.items():
        filepath = WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "entities" / filename
        lines = sorted(tag_set)
        filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"✓ Exported {len(lines)} tags to {filename}")


def export_ignore_lists():
    """Export ALWAYS_IGNORE_* to validators"""
    exports = {
        "ignore_tags.txt": constants.ALWAYS_IGNORE_TAGS,
        "ignore_classes.txt": constants.ALWAYS_IGNORE_CLASSES,
        "ignore_ids.txt": constants.ALWAYS_IGNORE_IDS,
    }
    for filename, ignore_set in exports.items():
        filepath = WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "validators" / filename
        lines = sorted(ignore_set)
        filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"✓ Exported {len(lines)} items to {filename}")


def export_icon_button_heading_classes():
    """Export icon/button/heading/panel classes to separate files"""
    exports = {
        "icon_classes.txt": constants.ICON_CLASSES,
        "button_classes.txt": constants.BUTTON_CLASSES,
        "heading_classes.txt": constants.HEADING_CLASSES,
        "panel_classes.txt": constants.PANEL_CLASSES,
        "timestamp_classes.txt": constants.TIMESTAMP_CLASSES,
    }
    for filename, class_set in exports.items():
        filepath = WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "validators" / filename
        lines = sorted(class_set)
        filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"✓ Exported {len(lines)} classes to {filename}")


def export_structural():
    """Export structural tag sets"""
    exports = {
        "structural_tags.txt": constants.STRUCTURAL_TAGS,
        "icon_tags.txt": constants.ICON_TAGS,
        "root_container_tags.txt": constants.ROOT_CONTAINER_TAGS,
    }
    for filename, tag_set in exports.items():
        filepath = WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "entities" / filename
        lines = sorted(tag_set)
        filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"✓ Exported {len(lines)} tags to {filename}")


def export_nlp_phrases():
    """Export NLP-related phrase lists"""
    exports = {
        "nlp_skip_phrases.txt": constants.NLP_SKIP_PHRASES,
        "view_by_phrases.txt": constants.VIEW_BY_PHRASES,
        "update_panel_keywords.txt": constants.UPDATE_PANEL_KEYWORDS,
    }
    for filename, phrase_list in exports.items():
        filepath = WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "entities" / filename
        lines = list(phrase_list)
        filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"✓ Exported {len(lines)} phrases to {filename}")


def export_contest_keywords():
    """Export contest/office/election related keywords"""
    exports = {
        "office_keywords.txt": [f"{kw} -> {cat}" for kw, cat in constants.OFFICE_KEYWORDS],
        "contest_title_skip_phrases.txt": constants.CONTEST_TITLE_SKIP_PHRASES,
        "contest_header_keywords.txt": constants.CONTEST_HEADER_KEYWORDS,
        "contest_header_preference.txt": constants.CONTEST_HEADER_PREFERENCE,
    }
    for filename, data in exports.items():
        filepath = WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "entities" / filename
        if isinstance(data, list):
            lines = data
        else:
            lines = list(data)
        filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"✓ Exported {len(lines)} items to {filename}")


def export_pattern_lists():
    """Export misaligned and noisy label patterns"""
    # MISALIGNED_PATTERNS is already a list of regex strings
    filepath_misaligned = WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "validators" / "misaligned_patterns.txt"
    filepath_misaligned.write_text("\n".join(constants.MISALIGNED_PATTERNS) + "\n", encoding="utf-8")
    print(f"✓ Exported {len(constants.MISALIGNED_PATTERNS)} misaligned patterns to misaligned_patterns.txt")
    
    # NOISY_LABEL_PATTERNS is a list of compiled regex; extract pattern strings
    filepath_noisy = WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "validators" / "noisy_label_patterns.txt"
    noisy_patterns = [p.pattern for p in constants.NOISY_LABEL_PATTERNS]
    filepath_noisy.write_text("\n".join(noisy_patterns) + "\n", encoding="utf-8")
    print(f"✓ Exported {len(noisy_patterns)} noisy label patterns to noisy_label_patterns.txt")


def export_camelot_noise():
    """Export Camelot noise categories"""
    # CAMELOT_NOISE_CATEGORIES is a dict[str, list[str]]
    for category, patterns in constants.CAMELOT_NOISE_CATEGORIES.items():
        filename = f"camelot_noise_{category}.txt"
        filepath = WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "validators" / filename
        filepath.write_text("\n".join(patterns) + "\n", encoding="utf-8")
        print(f"✓ Exported {len(patterns)} {category} camelot patterns to {filename}")
    
    # CAMELOT_STATE_NOISE_OVERRIDES
    filepath_state = WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "validators" / "camelot_state_overrides.txt"
    lines = []
    for state, categories in constants.CAMELOT_STATE_NOISE_OVERRIDES.items():
        for cat, patterns in categories.items():
            for pat in patterns:
                lines.append(f"{state}|{cat} -> {pat}")
    filepath_state.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ Exported {len(lines)} state-level camelot overrides to camelot_state_overrides.txt")
    
    # CAMELOT_COUNTY_NOISE_OVERRIDES
    filepath_county = WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "validators" / "camelot_county_overrides.txt"
    lines = []
    for (state, county), categories in constants.CAMELOT_COUNTY_NOISE_OVERRIDES.items():
        for cat, patterns in categories.items():
            for pat in patterns:
                lines.append(f"{state}|{county}|{cat} -> {pat}")
    filepath_county.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ Exported {len(lines)} county-level camelot overrides to camelot_county_overrides.txt")


if __name__ == "__main__":
    print("Exporting all remaining constants to vocab files...")
    print()
    
    print("=== Party Canonical & Codes ===")
    export_party_canonical()
    export_party_codes()
    export_party_code_descriptions()
    print()
    
    print("=== Location & Table Heuristics ===")
    export_location_synonyms()
    export_location_abbreviations()
    export_table_builder_priority()
    export_table_builder_tokens()
    export_table_builder_suffixes()
    print()
    
    print("=== HTML/DOM Tags & Classes ===")
    export_html_tags()
    export_ignore_lists()
    export_icon_button_heading_classes()
    export_structural()
    print()
    
    print("=== NLP Phrases & Keywords ===")
    export_nlp_phrases()
    export_contest_keywords()
    export_pattern_lists()
    print()
    
    print("=== Camelot Noise Patterns ===")
    export_camelot_noise()
    print()
    
    print("✅ All vocab exports complete!")

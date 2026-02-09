#!/usr/bin/env python3
"""
Export remaining constants to vocab by directly parsing constants.py.
Avoids heavy imports (sentence_transformers, etc.).
"""
import ast
import re
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).parent.parent
CONSTANTS_FILE = WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "Context_Library" / "constants.py"


def extract_dict_from_code(code_lines: str, dict_name: str) -> dict:
    """Extract a Python dict from source code."""
    try:
        tree = ast.parse(code_lines)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == dict_name:
                        return ast.literal_eval(ast.unparse(node.value))
    except Exception:
        pass
    return {}


def extract_set_from_code(code_lines: str, set_name: str) -> set:
    """Extract a Python set from source code."""
    try:
        tree = ast.parse(code_lines)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == set_name:
                        val = ast.literal_eval(ast.unparse(node.value))
                        return val if isinstance(val, set) else set(val)
    except Exception:
        pass
    return set()


def extract_list_from_code(code_lines: str, list_name: str) -> list:
    """Extract a Python list from source code."""
    try:
        tree = ast.parse(code_lines)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == list_name:
                        val = ast.literal_eval(ast.unparse(node.value))
                        return list(val) if isinstance(val, (list, tuple, set)) else [val]
    except Exception:
        pass
    return []


def write_vocab_file(filepath: Path, lines: list, description: str = ""):
    """Write vocab file with header comment."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + "\n"
    filepath.write_text(content, encoding="utf-8")
    count = len([l for l in lines if l.strip() and not l.startswith("#")])
    print(f"✓ Exported {count} items to {filepath.name}")


# Read constants file
constants_code = CONSTANTS_FILE.read_text(encoding="utf-8")

print("Exporting constants to vocab files (direct parsing)...")
print()

# Party canonical
print("=== Party Canonical & Codes ===")
try:
    party_canon_map = extract_dict_from_code(constants_code, "_PARTY_CANON_MAP")
    if party_canon_map:
        lines = list(party_canon_map.keys())
        write_vocab_file(
            WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "entities" / "party_canonical.txt",
            sorted(lines)
        )
except Exception as e:
    print(f"✗ Error exporting party canonical: {e}")

try:
    party_code_map = extract_dict_from_code(constants_code, "PARTY_CODE_MAP")
    if party_code_map:
        lines = [f"{code} -> {canonical}" for code, canonical in sorted(party_code_map.items())]
        write_vocab_file(
            WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "validators" / "party_codes.txt",
            lines
        )
except Exception as e:
    print(f"✗ Error exporting party codes: {e}")

try:
    party_code_desc = extract_dict_from_code(constants_code, "PARTY_CODE_DESCRIPTIONS")
    if party_code_desc:
        lines = []
        for code, info in sorted(party_code_desc.items()):
            desc = info.get("description", "")
            notes = info.get("notes", "")
            value = desc
            if notes:
                value = f"{desc} | {notes}"
            lines.append(f"{code} -> {value}")
        write_vocab_file(
            WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "validators" / "party_code_descriptions.txt",
            lines
        )
except Exception as e:
    print(f"✗ Error exporting party descriptions: {e}")

print()

# Location heuristics
print("=== Location & Table Heuristics ===")
try:
    loc_syn_map = extract_dict_from_code(constants_code, "LOCATION_SYNONYM_MAP")
    if loc_syn_map:
        lines = [f"{syn} -> {canonical}" for syn, canonical in sorted(loc_syn_map.items())]
        write_vocab_file(
            WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "validators" / "location_synonyms.txt",
            lines
        )
except Exception as e:
    print(f"✗ Error exporting location synonyms: {e}")

try:
    loc_abbr = extract_dict_from_code(constants_code, "LOCATION_ABBREVIATIONS")
    if loc_abbr:
        lines = [f"{abbr} -> {', '.join(expansions)}" for abbr, expansions in sorted(loc_abbr.items())]
        write_vocab_file(
            WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "validators" / "location_abbreviations.txt",
            lines
        )
except Exception as e:
    print(f"✗ Error exporting location abbreviations: {e}")

try:
    table_priority = extract_set_from_code(constants_code, "TABLE_BUILDER_LOCATION_PRIORITY")
    if not table_priority:
        table_priority = extract_list_from_code(constants_code, "TABLE_BUILDER_LOCATION_PRIORITY")
    if table_priority:
        lines = sorted(list(table_priority))
        write_vocab_file(
            WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "entities" / "table_builder_location_priority.txt",
            lines
        )
except Exception as e:
    print(f"✗ Error exporting table builder priority: {e}")

try:
    table_tokens = extract_set_from_code(constants_code, "TABLE_BUILDER_LOCATION_TOKENS")
    if not table_tokens:
        table_tokens = extract_list_from_code(constants_code, "TABLE_BUILDER_LOCATION_TOKENS")
    if table_tokens:
        lines = sorted(list(table_tokens))
        write_vocab_file(
            WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "entities" / "table_builder_location_tokens.txt",
            lines
        )
except Exception as e:
    print(f"✗ Error exporting table builder tokens: {e}")

try:
    table_suffixes = extract_set_from_code(constants_code, "TABLE_BUILDER_CANDIDATE_SUFFIXES")
    if not table_suffixes:
        table_suffixes = extract_list_from_code(constants_code, "TABLE_BUILDER_CANDIDATE_SUFFIXES")
    if table_suffixes:
        lines = sorted(list(table_suffixes))
        write_vocab_file(
            WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "entities" / "table_builder_candidate_suffixes.txt",
            lines
        )
except Exception as e:
    print(f"✗ Error exporting table builder suffixes: {e}")

print()

# HTML/DOM tags
print("=== HTML/DOM Tags & Classes ===")
tag_exports = {
    "html_tags.txt": "HTML_TAGS",
    "panel_tags.txt": "PANEL_TAGS",
    "table_tags.txt": "TABLE_TAGS",
    "state_tags.txt": "STATE_TAGS",
    "button_tags.txt": "BUTTON_TAGS",
    "heading_tags.txt": "HEADING_TAGS",
    "extra_heading_tags.txt": "EXTRA_HEADING_TAGS",
}
for filename, var_name in tag_exports.items():
    try:
        tag_set = extract_set_from_code(constants_code, var_name)
        if tag_set:
            lines = sorted(tag_set)
            write_vocab_file(
                WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "entities" / filename,
                lines
            )
    except Exception as e:
        print(f"✗ Error exporting {var_name}: {e}")

# Ignore lists
print()
ignore_exports = {
    "ignore_tags.txt": "ALWAYS_IGNORE_TAGS",
    "ignore_classes.txt": "ALWAYS_IGNORE_CLASSES",
    "ignore_ids.txt": "ALWAYS_IGNORE_IDS",
}
for filename, var_name in ignore_exports.items():
    try:
        ignore_set = extract_set_from_code(constants_code, var_name)
        if ignore_set:
            lines = sorted(ignore_set)
            write_vocab_file(
                WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "validators" / filename,
                lines
            )
    except Exception as e:
        print(f"✗ Error exporting {var_name}: {e}")

# Class sets
print()
class_exports = {
    "icon_classes.txt": "ICON_CLASSES",
    "button_classes.txt": "BUTTON_CLASSES",
    "heading_classes.txt": "HEADING_CLASSES",
    "panel_classes.txt": "PANEL_CLASSES",
    "timestamp_classes.txt": "TIMESTAMP_CLASSES",
}
for filename, var_name in class_exports.items():
    try:
        class_set = extract_set_from_code(constants_code, var_name)
        if class_set:
            lines = sorted(class_set)
            write_vocab_file(
                WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "validators" / filename,
                lines
            )
    except Exception as e:
        print(f"✗ Error exporting {var_name}: {e}")

# Structural tags
print()
struct_exports = {
    "structural_tags.txt": "STRUCTURAL_TAGS",
    "icon_tags.txt": "ICON_TAGS",
    "root_container_tags.txt": "ROOT_CONTAINER_TAGS",
}
for filename, var_name in struct_exports.items():
    try:
        tag_set = extract_set_from_code(constants_code, var_name)
        if tag_set:
            lines = sorted(tag_set)
            write_vocab_file(
                WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "entities" / filename,
                lines
            )
    except Exception as e:
        print(f"✗ Error exporting {var_name}: {e}")

print()
print("=== NLP Phrases & Keywords ===")

phrase_exports = {
    "nlp_skip_phrases.txt": "NLP_SKIP_PHRASES",
    "view_by_phrases.txt": "VIEW_BY_PHRASES",
    "update_panel_keywords.txt": "UPDATE_PANEL_KEYWORDS",
}
for filename, var_name in phrase_exports.items():
    try:
        phrase_list = extract_list_from_code(constants_code, var_name)
        if phrase_list:
            write_vocab_file(
                WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "entities" / filename,
                phrase_list
            )
    except Exception as e:
        print(f"✗ Error exporting {var_name}: {e}")

print()
print("=== Contest/Office Keywords ===")

try:
    office_kw = extract_list_from_code(constants_code, "OFFICE_KEYWORDS")
    if office_kw:
        lines = [f"{kw} -> {cat}" for kw, cat in office_kw]
        write_vocab_file(
            WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "entities" / "office_keywords.txt",
            lines
        )
except Exception as e:
    print(f"✗ Error exporting OFFICE_KEYWORDS: {e}")

try:
    contest_skip = extract_set_from_code(constants_code, "CONTEST_TITLE_SKIP_PHRASES")
    if contest_skip:
        lines = sorted(contest_skip)
        write_vocab_file(
            WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "entities" / "contest_title_skip_phrases.txt",
            lines
        )
except Exception as e:
    print(f"✗ Error exporting CONTEST_TITLE_SKIP_PHRASES: {e}")

try:
    contest_hdr = extract_set_from_code(constants_code, "CONTEST_HEADER_KEYWORDS")
    if contest_hdr:
        lines = sorted(contest_hdr)
        write_vocab_file(
            WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "entities" / "contest_header_keywords.txt",
            lines
        )
except Exception as e:
    print(f"✗ Error exporting CONTEST_HEADER_KEYWORDS: {e}")

try:
    contest_pref = extract_list_from_code(constants_code, "CONTEST_HEADER_PREFERENCE")
    if contest_pref:
        write_vocab_file(
            WORKSPACE_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab" / "entities" / "contest_header_preference.txt",
            contest_pref
        )
except Exception as e:
    print(f"✗ Error exporting CONTEST_HEADER_PREFERENCE: {e}")

print()
print("✅ Direct parsing export complete!")

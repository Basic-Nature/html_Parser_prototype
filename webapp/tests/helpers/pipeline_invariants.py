"""Reusable invariant assertions and diff helpers for pipeline contract tests."""
from __future__ import annotations

import pprint
from collections import Counter
from typing import Any


# ---------------------------------------------------------------------------
# Output-shape invariants
# ---------------------------------------------------------------------------

def assert_election_output_invariants(
    result: tuple,
    *,
    require_votes: bool = False,
    label: str = "",
) -> None:
    """Assert the canonical 4-tuple (headers, rows, contest, metadata) is well-formed.

    Args:
        result: The 4-tuple returned by a handler or safe_parse.
        require_votes: When True, assert every row has at least one numeric-looking value.
        label: Optional test label for failure messages.
    """
    prefix = f"[{label}] " if label else ""
    assert isinstance(result, tuple), f"{prefix}result must be a tuple, got {type(result)}"
    assert len(result) == 4, f"{prefix}result must be a 4-tuple, got {len(result)} items"

    headers, rows, contest, metadata = result

    # headers
    assert isinstance(headers, list), f"{prefix}headers must be a list, got {type(headers)}"
    assert all(isinstance(h, str) for h in headers), f"{prefix}all headers must be strings"

    # rows
    assert isinstance(rows, list), f"{prefix}rows must be a list, got {type(rows)}"
    assert all(isinstance(r, dict) for r in rows), f"{prefix}all rows must be dicts"

    # metadata
    assert isinstance(metadata, dict), f"{prefix}metadata must be a dict, got {type(metadata)}"
    assert "error" not in metadata, (
        f"{prefix}metadata contains error key: {metadata.get('error')} — "
        f"exception: {metadata.get('exception', '')}"
    )

    # optional vote sanity
    if require_votes and rows:
        for row in rows:
            values = list(row.values())
            numeric_found = any(
                _lookslike_number(v) for v in values
            )
            assert numeric_found, f"{prefix}row has no numeric vote value: {row}"


def assert_no_duplicate_keys(
    rows: list[dict],
    key_fields: tuple[str, ...] = ("contest", "county", "candidate", "party"),
    label: str = "",
) -> None:
    """Assert no two rows share the same composite key.

    Args:
        rows: List of row dicts.
        key_fields: Fields that together form a unique row key.
        label: Optional test label.
    """
    prefix = f"[{label}] " if label else ""
    keys = [
        tuple(r.get(f, "") for f in key_fields)
        for r in rows
    ]
    counts = Counter(keys)
    dupes = {k: v for k, v in counts.items() if v > 1}
    assert not dupes, (
        f"{prefix}duplicate composite keys found ({key_fields}): "
        + pprint.pformat(dupes)
    )


# ---------------------------------------------------------------------------
# Dual-source diff helper
# ---------------------------------------------------------------------------

def diff_dual_sources(
    parser_rows: list[dict],
    warehouse_rows: list[dict],
    key_fields: tuple[str, ...] = ("contest", "county", "candidate", "party"),
    value_fields: tuple[str, ...] = ("votes",),
) -> dict[str, list[dict[str, Any]]]:
    """Compare parser output rows against warehouse rows and return categorised diffs.

    Returns a dict with three keys:
      - ``missing_parser``: rows present in warehouse but absent from parser output.
      - ``missing_warehouse``: rows in parser output absent from warehouse.
      - ``mismatched_values``: rows present in both but with differing value_fields.

    The diff treats rows as equal if their key_fields match after lowercasing.
    """

    def _make_key(row: dict) -> tuple:
        return tuple(str(row.get(f, "")).strip().lower() for f in key_fields)

    parser_index: dict[tuple, dict] = {_make_key(r): r for r in parser_rows}
    warehouse_index: dict[tuple, dict] = {_make_key(r): r for r in warehouse_rows}

    missing_parser: list[dict] = []
    missing_warehouse: list[dict] = []
    mismatched_values: list[dict] = []

    all_keys = set(parser_index) | set(warehouse_index)
    for key in sorted(all_keys):
        in_parser = key in parser_index
        in_warehouse = key in warehouse_index

        if in_warehouse and not in_parser:
            missing_parser.append({"key": key, "warehouse_row": warehouse_index[key]})
        elif in_parser and not in_warehouse:
            missing_warehouse.append({"key": key, "parser_row": parser_index[key]})
        else:
            p_row = parser_index[key]
            w_row = warehouse_index[key]
            diffs = {}
            for f in value_fields:
                pv = p_row.get(f)
                wv = w_row.get(f)
                if str(pv).strip() != str(wv).strip():
                    diffs[f] = {"parser": pv, "warehouse": wv}
            if diffs:
                mismatched_values.append({
                    "key": key,
                    "diffs": diffs,
                    "parser_row": p_row,
                    "warehouse_row": w_row,
                })

    return {
        "missing_parser": missing_parser,
        "missing_warehouse": missing_warehouse,
        "mismatched_values": mismatched_values,
    }


def format_diff_report(diff: dict[str, list]) -> str:
    """Return a human-readable string summarising a diff_dual_sources result."""
    lines: list[str] = []
    for category, items in diff.items():
        lines.append(f"\n=== {category.upper().replace('_', ' ')} ({len(items)}) ===")
        for item in items[:20]:          # cap to 20 per category in reports
            lines.append(pprint.pformat(item, width=120))
        if len(items) > 20:
            lines.append(f"  ... and {len(items) - 20} more")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lookslike_number(v: Any) -> bool:
    """Return True if *v* looks like a vote count (int, float, or digit-only string)."""
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str):
        s = v.strip().replace(",", "")
        return s.lstrip("-").isdigit() and bool(s)
    return False

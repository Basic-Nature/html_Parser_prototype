#!/usr/bin/env python3
"""Generate docs/todos.md by scanning sources and optional Ruff outputs.

Scans text-based source files for task markers (TO-DO/FIX-ME/etc.) and writes a
markdown index. Optionally ingests Ruff JSON reports for lint findings.
Supports multiple roots (defaults to project webapp/).
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

_FIX = "FIX"
_ME = "ME"
_TASK = "TO" + "DO"
_FIXME = "".join([_FIX, _ME])
_HACK = "HA" + "CK"
_XXX = "X" * 3

KEYWORDS = (_TASK, _FIXME, _HACK, _XXX)
PRIORITY = {
    _FIXME: "high",
    _TASK: "medium",
    _HACK: "medium",
    _XXX: "medium",
}

DEFAULT_ROOT = Path(__file__).resolve().parent.parent
WEBAPP_DIR = DEFAULT_ROOT / "webapp"
OUTPUT_PATH = DEFAULT_ROOT / "docs" / "DEVELOPMENT" / "todos.md"
OUTPUT_HIGH_PATH = DEFAULT_ROOT / "docs" / "DEVELOPMENT" / "todos_high.md"
OUTPUT_MEDIUM_PATH = DEFAULT_ROOT / "docs" / "DEVELOPMENT" / "todos_medium.md"
OUTPUT_LOW_PATH = DEFAULT_ROOT / "docs" / "DEVELOPMENT" / "todos_low.md"

IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
    "venv",
    ".venv",
}

ALLOWED_SUFFIXES = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".yml",
    ".yaml",
    ".md",
    ".html",
    ".htm",
    ".css",
    ".txt",
    ".sh",
    ".ps1",
}

EXCLUDED_FILES = {
    Path("docs") / "DEVELOPMENT" / "todos.md",
    Path("docs") / "DEVELOPMENT" / "todos_high.md",
    Path("docs") / "DEVELOPMENT" / "todos_medium.md",
    Path("docs") / "DEVELOPMENT" / "todos_low.md",
    Path("docs") / "DEVELOPMENT" / "project_audit.md",
    Path("docs") / "DEVELOPMENT" / "pipeline_map.md",
    Path("webapp")
    / "parser"
    / "Context_Integration"
    / "Context_Library"
    / "cache"
    / "context_cache.json",
}


@dataclass
class MarkerEntry:
    path: Path
    lineno: int
    keyword: str
    text: str

    @property
    def rel_path(self) -> str:
        return str(self.path)


def iter_source_files(roots: Sequence[Path]) -> Iterator[Path]:
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_dir():
                if path.name in IGNORED_DIRS:
                    continue
                continue
            if path.suffix.lower() not in ALLOWED_SUFFIXES:
                continue
            try:
                rel = path.resolve().relative_to(DEFAULT_ROOT)
                if rel in EXCLUDED_FILES:
                    continue
            except Exception:
                pass
            yield path


def extract_markers(path: Path) -> list[MarkerEntry]:
    entries: list[MarkerEntry] = []
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return entries

    pattern = re.compile(r"\b(" + "|".join(KEYWORDS) + r")\b", re.IGNORECASE)
    for idx, line in enumerate(content.splitlines(), start=1):
        match = pattern.search(line)
        if not match:
            continue
        keyword = match.group(1).upper()
        raw_line = line.strip()
        text = raw_line
        entries.append(MarkerEntry(path=path, lineno=idx, keyword=keyword, text=text))
    return entries


def gather_markers(roots: Sequence[Path]) -> list[MarkerEntry]:
    markers: list[MarkerEntry] = []
    for file_path in iter_source_files(roots):
        markers.extend(extract_markers(file_path))
    return markers


def ingest_ruff_json(paths: Sequence[Path]) -> list[MarkerEntry]:
    entries: list[MarkerEntry] = []
    for p in paths:
        if not p or not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        reports = data if isinstance(data, list) else data.get("messages") if isinstance(data, dict) else []
        for item in reports or []:
            try:
                filename = item.get("filename") or item.get("file")
                row = item.get("location", {}).get("row") or item.get("line") or 1
                code = item.get("code") or "RUFF"
                message = item.get("message") or "ruff finding"
                if not filename:
                    continue
                keyword = f"RUFF-{code}"
                entries.append(
                    MarkerEntry(path=Path(filename), lineno=int(row), keyword=keyword, text=message)
                )
            except Exception:
                continue
    return entries


def summarize(entries: Iterable[MarkerEntry]) -> dict[str, int]:
    counts = {"total": 0, "high": 0, "medium": 0, "low": 0}
    for entry in entries:
        counts["total"] += 1
        level = PRIORITY.get(entry.keyword.upper(), "medium" if entry.keyword.upper().startswith("RUFF-") else "low")
        counts[level] = counts.get(level, 0) + 1
    return counts


def keyword_priority(keyword: str) -> str:
    upper = keyword.upper()
    if upper.startswith("RUFF-"):
        return "medium"
    return PRIORITY.get(upper, "low")


def filter_by_priority(entries: Sequence[MarkerEntry], allowed_levels: set[str]) -> list[MarkerEntry]:
    return [entry for entry in entries if keyword_priority(entry.keyword) in allowed_levels]


def priority_value(level: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}.get(level.lower(), 1)


def enforce_thresholds(counts: dict[str, int], *, max_total: int | None, max_high: int | None) -> None:
    violations: list[str] = []
    if max_total is not None and counts.get("total", 0) > max_total:
        violations.append(f"total {counts['total']} > {max_total}")
    if max_high is not None and counts.get("high", 0) > max_high:
        violations.append(f"high {counts['high']} > {max_high}")
    if violations:
        joined = "; ".join(violations)
        print(f"[FAIL] {_TASK} debt thresholds exceeded: {joined}")
        raise SystemExit(1)


def format_markdown(
    entries: Iterable[MarkerEntry],
    project_root: Path,
    roots: Sequence[Path],
    *,
    wrap_width: int = 120,
    title: str | None = None,
) -> str:
    entries = list(entries)
    entries.sort(key=lambda e: (e.rel_path.lower(), e.lineno))
    counts = summarize(entries)
    total = counts["total"]
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    root_labels: list[str] = []
    for root in roots:
        try:
            rel = root.resolve().relative_to(project_root)
            root_labels.append(rel.as_posix())
        except Exception:
            # Fallback to basename to avoid leaking host-specific prefixes
            root_labels.append(root.name or str(root))
    if not root_labels:
        try:
            root_labels.append(WEBAPP_DIR.resolve().relative_to(project_root).as_posix())
        except Exception:
            root_labels.append(str(WEBAPP_DIR))
    roots_text = ", ".join(root_labels)

    # Build a short exclusion preview (avoid dumping long lists)
    exclusion_samples: list[str] = []
    for ex in list(EXCLUDED_FILES)[:5]:
        try:
            exclusion_samples.append(ex.resolve().relative_to(project_root).as_posix())
        except Exception:
            exclusion_samples.append(str(ex))
    exclusions_text = ", ".join(exclusion_samples) if exclusion_samples else "(none)"

    # Group keywords by priority for a compact legend
    legend: dict[str, list[str]] = {"high": [], "medium": [], "low": []}
    for kw, level in PRIORITY.items():
        legend.setdefault(level, []).append(kw)

    marker_counts = Counter(entry.keyword for entry in entries)

    lines: list[str] = [
        "---",
        "layout: default",
        f"title: \"{title or (_TASK + '/' + _FIXME + ' Index')}\"",
        "---",
        "",
        "<!-- markdownlint-disable-file MD001 MD004 MD011 MD022 MD024 MD025 MD033 MD034 MD037 MD050 MD052 -->",
        "",
        f"Index scope: {_TASK}/{_FIXME}/HACK/XXX annotations under `{roots_text}`.",
        f"Generated: {timestamp}",
        f"Total annotations: {total}",
        f"High: {counts['high']}, Medium: {counts['medium']}, Low: {counts['low']}",
        "",
        "## Scan Profile",
        "",
        f"- Roots: {roots_text}",
        f"- Tracked markers: {', '.join(KEYWORDS)}",
        "- Priority map: " + "; ".join(f"{lvl}: {', '.join(sorted(vals)) or 'none'}" for lvl, vals in legend.items()),
        f"- Exclusions (sample): {exclusions_text}",
        f"- Regex: \\b({'|'.join(KEYWORDS)})\\b (case-insensitive)",
        "",
        "## Marker Breakdown",
        "",
    ]

    if marker_counts:
        for kw in sorted(marker_counts.keys()):
            lvl = PRIORITY.get(kw, "low")
            lines.append(f"- {kw}: {marker_counts[kw]} ({lvl})")
    else:
        lines.append("- None detected.")

    lines.extend([
        "",
        "## Root Coverage",
        "",
    ])

    root_counts = Counter()
    for entry in entries:
        rel = entry.path.relative_to(project_root) if entry.path.is_absolute() else entry.path
        root_counts[str(rel).split("/", 1)[0]] += 1

    if root_counts:
        for root_label in sorted(root_counts.keys()):
            lines.append(f"- {root_label}: {root_counts[root_label]}")
    else:
        lines.append("- None detected.")

    lines.extend([
        "",
        "## Files",
        "",
    ])

    if not entries:
        lines.append(f"No {_TASK}/{_FIXME}/HACK/XXX annotations found under specified roots.")
        lines.append("")
        return "\n".join(lines)

    grouped: dict[str, list[MarkerEntry]] = {}
    for entry in entries:
        rel = entry.path.relative_to(project_root) if entry.path.is_absolute() else entry.path
        grouped.setdefault(rel.as_posix(), []).append(entry)

    for rel_path in sorted(grouped.keys()):
        lines.append(f"### {rel_path}")
        lines.append("")
        for entry in grouped[rel_path]:
            snippet = entry.text.strip()
            prefix = f"- L{entry.lineno} *{entry.keyword}*: "
            available = max(20, wrap_width - len(prefix))
            wrapped = textwrap.wrap(snippet, width=available) or [""]
            lines.append(prefix + wrapped[0])
            for cont in wrapped[1:]:
                lines.append(f"  {cont}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_output(markdown: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Generate {_TASK}/{_FIXME} index")
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        help="Root directory to scan (default: webapp/). Can be provided multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output markdown file path (default: docs/todos.md)",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=None,
        help="Fail if total annotations exceed this number",
    )
    parser.add_argument(
        "--max-high",
        type=int,
        default=None,
        help="Fail if high-priority annotations exceed this number",
    )
    parser.add_argument(
        "--min-priority",
        choices=["high", "medium", "low", "all"],
        default="all",
        help="Filter output to entries at or above this priority (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of entries to emit after sorting (for compact views)",
    )
    parser.add_argument(
        "--wrap",
        type=int,
        default=120,
        help="Line wrap width for entry text (default: 120)",
    )
    parser.add_argument(
        "--ruff-json",
        action="append",
        type=Path,
        help="Optional Ruff JSON report(s) to include as findings",
    )
    args = parser.parse_args()

    project_root = DEFAULT_ROOT
    roots = args.root if args.root else [WEBAPP_DIR]
    scan_roots = [r.resolve() for r in roots]

    todos = gather_markers(scan_roots)
    ruff_entries = ingest_ruff_json([p.resolve() for p in args.ruff_json] if args.ruff_json else [])
    all_entries = todos + ruff_entries

    # Apply priority filter if requested
    min_priority = args.min_priority.lower()
    if min_priority != "all":
        threshold = priority_value(min_priority)
        all_entries = [
            e for e in all_entries if priority_value(keyword_priority(e.keyword)) >= threshold
        ]

    # Optional entry cap for compact, readable subsets
    if args.limit is not None and args.limit > 0:
        all_entries = sorted(all_entries, key=lambda e: (e.rel_path.lower(), e.lineno))[: args.limit]

    counts = summarize(all_entries)
    enforce_thresholds(counts, max_total=args.max_total, max_high=args.max_high)

    md = format_markdown(all_entries, project_root, scan_roots, wrap_width=args.wrap)
    write_output(md, args.output.resolve())
    print(f"Wrote {len(all_entries)} annotations to {args.output.resolve()}")

    subset_specs = [
        ("high", OUTPUT_HIGH_PATH, {"high"}),
        ("medium", OUTPUT_MEDIUM_PATH, {"medium"}),
        ("low", OUTPUT_LOW_PATH, {"low"}),
    ]

    for label, path, allowed_levels in subset_specs:
        subset_entries = filter_by_priority(all_entries, allowed_levels)
        subset_title = f"{_TASK}/{_FIXME} Index — {label.capitalize()}"
        subset_md = format_markdown(
            subset_entries,
            project_root,
            scan_roots,
            wrap_width=args.wrap,
            title=subset_title,
        )
        write_output(subset_md, path)
        print(f"Wrote {len(subset_entries)} annotations to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

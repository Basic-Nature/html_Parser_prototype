#!/usr/bin/env python3
"""Generate docs/todos.md by scanning sources and optional Ruff outputs.

Scans text-based source files for TODO-like annotations and writes a markdown
index to docs/todos.md. Optionally ingests Ruff JSON reports for lint findings.
Supports multiple roots (defaults to project webapp/).
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

KEYWORDS = ("TODO", "FIXME", "HACK", "XXX", "WARNING", "WARN", "NOTE")
PRIORITY = {
    "FIXME": "high",
    "TODO": "medium",
    "HACK": "medium",
    "XXX": "medium",
    "WARNING": "low",
    "WARN": "low",
    "NOTE": "low",
}

DEFAULT_ROOT = Path(__file__).resolve().parent.parent
WEBAPP_DIR = DEFAULT_ROOT / "webapp"
OUTPUT_PATH = DEFAULT_ROOT / "docs" / "todos.md"

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


@dataclass
class TodoEntry:
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
            yield path


def extract_todos(path: Path) -> list[TodoEntry]:
    entries: list[TodoEntry] = []
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
        text = line.strip()
        entries.append(TodoEntry(path=path, lineno=idx, keyword=keyword, text=text))
    return entries


def gather_todos(roots: Sequence[Path]) -> list[TodoEntry]:
    todos: list[TodoEntry] = []
    for file_path in iter_source_files(roots):
        todos.extend(extract_todos(file_path))
    return todos


def ingest_ruff_json(paths: Sequence[Path]) -> list[TodoEntry]:
    entries: list[TodoEntry] = []
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
                    TodoEntry(path=Path(filename), lineno=int(row), keyword=keyword, text=message)
                )
            except Exception:
                continue
    return entries


def summarize(entries: Iterable[TodoEntry]) -> dict[str, int]:
    counts = {"total": 0, "high": 0, "medium": 0, "low": 0}
    for entry in entries:
        counts["total"] += 1
        level = PRIORITY.get(entry.keyword.upper(), "medium" if entry.keyword.upper().startswith("RUFF-") else "low")
        counts[level] = counts.get(level, 0) + 1
    return counts


def enforce_thresholds(counts: dict[str, int], *, max_total: int | None, max_high: int | None) -> None:
    violations: list[str] = []
    if max_total is not None and counts.get("total", 0) > max_total:
        violations.append(f"total {counts['total']} > {max_total}")
    if max_high is not None and counts.get("high", 0) > max_high:
        violations.append(f"high {counts['high']} > {max_high}")
    if violations:
        joined = "; ".join(violations)
        print(f"[FAIL] TODO debt thresholds exceeded: {joined}")
        raise SystemExit(1)


def format_markdown(entries: Iterable[TodoEntry], project_root: Path, roots: Sequence[Path]) -> str:
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

    lines: list[str] = [
        "---",
        "layout: default",
        "title: \"TODO/FIXME Index\"",
        "---",
        "",
        f"Index scope: TODO/FIXME/HACK/XXX/WARNING/NOTE annotations under `{roots_text}`.",
        f"Generated: {timestamp}",
        f"Total annotations: {total}",
        f"High: {counts['high']}, Medium: {counts['medium']}, Low: {counts['low']}",
        "",
        "## Files",
        "",
    ]

    if not entries:
        lines.append("No TODO/FIXME/WARNING/NOTE annotations found under specified roots.")
        lines.append("")
        return "\n".join(lines)

    grouped: dict[str, list[TodoEntry]] = {}
    for entry in entries:
        rel = entry.path.relative_to(project_root) if entry.path.is_absolute() else entry.path
        grouped.setdefault(rel.as_posix(), []).append(entry)

    wrap_width = 120
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
    parser = argparse.ArgumentParser(description="Generate TODO/FIXME index")
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
        "--ruff-json",
        action="append",
        type=Path,
        help="Optional Ruff JSON report(s) to include as findings",
    )
    args = parser.parse_args()

    project_root = DEFAULT_ROOT
    roots = args.root if args.root else [WEBAPP_DIR]
    scan_roots = [r.resolve() for r in roots]

    todos = gather_todos(scan_roots)
    ruff_entries = ingest_ruff_json([p.resolve() for p in args.ruff_json] if args.ruff_json else [])
    all_entries = todos + ruff_entries

    counts = summarize(all_entries)
    enforce_thresholds(counts, max_total=args.max_total, max_high=args.max_high)

    md = format_markdown(all_entries, project_root, scan_roots)
    write_output(md, args.output.resolve())
    print(f"Wrote {len(all_entries)} annotations to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

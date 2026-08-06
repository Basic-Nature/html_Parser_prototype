#!/usr/bin/env python3
"""Clean noisy OCR table output into a usable vote summary CSV.

Usage:
  python tools/clean_noisy_pdf_ocr_output.py --input output/ocr_debug/<file>__clean.txt
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

HEADING_PATTERN = re.compile(
    r'(?P<heading>'
    r'(?:County Committee AD:\s*\d+\s+ED:\s*\d+)'
    r'|(?:Member of the Assembly \d+(?:st|nd|rd|th) Assembly District)'
    r'|(?:Member of the State Committee \d+(?:st|nd|rd|th) Assembly District)'
    r'|(?:State Senator \d+(?:st|nd|rd|th) Senatorial District)'
    r'|(?:Representative in Congress \d+(?:st|nd|rd|th) Congressional District)'
    r'|(?:Comptroller Citywide)'
    r'|(?:Judge of the Civil Court)'
    r')'
)

ROW_PATTERN = re.compile(
    r'(?P<label>WRITE-INs?|Over Votes|Under Votes|[A-Z][A-Za-z0-9&\'’\.\- ]{3,80}?)'
    r'(?:\s*[,\.]{1,3}\s*|\s{2,}|\s+)'
    r'(?P<votes>\d{1,3}(?:,\d{3})?)'
    r'(?:\s+|[.,]+\s*)'
    r'(?P<percent>\d{1,3}(?:\.\d{1,2})?)'
)

JUNK_NAME_PATTERNS = [
    re.compile(r'Group[\.\s]*', re.IGNORECASE),
    re.compile(r'\b(?:ww|ee|oo|mm|nn|ll|ss|rr|cc|tt|pp|qq|kk|xx)\b', re.IGNORECASE),
    re.compile(r'[\.\-]{2,}'),
    re.compile(r'\s{2,}'),
]

NAME_EXTRACTOR = re.compile(
    r'([A-Z][A-Za-z\'’\.\-]+(?:\s+[A-Z][A-Za-z\'’\.\-]+)+)',
)


def load_text(path: Path) -> str:
    text = path.read_text(encoding='utf-8', errors='replace')
    return ' '.join(text.split())


def split_contests(text: str) -> list[tuple[str, str]]:
    positions = [(m.start(), m.end(), m.group('heading')) for m in HEADING_PATTERN.finditer(text)]
    if not positions:
        return [('', text)]

    segments: list[tuple[str, str]] = []
    for index, (start, end, heading) in enumerate(positions):
        next_start = positions[index + 1][0] if index + 1 < len(positions) else len(text)
        body = text[end:next_start].strip()
        segments.append((heading, body))
    return segments


def clean_label(raw_label: str) -> str:
    label = raw_label.strip()
    label = re.sub(r'[\u2018\u2019]', "'", label)
    for pattern in JUNK_NAME_PATTERNS:
        label = pattern.sub(' ', label)
    label = ' '.join(label.split())

    if re.search(r'\bWRITE-?INs?\b', label, re.IGNORECASE):
        return 'WRITE-IN'
    if re.search(r'\b(?:Over|Oven|Qver) Votes\b', label, re.IGNORECASE):
        return 'Over Votes'
    if re.search(r'\bUnder Votes\b', label, re.IGNORECASE):
        return 'Under Votes'

    label = label.strip(' .,-:;')

    if not re.search(r'[A-Za-z]', label):
        return ''

    candidate_matches = NAME_EXTRACTOR.findall(label)
    if not candidate_matches:
        if len(re.sub(r'[^A-Za-z]', '', label)) < 4:
            return ''
        return label

    # choose the most plausible name candidate and ignore generic junk.
    candidates = [candidate.strip() for candidate in candidate_matches if len(candidate.strip()) > 2]
    candidates = [c for c in candidates if not re.search(r'\b(Group|District|Court|County|Municipal|Kings|Queens|Brooklyn|Bronx|Staten)\b', c, re.IGNORECASE)]
    if candidates:
        return max(candidates, key=lambda c: (len(c.split()), len(c)))
    return max(candidate_matches, key=lambda c: (len(c.split()), len(c))).strip()


def parse_segment(heading: str, body: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for match in ROW_PATTERN.finditer(body):
        raw_label = match.group('label')
        votes = match.group('votes').replace(',', '')
        percent = match.group('percent')
        label = clean_label(raw_label)

        if label.lower().startswith('group'):
            continue
        if label in {'0', '00'}:
            continue

        row_type = 'candidate'
        if label in {'WRITE-IN', 'Over Votes', 'Under Votes'}:
            row_type = label

        if label in {'Qver Votes', 'Oven Votes'}:
            label = 'Over Votes'
            row_type = 'Over Votes'

        if not label or label.lower().startswith('group'):
            continue

        if label.isdigit():
            continue

        rows.append(
            {
                'contest': heading,
                'candidate': label,
                'votes': votes,
                'percent': percent,
                'row_type': row_type,
                'raw_label': raw_label,
            }
        )
    return rows


def normalize_candidate_name(candidate: str) -> str:
    candidate = re.sub(r'[\u2018\u2019]', "'", candidate)
    candidate = re.sub(r'[^A-Za-z0-9\s\.&-]+', ' ', candidate)
    candidate = re.sub(r'\s+', ' ', candidate).strip(' .,-:;')
    return candidate


def is_short_invalid_candidate(candidate: str) -> bool:
    candidate = candidate.strip()
    if len(candidate.split()) == 1 and len(candidate) < 8:
        return True
    if '&' in candidate and len(candidate.split()) == 1:
        return True
    return False


def merge_similar_candidate_names(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_contest: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_contest.setdefault(row['contest'], []).append(row)

    merged: list[dict[str, str]] = []
    for contest, contest_rows in by_contest.items():
        label_map: dict[str, str] = {}
        normalized_labels = [normalize_candidate_name(row['candidate']) for row in contest_rows]
        for row, norm in zip(contest_rows, normalized_labels):
            row['candidate_normalized'] = norm

        for row in contest_rows:
            candidate = row['candidate_normalized']
            if '&' in candidate or is_short_invalid_candidate(candidate):
                first_token = candidate.split()[0] if candidate.split() else ''
                if first_token:
                    candidates = [r['candidate_normalized'] for r in contest_rows if r['candidate_normalized'].startswith(first_token) and r['candidate_normalized'] != candidate]
                    if candidates:
                        best_match = max(candidates, key=len)
                        label_map[candidate] = best_match
        for row in contest_rows:
            candidate = row['candidate_normalized']
            if candidate in label_map:
                row['candidate'] = label_map[candidate]
                row['candidate_normalized'] = label_map[candidate]
            else:
                row['candidate'] = candidate
                row['candidate_normalized'] = candidate
            merged.append(row)
    return merged


def parse_score(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except ValueError:
        return default


def dedupe_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row['contest'], row['row_type'], normalize_candidate_name(row['candidate']))
        grouped.setdefault(key, []).append(row)

    best_rows: list[dict[str, str]] = []
    for key, group in grouped.items():
        if len(group) == 1:
            best_rows.append(group[0])
            continue

        def score(row: dict[str, str]) -> tuple[float, int]:
            percent = parse_score(row['percent'], -1.0)
            votes = int(row['votes']) if row['votes'].isdigit() else 0
            return (percent if 0 <= percent <= 100 else -1.0, votes)

        best = max(group, key=score)
        best_rows.append(best)

    return best_rows


def write_csv(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['contest', 'candidate', 'votes', 'percent', 'row_type', 'raw_label']
    with out_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: list[dict[str, str]]) -> str:
    by_contest = Counter((row['contest'], row['row_type']) for row in rows)
    lines = [f'Total rows: {len(rows)}', f'Total contests: {len({row["contest"] for row in rows})}']
    for (contest, row_type), count in sorted(by_contest.items()):
        lines.append(f'  {contest} [{row_type}]: {count}')
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description='Clean noisy OCR election results into a CSV.')
    parser.add_argument('--input', '-i', type=Path, required=True, help='Path to the OCR clean text file')
    parser.add_argument('--output', '-o', type=Path, help='Output CSV path')
    parser.add_argument('--show', action='store_true', help='Print a summary and sample rows')
    args = parser.parse_args()

    text = load_text(args.input)
    segments = split_contests(text)
    rows: list[dict[str, str]] = []
    for heading, body in segments:
        rows.extend(parse_segment(heading, body))

    rows = merge_similar_candidate_names(rows)
    rows = [
        {k: v for k, v in row.items() if k != 'candidate_normalized'}
        for row in rows
    ]
    rows = [row for row in rows if row['row_type'] != 'candidate' or not is_short_invalid_candidate(row['candidate'])]
    rows = dedupe_rows(rows)
    if args.output is None:
        args.output = args.input.with_name(args.input.stem + '_cleaned.csv')

    write_csv(rows, args.output)

    if args.show:
        print(summarize(rows))
        for row in rows[:20]:
            print(row)

    print(f'Wrote cleaned CSV: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

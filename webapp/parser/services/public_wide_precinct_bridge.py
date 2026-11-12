"""REVIEW CANDIDATE ONLY: source-pinned, schema-driven, long/wide-independent CSV bridge.

No I/O/network/database, no registry mutation, and no automatic public dispatch.
Profile is server-reviewed data, not user-selected input. Labels stay source-native;
canonical vote-method totals are not inferred without an approved crosswalk.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass
from typing import Any, Mapping

MAX_SOURCE_BYTES = 25 * 1024 * 1024
MAX_SOURCE_ROWS = 150_000
MAX_INPUT_COLUMNS = 100
MAX_HEADERS = 500
MAX_CANDIDATES = 200
MAX_CELL_CHARS = 2_000
MAX_OUTPUT_ROWS = 30_000


class WideBridgeError(ValueError):
    pass


@dataclass(frozen=True)
class WideBridgeResult:
    headers: list[str]
    rows: list[dict[str, Any]]
    audit: dict[str, Any]


def _safe_label(s: str) -> bool:
    return bool(s) and len(s) <= 160 and s[0] not in '=+-@' and all(ord(ch) >= 32 for ch in s)


def _votes(s: str) -> int | None:
    text = str(s).strip().replace(',', '')
    if not text:
        return None
    if not re.fullmatch(r'[0-9]+', text):
        raise WideBridgeError('INVALID_VOTE_CELL')
    value = int(text)
    if value > 2**63 - 1:
        raise WideBridgeError('VOTE_CELL_OVERFLOW')
    return value


def _must(condition: bool, code: str) -> None:
    if not condition:
        raise WideBridgeError(code)


def _profile(profile: Mapping[str, Any], source: bytes) -> dict[str, Any]:
    required = {'contract', 'source_sha256', 'source_url', 'scope', 'expected_headers',
                'fields', 'vote_methods', 'against_methods', 'selected_contest_value',
                'precinct_pattern', 'expected_source_rows'}
    _must(isinstance(profile, dict) and set(profile) == required, 'PROFILE_EXACT_KEYS_REQUIRED')
    _must(profile['contract'] == 'electionpulse_reviewed_wide_csv_schema_v1', 'PROFILE_CONTRACT_INVALID')
    _must(isinstance(source, bytes) and 0 < len(source) <= MAX_SOURCE_BYTES, 'SOURCE_BYTE_BUDGET')
    sha = hashlib.sha256(source).hexdigest()
    _must(sha == profile['source_sha256'], 'SOURCE_DIGEST_PROFILE_MISMATCH')
    _must(isinstance(profile['source_url'], str) and profile['source_url'].startswith('https://'), 'PROFILE_SOURCE_URL_INVALID')
    _must(isinstance(profile['scope'], dict) and set(profile['scope']) == {'state', 'year', 'contest'}, 'PROFILE_SCOPE_INVALID')
    for val in profile['scope'].values():
        _must(isinstance(val, str) and _safe_label(val), 'PROFILE_SCOPE_VALUE_INVALID')
    fields = profile['fields']
    _must(isinstance(fields, dict) and set(fields) == {'locality_code', 'locality_label', 'precinct', 'contest', 'candidate', 'party', 'writein_flag'}, 'PROFILE_FIELDS_INVALID')
    required_cols = profile['expected_headers']
    _must(isinstance(required_cols, list) and 1 <= len(required_cols) <= MAX_INPUT_COLUMNS and len(set(required_cols)) == len(required_cols), 'PROFILE_HEADERS_INVALID')
    _must(all(isinstance(h, str) and 0 < len(h) <= 160 for h in required_cols), 'PROFILE_HEADER_LABEL_INVALID')
    _must(isinstance(profile['vote_methods'], list) and 1 <= len(profile['vote_methods']) <= 16, 'PROFILE_METHOD_COUNT')
    _must(isinstance(profile['against_methods'], list) and len(profile['against_methods']) == len(profile['vote_methods']), 'PROFILE_AGAINST_METHOD_COUNT')
    indices = list(fields.values()) + list(profile['vote_methods']) + list(profile['against_methods'])
    _must(len(indices) == len(set(indices)) and set(indices).issubset(set(required_cols)), 'PROFILE_SOURCE_FIELDS_COLLIDE_OR_MISSING')
    _must(all(isinstance(i, str) and _safe_label(i.strip()) for i in indices), 'PROFILE_SOURCE_FIELD_UNSAFE')
    _must(isinstance(profile['selected_contest_value'], str) and _safe_label(profile['selected_contest_value']), 'PROFILE_CONTEST_INVALID')
    _must(isinstance(profile['precinct_pattern'], str) and len(profile['precinct_pattern']) <= 120, 'PROFILE_PRECINCT_REGEX_INVALID')
    _must(type(profile['expected_source_rows']) is int and 1 <= profile['expected_source_rows'] <= MAX_SOURCE_ROWS, 'PROFILE_ROW_COUNT_INVALID')
    return dict(profile)


def parse_reviewed_wide_csv(source: bytes, profile: Mapping[str, Any]) -> WideBridgeResult:
    """Return source-native, source-verified precinct rows; no canonical method guessing."""
    p = _profile(profile, source)
    try:
        decoded = source.decode('utf-8-sig', errors='strict')
    except UnicodeDecodeError as exc:
        raise WideBridgeError('SOURCE_ENCODING_NOT_UTF8') from exc
    if any((ord(c) < 32 and c not in '\n\r\t') or 127 <= ord(c) <= 159 for c in decoded):
        raise WideBridgeError('SOURCE_CONTROL_CHARACTER')
    if '\x00' in decoded[:4096]:
        raise WideBridgeError('SOURCE_BINARY_CONTENT')
    fld = p['fields']
    methods = p['vote_methods']
    against = p['against_methods']
    byprec: dict[tuple[str, str], dict[tuple[str, str, str], tuple[int | None, ...]]] = {}
    candidates: set[tuple[str, str, str]] = set()
    county_names: dict[str, str] = {}
    seen = set()
    source_rows = 0
    selected_rows = 0
    missing_cells = 0
    against_positive = 0
    method_stats = {m: {'zero': 0, 'positive': 0, 'missing': 0} for m in methods}
    try:
        reader = csv.DictReader(io.StringIO(decoded, newline=''), restkey='__extra__', restval=None)
        _must(reader.fieldnames == p['expected_headers'], 'SOURCE_HEADERS_NOT_EXACT_PROFILE')
        for raw in reader:
            source_rows += 1
            _must(source_rows <= MAX_SOURCE_ROWS, 'SOURCE_ROWS_OVER_BUDGET')
            _must(isinstance(raw, dict) and '__extra__' not in raw and all(isinstance(v, str) and len(v) <= MAX_CELL_CHARS for v in raw.values()), 'SOURCE_RAGGED_OR_OVERSIZED_ROW')
            # Validate source jurisdiction identities for every row, even other contests.
            code = raw[fld['locality_code']].strip()
            name = raw[fld['locality_label']].strip()
            precinct = raw[fld['precinct']].strip()
            _must(all(_safe_label(v) for v in (code, name, precinct)), 'SOURCE_IDENTITY_LABEL_UNSAFE')
            _must(re.fullmatch(p['precinct_pattern'], precinct) is not None, 'PRECINCT_PATTERN_REJECTED')
            old = county_names.setdefault(code, name)
            _must(old == name, 'LOCALITY_CODE_NAME_CONFLICT')
            if raw[fld['contest']].strip() != p['selected_contest_value']:
                continue
            selected_rows += 1
            c = (raw[fld['candidate']].strip(), raw[fld['party']].strip(), raw[fld['writein_flag']].strip())
            _must(_safe_label(c[0]) and (not c[1] or _safe_label(c[1])) and c[2] in ('Y', ''), 'CANDIDATE_PARTY_OR_WRITEIN_INVALID')
            k = (code, precinct)
            _must((k, c) not in seen, 'DUPLICATE_CANDIDATE_OBSERVATION')
            seen.add((k, c))
            candidates.add(c)
            _must(len(candidates) <= MAX_CANDIDATES, 'CANDIDATE_BUDGET_EXCEEDED')
            vals = tuple(_votes(raw[m]) for m in methods)
            missing_cells += sum(v is None for v in vals)
            for m, v in zip(methods, vals):
                kind = 'missing' if v is None else 'zero' if v == 0 else 'positive'
                method_stats[m][kind] += 1
            for against_column in against:
                vote = _votes(raw[against_column])
                if vote is not None and vote > 0:
                    against_positive += 1
            byprec.setdefault(k, {})[c] = vals
    except csv.Error as exc:
        raise WideBridgeError('CSV_ENGINE_REJECTED') from exc
    _must(source_rows == p['expected_source_rows'], 'SOURCE_ROW_COUNT_PROFILE_DRIFT')
    _must(bool(byprec) and len(byprec) <= MAX_OUTPUT_ROWS, 'NO_IN_SCOPE_PRECINCTS_OR_TOO_MANY')
    _must(against_positive == 0, 'IN_SCOPE_AGAINST_VOTES_REQUIRE_SEPARATE_CONTEST_MODEL')
    def label(c: tuple[str, str, str]) -> str:
        return c[0] + (f' ({c[1]})' if c[1] else '') + (' [Write-In:Y]' if c[2] == 'Y' else '')
    ordered = sorted(candidates, key=lambda c: (c[1].casefold(), c[0].casefold(), c[2]))
    labels = [label(c) for c in ordered]
    _must(len(set(labels)) == len(labels), 'CANDIDATE_LABEL_COLLISION')
    base = ['Precinct', '% Precincts Reporting', 'Election Day Total', 'Early Voting Total',
            'Absentee Mail Total', 'Provisional Total', 'Official Source Link', 'Source SHA-256',
            'Derived Source Method Total', 'Discrepancy Flag', 'Grand Total']
    native = [m + ' Total (source-native)' for m in methods]
    cand_methods = [n + ' - ' + m for n in labels for m in methods]
    cand_totals = [n + ' - Total Votes (derived)' for n in labels]
    headers = base + native + cand_methods + cand_totals
    _must(len(headers) <= MAX_HEADERS and len(set(headers)) == len(headers), 'OUTPUT_HEADER_BUDGET_OR_DUPLICATE')
    rows = []
    derived_sum = 0
    missing_candidate_occurrences = 0
    for key, groups in sorted(byprec.items()):
        row = {h: None for h in headers}
        row['Precinct'] = key[0] + ' / ' + key[1]
        row['Official Source Link'] = p['source_url']
        row['Source SHA-256'] = p['source_sha256']
        totals = []
        for cand in ordered:
            vals = groups.get(cand)
            if vals is None:
                missing_candidate_occurrences += 1
            for m, i in zip(methods, range(len(methods))):
                row[label(cand) + ' - ' + m] = None if vals is None else vals[i]
            if vals is not None and all(v is not None for v in vals):
                total = sum(vals)
                _must(total <= 2**63 - 1, 'CANDIDATE_TOTAL_OVERFLOW')
                row[label(cand) + ' - Total Votes (derived)'] = total
                totals.append(total)
        for i, m in enumerate(methods):
            vals = [groups[c][i] if c in groups else None for c in ordered]
            if all(v is not None for v in vals):
                row[m + ' Total (source-native)'] = sum(vals)
        if len(totals) == len(ordered):
            row['Grand Total'] = sum(totals)
            row['Derived Source Method Total'] = row['Grand Total']
            derived_sum += row['Grand Total']
        row['Discrepancy Flag'] = 'OFFICIAL_PRECINCT_TOTAL_NOT_PROVIDED; SOURCE_METHOD_CROSSWALK_PENDING'
        rows.append(row)
    grand = {h: None for h in headers}
    grand['Precinct'] = 'Grand Total'
    grand['Official Source Link'] = p['source_url']
    grand['Source SHA-256'] = p['source_sha256']
    grand['Discrepancy Flag'] = 'Summary only; not an individual precinct'
    for h in headers:
        if h in ('Precinct', 'Discrepancy Flag', '% Precincts Reporting', 'Official Source Link', 'Source SHA-256'):
            continue
        vals = [row[h] for row in rows]
        if vals and all(type(v) is int for v in vals):
            grand[h] = sum(vals)
    rows.append(grand)
    audit = {'source_rows': source_rows, 'selected_observations': selected_rows,
             'precinct_rows': len(rows)-1, 'candidate_count': len(candidates),
             'header_count': len(headers), 'derived_candidate_vote_sum': derived_sum,
             'missing_source_method_cells': missing_cells,
             'missing_candidate_occurrences': missing_candidate_occurrences,
             'against_positive': against_positive, 'source_native_method_stats': method_stats,
             'source_native_methods': list(methods),
             'canonical_method_crosswalk': 'UNAUTHORIZED_UNAVAILABLE',
             'official_precinct_total_reconciliation': 'UNAVAILABLE',
             'context_dynamic_inference': 'NOT_CALLED_BOUNDED_PROFILE_ONLY'}
    return WideBridgeResult(headers=headers, rows=rows, audit=audit)

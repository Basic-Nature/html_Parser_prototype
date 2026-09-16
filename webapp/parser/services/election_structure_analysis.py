"""Pure Smart Elections structural analysis for ``TablePipelineResult``.

This service never mutates headers/rows, fills missing values, publishes data,
or crosses the canonical boundary. It reuses the existing ballot-pattern,
ballot-group, header-normalization, and numeric-parsing authorities.
"""
from __future__ import annotations

from collections import OrderedDict
import re
from typing import Any, Mapping

from webapp.parser.Context_Integration.Context_Library.constants import (
    BALLOT_TYPES,
    BALLOT_TYPES_SORT_ORDER,
    CANDIDATE_BALLOT_SPLIT_PATTERN,
    TOTAL_KEYWORDS,
    canonical_ballot_group,
)
from webapp.parser.contracts.election_structure import (
    CandidateMethodBlock,
    ElectionStructureAnalysis,
    ElectionStructureIssue,
    ElectionStructureIssueCode,
    ElectionStructureSeverity,
)
from webapp.parser.contracts.table_pipeline import TablePipelineResult
from webapp.parser.utils.detect import normalize_header, parse_numeric

_CAND_RE = re.compile(CANDIDATE_BALLOT_SPLIT_PATTERN, re.UNICODE)
_PARTY = {normalize_header("Party"), normalize_header("Ballot Party")}
_PERCENT = {
    normalize_header("% Vote"),
    normalize_header("Percent Vote"),
    normalize_header("Vote Share"),
    normalize_header("Cumulative %"),
    normalize_header("Cumulative Percent"),
    normalize_header("Cumulative Vote"),
}
_TOTAL = {normalize_header(x) for x in TOTAL_KEYWORDS if isinstance(x, str)}
_TOTAL.update(
    normalize_header(x)
    for x in ("Total", "Total Vote", "Total Votes", "Total Reported", "Vote Total")
)
_KNOWN_METHODS = {
    normalize_header(x)
    for x in tuple(BALLOT_TYPES) + tuple(BALLOT_TYPES_SORT_ORDER)
    if isinstance(x, str) and x.strip()
}
_PRECINCT_HEADERS = (
    "Precinct",
    "Division Name",
    "Election District",
    "District",
    "Ward",
    "Municipality",
)
_REPORTING_HEADERS = (
    "% Precincts Reporting",
    "Percent Reported",
    "% Reported",
    "Precincts Reporting",
)
_GRAND_HEADERS = ("Grand Total", "Total Votes", "Overall Total")
_MISSING = {"", "na", "n/a", "n.a.", "null", "none", "pending", "tbd", "-", "--", "—", "–"}


def _display(value: Any) -> Any:
    return value if value is None or isinstance(value, (str, int, float, bool)) else f"<{type(value).__name__}>"


def _missing(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip().lower() in _MISSING
    )


def _numeric(value: Any) -> tuple[int | None, str]:
    if _missing(value):
        return None, "missing"
    parsed, is_percent = parse_numeric(value)
    if parsed is None or is_percent:
        return None, "invalid"
    return parsed, "numeric"


def _find_header(headers: tuple[str, ...], priority: tuple[str, ...]) -> str | None:
    by_norm = {normalize_header(h): h for h in headers}
    for label in priority:
        match = by_norm.get(normalize_header(label))
        if match is not None:
            return match
    return None


def _canonical_method(label: str) -> str:
    label = str(label or "").strip()
    if not label:
        return ""
    try:
        canonical = canonical_ballot_group(label)
    except Exception:
        canonical = label
    return str(canonical or label).strip()


def _split(header: str) -> tuple[str, str] | None:
    if " - " not in header:
        return None
    candidate, suffix = header.rsplit(" - ", 1)
    candidate, suffix = candidate.strip(), suffix.strip()
    return (candidate, suffix) if candidate and suffix else None


def _role(suffix: str) -> str:
    norm = normalize_header(suffix)
    if norm in _PARTY:
        return "party"
    if norm in _PERCENT:
        return "percent"
    if norm in _TOTAL:
        return "total"
    return "method"


def _seed_candidates(headers: tuple[str, ...]) -> tuple[str, ...]:
    ordered, seen = [], set()
    for header in headers:
        match = _CAND_RE.match(header)
        if match:
            candidate = str(match.group("cand") or "").strip()
            if candidate and candidate not in seen:
                ordered.append(candidate)
                seen.add(candidate)
            continue
        split = _split(header)
        if split is None:
            continue
        candidate, suffix = split
        if _role(suffix) != "method" or normalize_header(suffix) in _KNOWN_METHODS:
            if candidate not in seen:
                ordered.append(candidate)
                seen.add(candidate)
    return tuple(ordered)


def _build_blocks(headers: tuple[str, ...]) -> tuple[tuple[CandidateMethodBlock, ...], list[ElectionStructureIssue]]:
    work = OrderedDict(
        (
            candidate,
            {
                "methods": OrderedDict(),
                "total": None,
                "party": None,
                "percent": None,
            },
        )
        for candidate in _seed_candidates(headers)
    )
    issues: list[ElectionStructureIssue] = []
    for header in headers:
        split = _split(header)
        if split is None:
            continue
        candidate, suffix = split
        if candidate not in work:
            continue
        role = _role(suffix)
        if role in {"total", "party", "percent"}:
            work[candidate][role] = work[candidate][role] or header
            continue
        method = _canonical_method(suffix)
        if not method:
            continue
        methods = work[candidate]["methods"]
        if method in methods:
            issues.append(
                ElectionStructureIssue(
                    code=ElectionStructureIssueCode.DUPLICATE_METHOD_COLUMN,
                    severity=ElectionStructureSeverity.ERROR,
                    message=f"Duplicate method column for {candidate!r}/{method!r}.",
                    candidate=candidate,
                    method=method,
                    header=header,
                    details={"first_header": methods[method]},
                )
            )
        else:
            methods[method] = header
    blocks = tuple(
        CandidateMethodBlock(
            candidate=candidate,
            method_headers=tuple(entry["methods"].items()),
            total_header=entry["total"],
            party_header=entry["party"],
            percent_header=entry["percent"],
        )
        for candidate, entry in work.items()
    )
    return blocks, issues


def _vote_methods(blocks: tuple[CandidateMethodBlock, ...]) -> tuple[str, ...]:
    out, seen = [], set()
    for block in blocks:
        for method in block.methods:
            if method not in seen:
                out.append(method)
                seen.add(method)
    return tuple(out)


def _method_totals(headers: tuple[str, ...], methods: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    by_norm = {normalize_header(method): method for method in methods}
    grand_norms = {normalize_header(x) for x in _GRAND_HEADERS}
    out, seen = [], set()
    for header in headers:
        if " - " in header or normalize_header(header) in grand_norms:
            continue
        match = re.match(r"^(?P<method>.+?)\s+total$", str(header).strip(), re.I)
        if not match:
            continue
        canonical = _canonical_method(match.group("method"))
        method = by_norm.get(normalize_header(canonical))
        if method is None and normalize_header(canonical) in _KNOWN_METHODS:
            method = canonical
        if method and method not in seen:
            out.append((method, header))
            seen.add(method)
    return tuple(out)


def _issue(
    issues: list[ElectionStructureIssue],
    code: ElectionStructureIssueCode,
    message: str,
    *,
    severity: ElectionStructureSeverity = ElectionStructureSeverity.WARNING,
    row_index: int | None = None,
    precinct: str | None = None,
    candidate: str | None = None,
    method: str | None = None,
    header: str | None = None,
    observed: Any = None,
    expected: Any = None,
    details: Mapping[str, Any] | None = None,
) -> None:
    issues.append(
        ElectionStructureIssue(
            code=code,
            message=message,
            severity=severity,
            row_index=row_index,
            precinct=precinct,
            candidate=candidate,
            method=method,
            header=header,
            observed=_display(observed),
            expected=_display(expected),
            details=details or {},
        )
    )


def analyze_election_structure(result: TablePipelineResult) -> ElectionStructureAnalysis:
    if not isinstance(result, TablePipelineResult):
        raise TypeError("result must be a TablePipelineResult")

    headers, rows = tuple(result.headers), tuple(result.rows)
    precinct_header = _find_header(headers, _PRECINCT_HEADERS)
    reporting_header = _find_header(headers, _REPORTING_HEADERS)
    grand_header = _find_header(headers, _GRAND_HEADERS)
    blocks, issues = _build_blocks(headers)
    methods = _vote_methods(blocks)
    method_total_headers = _method_totals(headers, methods)
    method_total_map = dict(method_total_headers)
    candidates = tuple(block.candidate for block in blocks)

    if precinct_header is None:
        _issue(
            issues,
            ElectionStructureIssueCode.MISSING_PRECINCT_HEADER,
            "No recognized precinct/division header was found.",
            severity=ElectionStructureSeverity.ERROR,
        )

    matrix_complete = bool(blocks and methods)
    for block in blocks:
        if block.total_header is None:
            matrix_complete = False
            _issue(
                issues,
                ElectionStructureIssueCode.MISSING_CANDIDATE_TOTAL_COLUMN,
                f"Candidate {block.candidate!r} has no total column.",
                severity=ElectionStructureSeverity.ERROR,
                candidate=block.candidate,
            )
        for method in methods:
            if block.header_for_method(method) is None:
                matrix_complete = False
                _issue(
                    issues,
                    ElectionStructureIssueCode.MISSING_METHOD_COLUMN,
                    f"Candidate {block.candidate!r} is missing method {method!r}.",
                    severity=ElectionStructureSeverity.ERROR,
                    candidate=block.candidate,
                    method=method,
                )

    for method in methods:
        if method not in method_total_map:
            _issue(
                issues,
                ElectionStructureIssueCode.MISSING_METHOD_TOTAL_COLUMN,
                f"No row-level method total exists for {method!r}.",
                method=method,
            )

    schema_complete = bool(
        precinct_header
        and grand_header
        and blocks
        and methods
        and matrix_complete
        and all(method in method_total_map for method in methods)
    )

    seen_precincts: dict[str, int] = {}
    duplicate_precincts, duplicate_seen = [], set()
    zero_count = null_count = 0
    cand_assessable = method_assessable = precinct_assessable = bool(rows)
    if grand_header is None:
        precinct_assessable = False
    cand_checks = cand_mismatches = 0
    method_checks = method_mismatches = 0
    precinct_checks = precinct_mismatches = 0

    for row_index, row in enumerate(rows):
        precinct = None
        if precinct_header is not None:
            raw_precinct = row.get(precinct_header)
            if _missing(raw_precinct):
                _issue(
                    issues,
                    ElectionStructureIssueCode.MISSING_PRECINCT_VALUE,
                    "Precinct/division identity is missing.",
                    severity=ElectionStructureSeverity.ERROR,
                    row_index=row_index,
                    header=precinct_header,
                    observed=raw_precinct,
                )
            else:
                precinct = str(raw_precinct).strip()
                if precinct in seen_precincts:
                    if precinct not in duplicate_seen:
                        duplicate_precincts.append(precinct)
                        duplicate_seen.add(precinct)
                    _issue(
                        issues,
                        ElectionStructureIssueCode.DUPLICATE_PRECINCT,
                        f"Duplicate precinct/division {precinct!r}.",
                        severity=ElectionStructureSeverity.ERROR,
                        row_index=row_index,
                        precinct=precinct,
                        header=precinct_header,
                        details={"first_row_index": seen_precincts[precinct]},
                    )
                else:
                    seen_precincts[precinct] = row_index

        values: dict[tuple[str, str], int | None] = {}
        candidate_expected: dict[str, int | None] = {}

        for block in blocks:
            complete, subtotal = True, 0
            for method in methods:
                header = block.header_for_method(method)
                if header is None:
                    complete = False
                    values[(block.candidate, method)] = None
                    continue
                raw = row.get(header)
                numeric, state = _numeric(raw)
                values[(block.candidate, method)] = numeric
                if state == "missing":
                    null_count += 1
                    complete = False
                    _issue(
                        issues,
                        ElectionStructureIssueCode.MISSING_VOTE_VALUE,
                        f"Missing vote value for {block.candidate!r}/{method!r}.",
                        row_index=row_index,
                        precinct=precinct,
                        candidate=block.candidate,
                        method=method,
                        header=header,
                        observed=raw,
                    )
                    continue
                if state == "invalid":
                    complete = False
                    _issue(
                        issues,
                        ElectionStructureIssueCode.NON_NUMERIC_VOTE_VALUE,
                        f"Non-numeric vote value for {block.candidate!r}/{method!r}.",
                        severity=ElectionStructureSeverity.ERROR,
                        row_index=row_index,
                        precinct=precinct,
                        candidate=block.candidate,
                        method=method,
                        header=header,
                        observed=raw,
                    )
                    continue
                assert numeric is not None
                if numeric == 0:
                    zero_count += 1
                if numeric < 0:
                    _issue(
                        issues,
                        ElectionStructureIssueCode.NEGATIVE_VOTE_VALUE,
                        f"Negative signed vote evidence preserved for {block.candidate!r}/{method!r}.",
                        severity=ElectionStructureSeverity.ERROR,
                        row_index=row_index,
                        precinct=precinct,
                        candidate=block.candidate,
                        method=method,
                        header=header,
                        observed=numeric,
                    )
                subtotal += numeric

            candidate_expected[block.candidate] = subtotal if complete else None
            if block.total_header is None or not complete:
                cand_assessable = False
                continue
            raw_total = row.get(block.total_header)
            observed_total, state = _numeric(raw_total)
            if state == "missing":
                cand_assessable = False
                _issue(
                    issues,
                    ElectionStructureIssueCode.CANDIDATE_TOTAL_MISSING,
                    f"Candidate total is missing for {block.candidate!r}.",
                    row_index=row_index,
                    precinct=precinct,
                    candidate=block.candidate,
                    header=block.total_header,
                    observed=raw_total,
                    expected=subtotal,
                )
            elif state == "invalid":
                cand_assessable = False
                _issue(
                    issues,
                    ElectionStructureIssueCode.CANDIDATE_TOTAL_NON_NUMERIC,
                    f"Candidate total is non-numeric for {block.candidate!r}.",
                    severity=ElectionStructureSeverity.ERROR,
                    row_index=row_index,
                    precinct=precinct,
                    candidate=block.candidate,
                    header=block.total_header,
                    observed=raw_total,
                    expected=subtotal,
                )
            else:
                cand_checks += 1
                if observed_total != subtotal:
                    cand_mismatches += 1
                    _issue(
                        issues,
                        ElectionStructureIssueCode.CANDIDATE_TOTAL_MISMATCH,
                        f"Candidate total mismatch for {block.candidate!r}.",
                        severity=ElectionStructureSeverity.ERROR,
                        row_index=row_index,
                        precinct=precinct,
                        candidate=block.candidate,
                        header=block.total_header,
                        observed=observed_total,
                        expected=subtotal,
                    )

        for method in methods:
            total_header = method_total_map.get(method)
            if total_header is None:
                method_assessable = False
                continue
            method_values = [values.get((b.candidate, method)) for b in blocks]
            if not method_values or any(value is None for value in method_values):
                method_assessable = False
                continue
            expected_total = sum(int(value) for value in method_values if value is not None)
            raw_total = row.get(total_header)
            observed_total, state = _numeric(raw_total)
            if state == "missing":
                method_assessable = False
                _issue(
                    issues,
                    ElectionStructureIssueCode.METHOD_TOTAL_MISSING,
                    f"Method total is missing for {method!r}.",
                    row_index=row_index,
                    precinct=precinct,
                    method=method,
                    header=total_header,
                    observed=raw_total,
                    expected=expected_total,
                )
            elif state == "invalid":
                method_assessable = False
                _issue(
                    issues,
                    ElectionStructureIssueCode.METHOD_TOTAL_NON_NUMERIC,
                    f"Method total is non-numeric for {method!r}.",
                    severity=ElectionStructureSeverity.ERROR,
                    row_index=row_index,
                    precinct=precinct,
                    method=method,
                    header=total_header,
                    observed=raw_total,
                    expected=expected_total,
                )
            else:
                method_checks += 1
                if observed_total != expected_total:
                    method_mismatches += 1
                    _issue(
                        issues,
                        ElectionStructureIssueCode.METHOD_TOTAL_MISMATCH,
                        f"Method total mismatch for {method!r}.",
                        severity=ElectionStructureSeverity.ERROR,
                        row_index=row_index,
                        precinct=precinct,
                        method=method,
                        header=total_header,
                        observed=observed_total,
                        expected=expected_total,
                    )

        expected_candidates = list(candidate_expected.values())
        expected_precinct = (
            sum(int(v) for v in expected_candidates if v is not None)
            if blocks and expected_candidates and all(v is not None for v in expected_candidates)
            else None
        )
        if grand_header is None or expected_precinct is None:
            precinct_assessable = False
        else:
            raw_grand = row.get(grand_header)
            observed_grand, state = _numeric(raw_grand)
            if state == "missing":
                precinct_assessable = False
                _issue(
                    issues,
                    ElectionStructureIssueCode.GRAND_TOTAL_MISSING,
                    "Grand Total is missing.",
                    row_index=row_index,
                    precinct=precinct,
                    header=grand_header,
                    observed=raw_grand,
                    expected=expected_precinct,
                )
            elif state == "invalid":
                precinct_assessable = False
                _issue(
                    issues,
                    ElectionStructureIssueCode.GRAND_TOTAL_NON_NUMERIC,
                    "Grand Total is non-numeric.",
                    severity=ElectionStructureSeverity.ERROR,
                    row_index=row_index,
                    precinct=precinct,
                    header=grand_header,
                    observed=raw_grand,
                    expected=expected_precinct,
                )
            else:
                precinct_checks += 1
                if observed_grand != expected_precinct:
                    precinct_mismatches += 1
                    _issue(
                        issues,
                        ElectionStructureIssueCode.PRECINCT_TOTAL_MISMATCH,
                        "Grand Total does not equal candidate vote-method sums.",
                        severity=ElectionStructureSeverity.ERROR,
                        row_index=row_index,
                        precinct=precinct,
                        header=grand_header,
                        observed=observed_grand,
                        expected=expected_precinct,
                    )

    return ElectionStructureAnalysis(
        source_stage=result.stage,
        row_count=len(rows),
        precinct_header=precinct_header,
        reporting_header=reporting_header,
        grand_total_header=grand_header,
        method_total_headers=method_total_headers,
        candidates=candidates,
        vote_methods=methods,
        candidate_blocks=blocks,
        duplicate_precincts=tuple(duplicate_precincts),
        issues=tuple(issues),
        candidate_method_matrix_complete=matrix_complete,
        schema_complete=schema_complete,
        candidate_totals_reconciled=(
            None if not cand_assessable or cand_checks == 0 else cand_mismatches == 0
        ),
        method_totals_reconciled=(
            None
            if not method_assessable or method_checks == 0
            else method_mismatches == 0
        ),
        precinct_totals_reconciled=(
            None
            if not precinct_assessable or precinct_checks == 0
            else precinct_mismatches == 0
        ),
        observed_zero_count=zero_count,
        observed_null_count=null_count,
    )

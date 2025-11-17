from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ..Context_Integration.Context_Library.constants import (
    DEFAULT_TOTAL_RESULT_DISPLAY,
    PARTY_CODE_MAP,
    normalize_party_label,
    normalize_result_group_label,
)
from ..Context_Integration.librarian import clean_for_json
from .contest_normalization import normalize_contest_label

__all__ = [
    "NormalizedResultRow",
    "ContestCoverage",
    "NormalizedExport",
    "load_state_export",
    "load_json_export",
]

# Use candidate id when available; otherwise fall back to normalized name key.
_ALL_COUNTIES_LABEL = "ALL_COUNTIES"
_KNOWN_PARTY_HINTS = {code.upper() for code in PARTY_CODE_MAP.keys()}
_INCUMBENT_TOKEN_RE = re.compile(r"\(\s*I\s*\)", re.I)
_PARTY_SUFFIX_RE = re.compile(r"(?P<body>.*?)(?:\(|\s)(?P<token>[A-Z]{1,4})\)?\s*$")
_EXTRA_SPACE_RE = re.compile(r"\s{2,}")
_DISTRICT_LABEL_RE = re.compile(r"(district\s+\d+[A-Za-z-]*)", re.I)
def _safe_int(value: object) -> int:
    try:
        if value is None:
            return 0
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value).strip()
        if not text:
            return 0
        return int(float(text))
    except (TypeError, ValueError):
        return 0


def _collapse_spaces(label: str) -> str:
    return _EXTRA_SPACE_RE.sub(" ", label).strip()


def _strip_party_from_name(raw_name: str | None) -> Tuple[str, Optional[str]]:
    if not raw_name:
        return "", None
    text = _INCUMBENT_TOKEN_RE.sub(" ", raw_name)
    text = _collapse_spaces(text)
    match = _PARTY_SUFFIX_RE.search(text)
    party_hint: Optional[str] = None
    base = text
    if match:
        candidate_token = match.group("token").upper()
        if candidate_token in _KNOWN_PARTY_HINTS:
            party_hint = candidate_token
            base = match.group("body")
    base = base.rstrip(" -/,(")
    base = base.strip('\"')
    base = _collapse_spaces(base)
    return base or text, party_hint


def _normalize_candidate(option: dict) -> Tuple[Optional[str], str, str, str]:
    candidate_id = option.get("id")
    candidate_id_str = str(candidate_id) if candidate_id is not None else None
    raw_name = option.get("name") or ""
    name_clean, party_hint = _strip_party_from_name(raw_name)
    explicit_party = option.get("politicalParty")
    resolved_party = explicit_party or party_hint
    party_full = normalize_party_label(resolved_party)
    if not name_clean:
        name_clean = _collapse_spaces(raw_name)
    return candidate_id_str, name_clean, party_full, raw_name


@dataclass(frozen=True)
class NormalizedResultRow:
    contest_id: str
    contest_name: str
    county: str
    candidate_id: Optional[str]
    candidate_name: str
    candidate_raw: str
    party_full: str
    group_name: str
    group_display: str
    vote_count: int


@dataclass
class ContestCoverage:
    contest_id: str
    contest_name: str
    vote_for: Optional[int]
    contest_type: Optional[str]
    contest_name_raw: Optional[str] = None
    contest_question: Optional[str] = None
    counties_total: int = 0
    counties_reporting: int = 0
    total_precincts_participating: int = 0
    total_precincts_reporting: int = 0
    county_details: Dict[str, Dict[str, int]] = field(default_factory=dict)
    counties_in_scope: List[str] = field(default_factory=list)
    total_counties_available: int = 0
    division_scope: str = "unknown"
    division_identifier: Optional[str] = None


@dataclass
class NormalizedExport:
    election_date: Optional[str]
    election_name: Optional[str]
    county_rows: List[NormalizedResultRow]
    statewide_rows: List[NormalizedResultRow]
    coverage: Dict[str, ContestCoverage]
    statewide_reference: Dict[str, dict]
    total_counties: int = 0
    candidate_label_map: Dict[str, str] = field(default_factory=dict)
    context_snapshot: Dict[str, object] = field(default_factory=dict)


def _iter_county_contests(payload: dict) -> Iterable[Tuple[str, dict]]:
    for county in payload.get("localResults", []) or []:
        county_name = county.get("name") or ""
        for contest in county.get("ballotItems", []) or []:
            yield county_name, contest


def _normalize_group_labels(raw: object) -> Tuple[str, str]:
    return normalize_result_group_label(raw)


def _derive_division_metadata(entry: ContestCoverage, statewide_contest: Optional[dict]) -> Tuple[str, Optional[str]]:
    name = entry.contest_name or ""
    if not name and statewide_contest:
        name = statewide_contest.get("name") or ""
    name_low = name.lower()

    total_counties = entry.total_counties_available or 0
    counties_total = entry.counties_total

    if total_counties and counties_total == total_counties:
        return "statewide", "Statewide"

    if counties_total == 1 and entry.counties_in_scope:
        return "single-county", entry.counties_in_scope[0]

    district_label = None
    if statewide_contest:
        district_label = statewide_contest.get("districtName") or statewide_contest.get("district")
    if not district_label:
        district_match = _DISTRICT_LABEL_RE.search(name)
        if district_match:
            district_label = district_match.group(1).title()

    if district_label:
        return "district", district_label

    if "district" in name_low:
        return "district", None
    if "county" in name_low:
        return "county", None
    if counties_total > 1:
        return "multi-county", None

    return "unknown", None


def _build_context_snapshot(
    payload: dict,
    county_rows: List[NormalizedResultRow],
    statewide_rows: List[NormalizedResultRow],
    coverage: Dict[str, ContestCoverage],
) -> Dict[str, object]:
    ballot_groups = sorted({row.group_display for row in county_rows + statewide_rows})
    party_labels = sorted({row.party_full for row in county_rows + statewide_rows if row.party_full})
    county_names = sorted({row.county for row in county_rows if row.county != _ALL_COUNTIES_LABEL})
    contest_scope = {
        contest_id: {
            "division_scope": entry.division_scope,
            "division_identifier": entry.division_identifier,
            "counties_total": entry.counties_total,
            "counties_reporting": entry.counties_reporting,
        }
        for contest_id, entry in coverage.items()
    }
    contest_questions = {
        contest_id: {
            "label": entry.contest_name,
            "question": entry.contest_question,
            "raw": entry.contest_name_raw,
        }
        for contest_id, entry in coverage.items()
        if entry.contest_question or entry.contest_name_raw
    }
    snapshot = {
        "election_date": payload.get("electionDate"),
        "election_name": payload.get("electionName"),
        "ballot_groups": ballot_groups,
        "party_labels": party_labels,
        "county_names": county_names,
        "contest_scope": contest_scope,
        "contest_questions": contest_questions,
        "contest_count": len(coverage),
        "total_counties": len(payload.get("localResults") or []),
    }
    try:
        return clean_for_json(snapshot)
    except Exception:
        return snapshot


def load_state_export(path: str | Path) -> NormalizedExport:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    statewide_contests = (payload.get("results") or {}).get("ballotItems", []) or []
    statewide_reference = {str(contest.get("id")): contest for contest in statewide_contests}

    county_rows: List[NormalizedResultRow] = []
    coverage: Dict[str, ContestCoverage] = {}
    aggregate_totals: Dict[str, Dict[str, Dict[str, Dict[str, Optional[str] | int]]]] = defaultdict(lambda: defaultdict(dict))
    aggregate_meta: Dict[Tuple[str, str], Dict[str, Optional[str]]] = {}
    candidate_label_map: Dict[str, str] = {}

    for county_name, contest in _iter_county_contests(payload):
        contest_id = str(contest.get("id"))
        contest_name_raw = contest.get("name") or ""
        contest_short = contest.get("shortTitle") or ""
        normalized_name, referendum_question, raw_label = normalize_contest_label(
            contest_name_raw,
            short_title=contest_short,
        )

        coverage_entry = coverage.setdefault(
            contest_id,
            ContestCoverage(
                contest_id=contest_id,
                contest_name=normalized_name,
                contest_name_raw=raw_label or contest_name_raw or None,
                contest_question=referendum_question,
                vote_for=contest.get("voteFor"),
                contest_type=contest.get("contestType"),
            ),
        )
        if not coverage_entry.contest_name:
            coverage_entry.contest_name = normalized_name
        if not coverage_entry.contest_name_raw:
            coverage_entry.contest_name_raw = contest_name_raw or None
        if not coverage_entry.contest_question and referendum_question:
            coverage_entry.contest_question = referendum_question

        coverage_entry.counties_total += 1
        if county_name not in coverage_entry.counties_in_scope:
            coverage_entry.counties_in_scope.append(county_name)
        precincts_participating = _safe_int(contest.get("precinctsParticipating"))
        precincts_reporting = _safe_int(contest.get("precinctsReporting"))
        coverage_entry.total_precincts_participating += precincts_participating
        coverage_entry.total_precincts_reporting += precincts_reporting
        if precincts_reporting:
            coverage_entry.counties_reporting += 1
        coverage_entry.county_details[county_name] = {
            "precincts_participating": precincts_participating,
            "precincts_reporting": precincts_reporting,
        }

        for option in contest.get("ballotOptions", []) or []:
            candidate_id, candidate_name, party_full, candidate_raw = _normalize_candidate(option)
            candidate_key = candidate_id or candidate_name.lower()
            total_votes = _safe_int(option.get("voteCount"))
            total_key, total_display = _normalize_group_labels("total")
            county_rows.append(
                NormalizedResultRow(
                    contest_id=contest_id,
                    contest_name=normalized_name,
                    county=county_name,
                    candidate_id=candidate_id,
                    candidate_name=candidate_name,
                    candidate_raw=candidate_raw,
                    party_full=party_full,
                    group_name=total_key,
                    group_display=total_display,
                    vote_count=total_votes,
                )
            )
            ct_map = aggregate_totals[contest_id][candidate_key].setdefault(
                total_key,
                {"votes": 0, "display": total_display},
            )
            ct_map["votes"] = int(ct_map["votes"] or 0) + total_votes
            aggregate_meta.setdefault(
                (contest_id, candidate_key),
                {
                    "contest_name": normalized_name,
                    "candidate_name": candidate_name,
                    "party_full": party_full,
                    "candidate_id": candidate_id,
                    "candidate_raw": candidate_raw,
                },
            )
            candidate_label_map.setdefault(candidate_raw, candidate_name)

            for group in option.get("groupResults", []) or []:
                if group.get("voteCount") is None:
                    continue
                group_votes = _safe_int(group.get("voteCount"))
                group_key, group_display = _normalize_group_labels(group.get("groupName"))
                county_rows.append(
                    NormalizedResultRow(
                        contest_id=contest_id,
                        contest_name=normalized_name,
                        county=county_name,
                        candidate_id=candidate_id,
                        candidate_name=candidate_name,
                        candidate_raw=candidate_raw,
                        party_full=party_full,
                        group_name=group_key,
                        group_display=group_display,
                        vote_count=group_votes,
                    )
                )
                g_map = aggregate_totals[contest_id][candidate_key].setdefault(
                    group_key,
                    {"votes": 0, "display": group_display},
                )
                g_map["votes"] = int(g_map.get("votes", 0) or 0) + group_votes

    statewide_rows: List[NormalizedResultRow] = []
    for contest_id, candidates in aggregate_totals.items():
        for candidate_key, groups in candidates.items():
            meta = aggregate_meta[(contest_id, candidate_key)]
            for group_name, details in groups.items():
                votes = int(details.get("votes", 0) or 0)
                display = details.get("display") or DEFAULT_TOTAL_RESULT_DISPLAY
                statewide_rows.append(
                    NormalizedResultRow(
                        contest_id=contest_id,
                        contest_name=meta["contest_name"] or "",
                        county=_ALL_COUNTIES_LABEL,
                        candidate_id=meta["candidate_id"],
                        candidate_name=meta["candidate_name"] or "",
                        candidate_raw=meta.get("candidate_raw") or meta["candidate_name"] or "",
                        party_full=meta["party_full"] or "Other",
                        group_name=group_name,
                        group_display=display or group_name.replace("_", " ").title(),
                        vote_count=votes,
                    )
                )

    total_counties_available = len(payload.get("localResults") or [])
    for contest_id, entry in coverage.items():
        entry.counties_in_scope.sort()
        entry.total_counties_available = total_counties_available
        statewide_contest = statewide_reference.get(contest_id)
        if statewide_contest:
            statewide_raw = statewide_contest.get("name") or ""
            statewide_short = statewide_contest.get("shortTitle") or ""
            normalized_title, statewide_question, statewide_raw_label = normalize_contest_label(
                statewide_raw,
                short_title=statewide_short,
            )
            if normalized_title and normalized_title != entry.contest_name:
                entry.contest_name = normalized_title
            if not entry.contest_name_raw and statewide_raw_label:
                entry.contest_name_raw = statewide_raw_label
            elif not entry.contest_name_raw and statewide_raw:
                entry.contest_name_raw = statewide_raw
            if not entry.contest_question:
                if statewide_question:
                    entry.contest_question = statewide_question
                elif statewide_raw and statewide_raw != entry.contest_name:
                    entry.contest_question = statewide_raw
        scope, label = _derive_division_metadata(entry, statewide_contest)
        entry.division_scope = scope
        entry.division_identifier = label

    county_rows.sort(key=lambda r: (r.contest_id, r.county, r.candidate_name, r.group_name))
    statewide_rows.sort(key=lambda r: (r.contest_id, r.candidate_name, r.group_name))

    context_snapshot = _build_context_snapshot(payload, county_rows, statewide_rows, coverage)

    return NormalizedExport(
        election_date=payload.get("electionDate"),
        election_name=payload.get("electionName"),
        county_rows=county_rows,
        statewide_rows=statewide_rows,
        coverage=coverage,
        statewide_reference=statewide_reference,
        total_counties=total_counties_available,
        candidate_label_map=candidate_label_map,
        context_snapshot=context_snapshot,
    )


def load_json_export(path: str | Path) -> NormalizedExport:
    """Alias for load_state_export for clearer intent in handlers."""
    return load_state_export(path)

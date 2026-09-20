"""Deterministic Workflow semantic artifact materialization for Smart Elections rows.

This service converts the finalized Smart Elections wide-row boundary into the
existing W4 normalized semantic comparison contract. It has no HTTP, identity,
database, or canonical publication authority.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from webapp.parser.contracts.workflow_comparison import (
    WORKFLOW_COMPARISON_PAYLOAD_CONTRACT,
    WORKFLOW_COMPARISON_PAYLOAD_SCHEMA_VERSION,
    WORKFLOW_COMPARISON_VERSION,
    canonical_percent_string,
    normalize_text,
    normalize_vote_method,
    ordered_vote_methods,
    semantic_sha256,
    validate_comparison_payload,
    validate_normalized_semantic,
)

WORKFLOW_NORMALIZED_ARTIFACT_CONTRACT = "workflow_normalized_semantic_artifact_v1"
WORKFLOW_PARSER_OBSERVATION_MANIFEST_CONTRACT = "workflow_parser_observation_manifest_v1"
_TOTAL_SUFFIXES = frozenset({"total", "total vote", "total votes", "total reported"})
_NULL_TEXT = frozenset({"", "na", "n/a", "null", "none"})
_PARTY_SUFFIX_RE = re.compile(r"^(?P<name>.+?)\s+\((?P<party>[^()]*)\)\s*$")


class WorkflowNormalizedArtifactError(ValueError):
    pass


def _normalized_nullable(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return normalize_text(text)


def workflow_scope_from_item(item: object) -> dict[str, object]:
    election_date = getattr(item, "election_date", None)
    if isinstance(election_date, date):
        election_date_value = election_date.isoformat()
    elif election_date is None:
        election_date_value = None
    else:
        election_date_value = str(election_date).strip() or None
    return {
        "election_year": getattr(item, "election_year", None),
        "election_date": election_date_value,
        "state": _normalized_nullable(getattr(item, "state", None)),
        "jurisdiction_name": _normalized_nullable(
            getattr(item, "jurisdiction_name", None)
        ),
        "jurisdiction_type": _normalized_nullable(
            getattr(item, "jurisdiction_type", None)
        ),
        "contest": _normalized_nullable(getattr(item, "contest", None)),
    }


def _candidate_identity(label: str) -> tuple[str, str | None]:
    normalized = normalize_text(label)
    assert isinstance(normalized, str)
    match = _PARTY_SUFFIX_RE.fullmatch(normalized)
    if match is None:
        return normalized, None
    name = normalize_text(match.group("name"))
    party_raw = match.group("party").strip()
    party = normalize_text(party_raw) if party_raw else None
    assert isinstance(name, str)
    return name, party


def _vote_state(row: Mapping[str, object], key: str) -> dict[str, object]:
    if key not in row:
        return {"state": "missing", "votes": None}
    raw = row.get(key)
    if raw is None:
        return {"state": "null", "votes": None}
    if isinstance(raw, bool):
        raise WorkflowNormalizedArtifactError(f"{key} must not be boolean")
    if isinstance(raw, int):
        if raw < 0:
            raise WorkflowNormalizedArtifactError(f"{key} must be nonnegative")
        return {"state": "value", "votes": raw}
    if isinstance(raw, float):
        if not raw.is_integer() or raw < 0:
            raise WorkflowNormalizedArtifactError(
                f"{key} must be a nonnegative integer value"
            )
        return {"state": "value", "votes": int(raw)}
    text = str(raw).strip()
    if text.casefold() in _NULL_TEXT:
        return {"state": "null", "votes": None}
    compact = text.replace(",", "")
    if not compact.isdigit():
        raise WorkflowNormalizedArtifactError(
            f"{key} is not an integer/null vote value"
        )
    return {"state": "value", "votes": int(compact)}


def _percent_state(
    row: Mapping[str, object],
    keys: Sequence[str] = ("% Precincts Reporting", "Percent Reported"),
) -> dict[str, object]:
    key = next((candidate for candidate in keys if candidate in row), None)
    if key is None:
        return {"state": "missing", "value": None}
    raw = row.get(key)
    if raw is None:
        return {"state": "null", "value": None}
    if isinstance(raw, bool):
        raise WorkflowNormalizedArtifactError("percent reporting must not be boolean")
    text = str(raw).strip()
    if text.casefold() in _NULL_TEXT:
        return {"state": "null", "value": None}
    if text.endswith("%"):
        text = text[:-1].strip()
    canonical = canonical_percent_string(text)
    return {"state": "value", "value": canonical}


def _column_model(headers: Sequence[str]) -> dict[str, object]:
    clean_headers = [str(header).strip() for header in headers]
    candidate_columns: dict[str, dict[str, str]] = {}
    candidate_total_columns: dict[str, str] = {}
    method_total_columns: dict[str, str] = {}
    raw_methods: list[str] = []

    for header in clean_headers:
        if not header or header in {"Precinct", "% Precincts Reporting", "Percent Reported", "Grand Total"}:
            continue
        if " - " in header:
            candidate_label, suffix = header.rsplit(" - ", 1)
            candidate_label = candidate_label.strip()
            suffix = suffix.strip()
            if not candidate_label or not suffix:
                continue
            if suffix.casefold() in _TOTAL_SUFFIXES:
                candidate_total_columns[candidate_label] = header
                candidate_columns.setdefault(candidate_label, {})
                continue
            canonical_method = normalize_vote_method(suffix)
            existing = candidate_columns.setdefault(candidate_label, {})
            if canonical_method in existing and existing[canonical_method] != header:
                raise WorkflowNormalizedArtifactError(
                    f"duplicate candidate/method column for {candidate_label!r} / {canonical_method!r}"
                )
            existing[canonical_method] = header
            raw_methods.append(canonical_method)
            continue

        if header.endswith(" Total"):
            method_label = header[:-6].strip()
            if method_label and method_label.casefold() != "grand":
                canonical_method = normalize_vote_method(method_label)
                if (
                    canonical_method in method_total_columns
                    and method_total_columns[canonical_method] != header
                ):
                    raise WorkflowNormalizedArtifactError(
                        f"duplicate method total column for {canonical_method!r}"
                    )
                method_total_columns[canonical_method] = header
                raw_methods.append(canonical_method)

    if not candidate_columns:
        raise WorkflowNormalizedArtifactError(
            "Smart Elections rows contain no candidate columns"
        )
    methods = ordered_vote_methods(tuple(dict.fromkeys(raw_methods)))
    if not methods:
        raise WorkflowNormalizedArtifactError(
            "Smart Elections rows contain no vote methods"
        )

    candidates: list[dict[str, object]] = []
    for label in candidate_columns:
        name, party = _candidate_identity(label)
        candidates.append({"label": label, "name": name, "party": party})
    candidates.sort(
        key=lambda item: (
            item["party"] is None,
            "" if item["party"] is None else str(item["party"]),
            str(item["name"]),
        )
    )
    return {
        "methods": methods,
        "candidate_columns": candidate_columns,
        "candidate_total_columns": candidate_total_columns,
        "method_total_columns": method_total_columns,
        "candidates": candidates,
    }


def build_workflow_semantic(
    headers: Sequence[str],
    rows: Sequence[Mapping[str, object]],
    *,
    scope: Mapping[str, object],
) -> dict[str, object]:
    if not rows:
        raise WorkflowNormalizedArtifactError(
            "Workflow normalized artifact requires at least one row"
        )
    model = _column_model(headers)
    methods = tuple(model["methods"])
    candidate_columns = model["candidate_columns"]
    candidate_total_columns = model["candidate_total_columns"]
    method_total_columns = model["method_total_columns"]
    candidate_defs = model["candidates"]

    records: list[dict[str, object]] = []
    seen_units: set[tuple[str | None, str | None]] = set()
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping):
            raise WorkflowNormalizedArtifactError(
                f"row[{index}] must be an object"
            )
        row = dict(raw_row)
        precinct_raw = row.get("Precinct")
        precinct = _normalized_nullable(precinct_raw)
        if precinct is None:
            raise WorkflowNormalizedArtifactError(
                f"row[{index}] requires Precinct"
            )
        unit_key = ("precinct", precinct)
        if unit_key in seen_units:
            raise WorkflowNormalizedArtifactError(
                f"duplicate Precinct semantic key: {precinct!r}"
            )
        seen_units.add(unit_key)

        method_totals = []
        for method in methods:
            header = method_total_columns.get(method)
            state = (
                _vote_state(row, header)
                if isinstance(header, str)
                else {"state": "missing", "votes": None}
            )
            method_totals.append({"method": method, **state})

        candidates = []
        for candidate in candidate_defs:
            label = str(candidate["label"])
            method_map = candidate_columns[label]
            method_votes = []
            for method in methods:
                header = method_map.get(method)
                state = (
                    _vote_state(row, header)
                    if isinstance(header, str)
                    else {"state": "missing", "votes": None}
                )
                method_votes.append({"method": method, **state})
            total_header = candidate_total_columns.get(label)
            total_state = (
                _vote_state(row, total_header)
                if isinstance(total_header, str)
                else {"state": "missing", "votes": None}
            )
            candidates.append(
                {
                    "name": candidate["name"],
                    "party": candidate["party"],
                    "method_votes": method_votes,
                    "total_votes": total_state,
                }
            )

        grand_total = _vote_state(row, "Grand Total")
        records.append(
            {
                "reporting_unit": {"name": precinct, "type": "precinct"},
                "percent_reporting": _percent_state(row),
                "vote_methods": list(methods),
                "method_totals": method_totals,
                "candidates": candidates,
                "grand_total": grand_total,
            }
        )

    records.sort(
        key=lambda record: (
            record["reporting_unit"]["type"] is None,
            "" if record["reporting_unit"]["type"] is None else str(record["reporting_unit"]["type"]),
            record["reporting_unit"]["name"] is None,
            "" if record["reporting_unit"]["name"] is None else str(record["reporting_unit"]["name"]),
        )
    )
    semantic = {"scope": dict(scope), "records": records}
    return validate_normalized_semantic(semantic)


def normalized_artifact_bytes(semantic: Mapping[str, object]) -> bytes:
    normalized = validate_normalized_semantic(semantic)
    payload = {
        "contract": WORKFLOW_NORMALIZED_ARTIFACT_CONTRACT,
        "semantic": normalized,
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def parser_observation_manifest_bytes(
    observations: Sequence[Mapping[str, object]],
) -> bytes:
    if not observations:
        raise WorkflowNormalizedArtifactError(
            "Workflow completion requires same-run parser observations"
        )
    for index, payload in enumerate(observations):
        if not isinstance(payload, Mapping):
            raise WorkflowNormalizedArtifactError(
                f"parser observation [{index}] must be an object"
            )
        if payload.get("contract") != "parser_observation_bundle_v1":
            raise WorkflowNormalizedArtifactError(
                "unexpected parser observation contract"
            )
        authority = payload.get("authority")
        if (
            not isinstance(authority, Mapping)
            or authority.get("canonical") is not False
            or payload.get("raw_rows_included") is not False
            or payload.get("raw_headers_included") is not False
            or payload.get("automatic_timestamp") is not False
        ):
            raise WorkflowNormalizedArtifactError(
                "parser observation violates W22 noncanonical evidence boundary"
            )
    payload = {
        "contract": WORKFLOW_PARSER_OBSERVATION_MANIFEST_CONTRACT,
        "observations": [dict(observation) for observation in observations],
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def write_exact_artifact(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing != data:
            raise WorkflowNormalizedArtifactError(
                f"refusing to overwrite different workflow artifact: {path}"
            )
        return
    path.write_bytes(data)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_comparison_payload(
    *,
    semantic: Mapping[str, object],
    binding: Mapping[str, object],
) -> dict[str, object]:
    normalized = validate_normalized_semantic(semantic)
    payload = {
        "schema": WORKFLOW_COMPARISON_PAYLOAD_CONTRACT,
        "schema_version": WORKFLOW_COMPARISON_PAYLOAD_SCHEMA_VERSION,
        "comparison_version": WORKFLOW_COMPARISON_VERSION,
        "binding": dict(binding),
        "semantic": normalized,
        "semantic_sha256": semantic_sha256(normalized),
    }
    return validate_comparison_payload(payload)

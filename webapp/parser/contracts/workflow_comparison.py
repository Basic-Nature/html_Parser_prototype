"""Pure normalized semantic Workflow comparison payload contract.

This module is dependency-free with respect to Flask, SQLAlchemy, identity
providers, and ElectionPulse runtime services. It freezes the accepted W4A
comparison payload vocabulary, deterministic normalization, canonical semantic
hashing, and fail-closed payload validation.

It does NOT read or write Workflow rows, execute the comparison service,
register routes, access the database, activate Keycloak, or call the canonical
writer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
import unicodedata
import uuid


WORKFLOW_COMPARISON_PAYLOAD_CONTRACT = (
    "workflow_normalized_semantic_comparison_payload_v1"
)
WORKFLOW_COMPARISON_PAYLOAD_SCHEMA_VERSION = 1
WORKFLOW_COMPARISON_VERSION = 1
WORKFLOW_COMPARISON_DIFFERENCE_SUMMARY_SCHEMA = (
    "workflow_comparison_difference_summary_v1"
)

VALUE_STATES = ("value", "null", "missing")
KNOWN_VOTE_METHOD_ORDER = (
    "Election Day",
    "Early Voting",
    "Absentee Mail",
    "Provisional",
)
DISCREPANCY_CATEGORIES = (
    "scope_mismatch",
    "missing_left",
    "missing_right",
    "null_vs_zero",
    "null_vs_value",
    "value_mismatch",
)

CANONICAL_JSON_RULE = (
    "UTF8_JSON_SORTED_OBJECT_KEYS_COMPACT_SEPARATORS_ENSURE_ASCII_FALSE_"
    "ALLOW_NAN_FALSE_NO_TRAILING_NEWLINE_SEMANTIC_CONTENT_ONLY"
)
NAME_NORMALIZATION_RULE = (
    "UNICODE_NFC_TRIM_AND_COLLAPSE_WHITESPACE_PRESERVE_CASE_PUNCTUATION_"
    "NO_FUZZY_ALIASING"
)
VOTE_METHOD_NORMALIZATION_RULE = (
    "CONTROLLED_CENTRAL_ALIAS_MAP_TO_CANONICAL_LABELS_"
    "KNOWN_METHOD_ORDER_THEN_UNKNOWN_METHODS_LEXICOGRAPHIC_"
    "UNKNOWN_METHODS_PRESERVED_NOT_DROPPED"
)
PERCENT_REPORTING_RULE = (
    "EXPLICIT_VALUE_STATE_AND_CANONICAL_BASE10_DECIMAL_STRING_WITHOUT_PERCENT_SIGN_"
    "NO_FLOAT_SERIALIZATION"
)
VOTE_VALUE_RULE = (
    "EXPLICIT_VALUE_STATE_VALUE_REQUIRES_NONNEGATIVE_INTEGER_"
    "NULL_AND_MISSING_REQUIRE_JSON_NULL_ZERO_IS_NUMERIC_VALUE_ZERO"
)

TOP_LEVEL_KEYS = frozenset({
    "schema",
    "schema_version",
    "comparison_version",
    "binding",
    "semantic",
    "semantic_sha256",
})
BINDING_KEYS = frozenset({
    "workflow_item_id",
    "workflow_pass_id",
    "pass_number",
    "revision_number",
    "source_evidence_ref",
    "staging_batch_id",
    "normalized_artifact_ref",
    "normalized_artifact_sha256",
})
SEMANTIC_KEYS = frozenset({"scope", "records"})
SEMANTIC_SCOPE_KEYS = frozenset({
    "election_year",
    "election_date",
    "state",
    "jurisdiction_name",
    "jurisdiction_type",
    "contest",
})
RECORD_KEYS = frozenset({
    "reporting_unit",
    "percent_reporting",
    "vote_methods",
    "method_totals",
    "candidates",
    "grand_total",
})
REPORTING_UNIT_KEYS = frozenset({"name", "type"})
VOTE_STATE_KEYS = frozenset({"state", "votes"})
PERCENT_STATE_KEYS = frozenset({"state", "value"})
METHOD_VALUE_KEYS = frozenset({"method", "state", "votes"})
CANDIDATE_KEYS = frozenset({
    "name",
    "party",
    "method_votes",
    "total_votes",
})

# W4A accepted a controlled central alias map. V1 intentionally begins with
# only canonical-label casefold aliases; broader semantic aliases require an
# explicit contract-version review rather than silent fuzzy equivalence.
VOTE_METHOD_ALIASES = {
    "election day": "Election Day",
    "early voting": "Early Voting",
    "absentee mail": "Absentee Mail",
    "provisional": "Provisional",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class WorkflowComparisonContractError(ValueError):
    """Fail-closed normalized comparison contract violation."""


def _require_exact_keys(
    name: str,
    value: Mapping[str, object],
    expected: frozenset[str],
) -> None:
    actual = frozenset(str(key) for key in value.keys())
    if actual != expected:
        raise WorkflowComparisonContractError(
            f"{name} keys must be exactly {sorted(expected)!r}; "
            f"observed {sorted(actual)!r}"
        )


def _require_mapping(name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise WorkflowComparisonContractError(f"{name} must be an object")
    return value


def _require_list(name: str, value: object) -> list[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        raise WorkflowComparisonContractError(f"{name} must be a list")
    return value


def _require_uuid(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkflowComparisonContractError(f"{name} must be a UUID string")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise WorkflowComparisonContractError(
            f"{name} must be a valid UUID"
        ) from exc
    return str(parsed)


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise WorkflowComparisonContractError(
            f"{name} must be a positive integer"
        )
    return value


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise WorkflowComparisonContractError(
            f"{name} must be lowercase 64-character SHA-256 hex"
        )
    return value


def normalize_text(value: object, *, allow_null: bool = False) -> str | None:
    if value is None:
        if allow_null:
            return None
        raise WorkflowComparisonContractError("text value must not be null")
    if not isinstance(value, str):
        raise WorkflowComparisonContractError("text value must be a string")
    normalized = unicodedata.normalize("NFC", value)
    normalized = " ".join(normalized.split())
    if not normalized:
        raise WorkflowComparisonContractError("text value must not be empty")
    return normalized


def normalize_vote_method(value: object) -> str:
    normalized = normalize_text(value)
    assert isinstance(normalized, str)
    alias = VOTE_METHOD_ALIASES.get(normalized.casefold())
    return alias if alias is not None else normalized


def ordered_vote_methods(methods: Sequence[object]) -> tuple[str, ...]:
    normalized = tuple(normalize_vote_method(method) for method in methods)
    if len(set(normalized)) != len(normalized):
        raise WorkflowComparisonContractError(
            "duplicate normalized vote method keys are invalid"
        )

    known = [
        method
        for method in KNOWN_VOTE_METHOD_ORDER
        if method in normalized
    ]
    unknown = sorted(
        method
        for method in normalized
        if method not in KNOWN_VOTE_METHOD_ORDER
    )
    return tuple(known + unknown)


def canonical_percent_string(value: object) -> str:
    if isinstance(value, bool) or isinstance(value, float):
        raise WorkflowComparisonContractError(
            "percent reporting must not use bool or float serialization"
        )

    if isinstance(value, Decimal):
        decimal_value = value
    elif isinstance(value, int):
        decimal_value = Decimal(value)
    elif isinstance(value, str):
        raw = value.strip()
        if not raw or "%" in raw:
            raise WorkflowComparisonContractError(
                "percent reporting string must be base-10 without percent sign"
            )
        try:
            decimal_value = Decimal(raw)
        except InvalidOperation as exc:
            raise WorkflowComparisonContractError(
                "percent reporting string is not a decimal"
            ) from exc
    else:
        raise WorkflowComparisonContractError(
            "percent reporting must be int, Decimal, or decimal string"
        )

    if not decimal_value.is_finite():
        raise WorkflowComparisonContractError(
            "percent reporting must be finite"
        )
    if decimal_value < 0 or decimal_value > 100:
        raise WorkflowComparisonContractError(
            "percent reporting must be between 0 and 100"
        )

    text = format(decimal_value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"-0", ""}:
        text = "0"
    return text


def _validate_vote_state(name: str, raw: object) -> dict[str, object]:
    value = _require_mapping(name, raw)
    _require_exact_keys(name, value, VOTE_STATE_KEYS)
    state = value["state"]
    votes = value["votes"]

    if state not in VALUE_STATES:
        raise WorkflowComparisonContractError(
            f"{name}.state must be one of {VALUE_STATES!r}"
        )
    if state == "value":
        if (
            isinstance(votes, bool)
            or not isinstance(votes, int)
            or votes < 0
        ):
            raise WorkflowComparisonContractError(
                f"{name}.votes must be a nonnegative integer for value state"
            )
    elif votes is not None:
        raise WorkflowComparisonContractError(
            f"{name}.votes must be null for {state} state"
        )

    return {"state": state, "votes": votes}


def _validate_percent_state(name: str, raw: object) -> dict[str, object]:
    value = _require_mapping(name, raw)
    _require_exact_keys(name, value, PERCENT_STATE_KEYS)
    state = value["state"]
    percent = value["value"]

    if state not in VALUE_STATES:
        raise WorkflowComparisonContractError(
            f"{name}.state must be one of {VALUE_STATES!r}"
        )
    if state == "value":
        canonical = canonical_percent_string(percent)
        if percent != canonical:
            raise WorkflowComparisonContractError(
                f"{name}.value must already be canonical decimal string "
                f"{canonical!r}"
            )
    elif percent is not None:
        raise WorkflowComparisonContractError(
            f"{name}.value must be null for {state} state"
        )

    return {"state": state, "value": percent}


def _nullable_normalized_text(name: str, value: object) -> str | None:
    if value is None:
        return None
    normalized = normalize_text(value)
    assert isinstance(normalized, str)
    if value != normalized:
        raise WorkflowComparisonContractError(
            f"{name} must already be normalized"
        )
    return normalized


def _validate_scope(raw: object) -> dict[str, object]:
    scope = _require_mapping("semantic.scope", raw)
    _require_exact_keys("semantic.scope", scope, SEMANTIC_SCOPE_KEYS)

    year = scope["election_year"]
    if year is not None and (
        isinstance(year, bool) or not isinstance(year, int)
    ):
        raise WorkflowComparisonContractError(
            "semantic.scope.election_year must be integer or null"
        )

    election_date = scope["election_date"]
    if election_date is not None:
        if not isinstance(election_date, str):
            raise WorkflowComparisonContractError(
                "semantic.scope.election_date must be YYYY-MM-DD or null"
            )
        try:
            datetime.strptime(election_date, "%Y-%m-%d")
        except ValueError as exc:
            raise WorkflowComparisonContractError(
                "semantic.scope.election_date must be YYYY-MM-DD or null"
            ) from exc

    normalized = dict(scope)
    for key in (
        "state",
        "jurisdiction_name",
        "jurisdiction_type",
        "contest",
    ):
        normalized[key] = _nullable_normalized_text(
            f"semantic.scope.{key}",
            scope[key],
        )
    return normalized


def _record_sort_key(record: Mapping[str, object]) -> tuple[object, ...]:
    unit = _require_mapping(
        "record.reporting_unit",
        record["reporting_unit"],
    )
    unit_type = unit["type"]
    unit_name = unit["name"]
    return (
        unit_type is None,
        "" if unit_type is None else str(unit_type),
        unit_name is None,
        "" if unit_name is None else str(unit_name),
    )


def _candidate_sort_key(candidate: Mapping[str, object]) -> tuple[object, ...]:
    party = candidate["party"]
    name = candidate["name"]
    return (
        party is None,
        "" if party is None else str(party),
        str(name),
    )


def _validate_method_values(
    *,
    name: str,
    raw: object,
    expected_methods: tuple[str, ...],
) -> list[dict[str, object]]:
    values = _require_list(name, raw)
    if len(values) != len(expected_methods):
        raise WorkflowComparisonContractError(
            f"{name} must contain exactly one entry per vote method"
        )

    normalized: list[dict[str, object]] = []
    observed_methods: list[str] = []
    for index, item_raw in enumerate(values):
        item = _require_mapping(f"{name}[{index}]", item_raw)
        _require_exact_keys(
            f"{name}[{index}]",
            item,
            METHOD_VALUE_KEYS,
        )
        method = normalize_vote_method(item["method"])
        if item["method"] != method:
            raise WorkflowComparisonContractError(
                f"{name}[{index}].method must already be canonical"
            )
        observed_methods.append(method)
        state = _validate_vote_state(
            f"{name}[{index}]",
            {"state": item["state"], "votes": item["votes"]},
        )
        normalized.append({
            "method": method,
            "state": state["state"],
            "votes": state["votes"],
        })

    if tuple(observed_methods) != expected_methods:
        raise WorkflowComparisonContractError(
            f"{name} methods must exactly match canonical vote method order"
        )
    return normalized


def _validate_candidate(
    raw: object,
    *,
    expected_methods: tuple[str, ...],
    index: int,
) -> dict[str, object]:
    candidate = _require_mapping(f"candidate[{index}]", raw)
    _require_exact_keys(f"candidate[{index}]", candidate, CANDIDATE_KEYS)

    name = normalize_text(candidate["name"])
    if candidate["name"] != name:
        raise WorkflowComparisonContractError(
            f"candidate[{index}].name must already be normalized"
        )

    party = _nullable_normalized_text(
        f"candidate[{index}].party",
        candidate["party"],
    )

    method_votes = _validate_method_values(
        name=f"candidate[{index}].method_votes",
        raw=candidate["method_votes"],
        expected_methods=expected_methods,
    )
    total_votes = _validate_vote_state(
        f"candidate[{index}].total_votes",
        candidate["total_votes"],
    )

    return {
        "name": name,
        "party": party,
        "method_votes": method_votes,
        "total_votes": total_votes,
    }


def _validate_record(raw: object, index: int) -> dict[str, object]:
    record = _require_mapping(f"record[{index}]", raw)
    _require_exact_keys(f"record[{index}]", record, RECORD_KEYS)

    unit = _require_mapping(
        f"record[{index}].reporting_unit",
        record["reporting_unit"],
    )
    _require_exact_keys(
        f"record[{index}].reporting_unit",
        unit,
        REPORTING_UNIT_KEYS,
    )
    unit_name = _nullable_normalized_text(
        f"record[{index}].reporting_unit.name",
        unit["name"],
    )
    unit_type = _nullable_normalized_text(
        f"record[{index}].reporting_unit.type",
        unit["type"],
    )

    raw_methods = _require_list(
        f"record[{index}].vote_methods",
        record["vote_methods"],
    )
    methods = ordered_vote_methods(raw_methods)
    if tuple(raw_methods) != methods:
        raise WorkflowComparisonContractError(
            f"record[{index}].vote_methods must already use canonical order"
        )

    percent = _validate_percent_state(
        f"record[{index}].percent_reporting",
        record["percent_reporting"],
    )
    method_totals = _validate_method_values(
        name=f"record[{index}].method_totals",
        raw=record["method_totals"],
        expected_methods=methods,
    )

    candidates_raw = _require_list(
        f"record[{index}].candidates",
        record["candidates"],
    )
    candidates = [
        _validate_candidate(
            item,
            expected_methods=methods,
            index=candidate_index,
        )
        for candidate_index, item in enumerate(candidates_raw)
    ]

    candidate_keys = [
        (candidate["name"], candidate["party"])
        for candidate in candidates
    ]
    if len(set(candidate_keys)) != len(candidate_keys):
        raise WorkflowComparisonContractError(
            f"record[{index}] duplicate candidate name/party keys are invalid"
        )
    if candidates != sorted(candidates, key=_candidate_sort_key):
        raise WorkflowComparisonContractError(
            f"record[{index}].candidates must use canonical candidate order"
        )

    grand_total = _validate_vote_state(
        f"record[{index}].grand_total",
        record["grand_total"],
    )

    return {
        "reporting_unit": {
            "name": unit_name,
            "type": unit_type,
        },
        "percent_reporting": percent,
        "vote_methods": list(methods),
        "method_totals": method_totals,
        "candidates": candidates,
        "grand_total": grand_total,
    }


def canonical_semantic_json(semantic: Mapping[str, object]) -> bytes:
    return json.dumps(
        semantic,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def semantic_sha256(semantic: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_semantic_json(semantic)).hexdigest()


def validate_normalized_semantic(raw: object) -> dict[str, object]:
    semantic = _require_mapping("semantic", raw)
    _require_exact_keys("semantic", semantic, SEMANTIC_KEYS)

    scope = _validate_scope(semantic["scope"])
    records_raw = _require_list("semantic.records", semantic["records"])
    records = [
        _validate_record(record, index)
        for index, record in enumerate(records_raw)
    ]

    record_keys = [
        (
            record["reporting_unit"]["name"],
            record["reporting_unit"]["type"],
        )
        for record in records
    ]
    if len(set(record_keys)) != len(record_keys):
        raise WorkflowComparisonContractError(
            "duplicate reporting unit semantic keys are invalid"
        )
    if records != sorted(records, key=_record_sort_key):
        raise WorkflowComparisonContractError(
            "semantic.records must use canonical reporting-unit order"
        )

    return {"scope": scope, "records": records}


def _validate_binding(raw: object) -> dict[str, object]:
    binding = _require_mapping("binding", raw)
    _require_exact_keys("binding", binding, BINDING_KEYS)

    normalized = {
        "workflow_item_id":
            _require_uuid(
                "binding.workflow_item_id",
                binding["workflow_item_id"],
            ),
        "workflow_pass_id":
            _require_uuid(
                "binding.workflow_pass_id",
                binding["workflow_pass_id"],
            ),
        "pass_number":
            _require_positive_int(
                "binding.pass_number",
                binding["pass_number"],
            ),
        "revision_number":
            _require_positive_int(
                "binding.revision_number",
                binding["revision_number"],
            ),
        "source_evidence_ref": binding["source_evidence_ref"],
        "staging_batch_id":
            _require_uuid(
                "binding.staging_batch_id",
                binding["staging_batch_id"],
            ),
        "normalized_artifact_ref":
            normalize_text(binding["normalized_artifact_ref"]),
        "normalized_artifact_sha256":
            _require_sha256(
                "binding.normalized_artifact_sha256",
                binding["normalized_artifact_sha256"],
            ),
    }

    source_ref = binding["source_evidence_ref"]
    if source_ref is not None:
        if not isinstance(source_ref, str) or not source_ref.strip():
            raise WorkflowComparisonContractError(
                "binding.source_evidence_ref must be non-empty string or null"
            )

    if (
        binding["normalized_artifact_ref"]
        != normalized["normalized_artifact_ref"]
    ):
        raise WorkflowComparisonContractError(
            "binding.normalized_artifact_ref must already be normalized"
        )

    return normalized


def validate_comparison_payload(raw: object) -> dict[str, object]:
    payload = _require_mapping("payload", raw)
    _require_exact_keys("payload", payload, TOP_LEVEL_KEYS)

    if payload["schema"] != WORKFLOW_COMPARISON_PAYLOAD_CONTRACT:
        raise WorkflowComparisonContractError(
            "payload.schema does not match v1 comparison contract"
        )
    if (
        payload["schema_version"]
        != WORKFLOW_COMPARISON_PAYLOAD_SCHEMA_VERSION
    ):
        raise WorkflowComparisonContractError(
            "payload.schema_version does not match v1"
        )
    if payload["comparison_version"] != WORKFLOW_COMPARISON_VERSION:
        raise WorkflowComparisonContractError(
            "payload.comparison_version does not match v1"
        )

    binding = _validate_binding(payload["binding"])
    semantic = validate_normalized_semantic(payload["semantic"])
    declared_hash = _require_sha256(
        "payload.semantic_sha256",
        payload["semantic_sha256"],
    )
    computed_hash = semantic_sha256(semantic)
    if declared_hash != computed_hash:
        raise WorkflowComparisonContractError(
            "payload.semantic_sha256 does not match canonical semantic content"
        )

    return {
        "schema": WORKFLOW_COMPARISON_PAYLOAD_CONTRACT,
        "schema_version": WORKFLOW_COMPARISON_PAYLOAD_SCHEMA_VERSION,
        "comparison_version": WORKFLOW_COMPARISON_VERSION,
        "binding": binding,
        "semantic": semantic,
        "semantic_sha256": computed_hash,
    }


def assert_comparable_payloads(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> None:
    left_valid = validate_comparison_payload(left)
    right_valid = validate_comparison_payload(right)

    if left_valid["schema_version"] != right_valid["schema_version"]:
        raise WorkflowComparisonContractError(
            "left/right schema versions must match"
        )
    if (
        left_valid["comparison_version"]
        != right_valid["comparison_version"]
    ):
        raise WorkflowComparisonContractError(
            "left/right comparison versions must match"
        )

    left_binding = left_valid["binding"]
    right_binding = right_valid["binding"]
    assert isinstance(left_binding, Mapping)
    assert isinstance(right_binding, Mapping)

    if (
        left_binding["workflow_item_id"]
        != right_binding["workflow_item_id"]
    ):
        raise WorkflowComparisonContractError(
            "left/right passes must belong to the same workflow item"
        )
    if (
        left_binding["workflow_pass_id"]
        == right_binding["workflow_pass_id"]
    ):
        raise WorkflowComparisonContractError(
            "left/right comparison requires distinct immutable pass revisions"
        )


def strict_semantic_equality(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> bool:
    assert_comparable_payloads(left, right)
    return left["semantic_sha256"] == right["semantic_sha256"]


def build_difference_summary(
    *,
    left_semantic_sha256: str,
    right_semantic_sha256: str,
    category_counts: Mapping[str, int],
) -> dict[str, object]:
    left_hash = _require_sha256(
        "left_semantic_sha256",
        left_semantic_sha256,
    )
    right_hash = _require_sha256(
        "right_semantic_sha256",
        right_semantic_sha256,
    )

    if frozenset(category_counts.keys()) != frozenset(
        DISCREPANCY_CATEGORIES
    ):
        raise WorkflowComparisonContractError(
            "category_counts must contain every v1 discrepancy category exactly"
        )

    normalized_counts: dict[str, int] = {}
    for category in DISCREPANCY_CATEGORIES:
        count = category_counts[category]
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise WorkflowComparisonContractError(
                f"category count for {category!r} "
                "must be nonnegative integer"
            )
        normalized_counts[category] = count

    difference_count = sum(normalized_counts.values())
    return {
        "schema": WORKFLOW_COMPARISON_DIFFERENCE_SUMMARY_SCHEMA,
        "comparison_version": WORKFLOW_COMPARISON_VERSION,
        "left_semantic_sha256": left_hash,
        "right_semantic_sha256": right_hash,
        "difference_count": difference_count,
        "category_counts": normalized_counts,
    }


W8J_SCOPE_FIELD_ORDER = (
    "election_year",
    "election_date",
    "state",
    "jurisdiction_name",
    "jurisdiction_type",
    "contest",
)


def _w8j_nullable_state(value: object) -> str:
    return "null" if value is None else "value"


def _w8j_is_zero(value: object) -> bool:
    return value == 0 or value == "0"


def _w8j_category(
    left_state: str,
    right_state: str,
    left_value: object,
    right_value: object,
) -> str | None:
    if left_state == right_state and left_value == right_value:
        return None
    if left_state == "missing" and right_state != "missing":
        return "missing_left"
    if right_state == "missing" and left_state != "missing":
        return "missing_right"
    if {left_state, right_state} == {"null", "value"}:
        value = right_value if right_state == "value" else left_value
        return "null_vs_zero" if _w8j_is_zero(value) else "null_vs_value"
    return "value_mismatch"


def _w8j_difference(
    *,
    category: str,
    semantic_key: Mapping[str, object],
    left_value: object,
    right_value: object,
    left_state: str | None,
    right_state: str | None,
) -> dict[str, object]:
    if category not in DISCREPANCY_CATEGORIES:
        raise WorkflowComparisonContractError(
            f"unsupported discrepancy category: {category!r}"
        )
    return {
        "category": category,
        "semantic_key": dict(semantic_key),
        "left_value": left_value,
        "right_value": right_value,
        "left_value_state": left_state,
        "right_value_state": right_state,
    }


def _w8j_state_value(
    raw: Mapping[str, object] | None,
    *,
    value_key: str,
) -> tuple[str, object]:
    if raw is None:
        return "missing", None
    state = str(raw["state"])
    return state, raw[value_key]


def _w8j_unit_key(
    record: Mapping[str, object],
) -> tuple[object, object]:
    unit = _require_mapping(
        "record.reporting_unit",
        record["reporting_unit"],
    )
    return unit["type"], unit["name"]


def _w8j_sort_nullable(value: object) -> tuple[bool, str]:
    return value is None, "" if value is None else str(value)


def _w8j_unit_sort_key(
    key: tuple[object, object],
) -> tuple[object, ...]:
    return (*_w8j_sort_nullable(key[0]), *_w8j_sort_nullable(key[1]))


def _w8j_candidate_key(
    candidate: Mapping[str, object],
) -> tuple[object, str]:
    return candidate["party"], str(candidate["name"])


def _w8j_candidate_sort_key(
    key: tuple[object, str],
) -> tuple[object, ...]:
    return (*_w8j_sort_nullable(key[0]), str(key[1]))


def _w8j_method_map(
    values: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    return {str(value["method"]): value for value in values}


def _w8j_append_state_difference(
    differences: list[dict[str, object]],
    *,
    semantic_key: Mapping[str, object],
    left: Mapping[str, object] | None,
    right: Mapping[str, object] | None,
    value_key: str,
) -> None:
    left_state, left_value = _w8j_state_value(left, value_key=value_key)
    right_state, right_value = _w8j_state_value(right, value_key=value_key)
    category = _w8j_category(
        left_state,
        right_state,
        left_value,
        right_value,
    )
    if category is None:
        return
    differences.append(
        _w8j_difference(
            category=category,
            semantic_key=semantic_key,
            left_value=left_value,
            right_value=right_value,
            left_state=left_state,
            right_state=right_state,
        )
    )


def enumerate_semantic_differences(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> list[dict[str, object]]:
    # Pure deterministic strict semantic difference enumeration.
    assert_comparable_payloads(left, right)
    left_valid = validate_comparison_payload(left)
    right_valid = validate_comparison_payload(right)
    if left_valid["semantic_sha256"] == right_valid["semantic_sha256"]:
        return []

    left_semantic = _require_mapping("left.semantic", left_valid["semantic"])
    right_semantic = _require_mapping("right.semantic", right_valid["semantic"])
    differences: list[dict[str, object]] = []

    left_scope = _require_mapping("left.semantic.scope", left_semantic["scope"])
    right_scope = _require_mapping("right.semantic.scope", right_semantic["scope"])
    for field in W8J_SCOPE_FIELD_ORDER:
        left_value = left_scope[field]
        right_value = right_scope[field]
        if left_value == right_value:
            continue
        differences.append(
            _w8j_difference(
                category="scope_mismatch",
                semantic_key={"kind": "scope", "field": field},
                left_value=left_value,
                right_value=right_value,
                left_state=_w8j_nullable_state(left_value),
                right_state=_w8j_nullable_state(right_value),
            )
        )

    left_records = _require_list("left.semantic.records", left_semantic["records"])
    right_records = _require_list("right.semantic.records", right_semantic["records"])
    left_by_unit = {
        _w8j_unit_key(_require_mapping("left.record", record)): record
        for record in left_records
    }
    right_by_unit = {
        _w8j_unit_key(_require_mapping("right.record", record)): record
        for record in right_records
    }

    for unit_key in sorted(
        set(left_by_unit) | set(right_by_unit),
        key=_w8j_unit_sort_key,
    ):
        left_record_raw = left_by_unit.get(unit_key)
        right_record_raw = right_by_unit.get(unit_key)
        unit_semantic_key = {
            "kind": "reporting_unit",
            "type": unit_key[0],
            "name": unit_key[1],
        }
        if left_record_raw is None:
            differences.append(
                _w8j_difference(
                    category="missing_left",
                    semantic_key=unit_semantic_key,
                    left_value=None,
                    right_value=right_record_raw,
                    left_state="missing",
                    right_state="value",
                )
            )
            continue
        if right_record_raw is None:
            differences.append(
                _w8j_difference(
                    category="missing_right",
                    semantic_key=unit_semantic_key,
                    left_value=left_record_raw,
                    right_value=None,
                    left_state="value",
                    right_state="missing",
                )
            )
            continue

        left_record = _require_mapping("left.record", left_record_raw)
        right_record = _require_mapping("right.record", right_record_raw)

        _w8j_append_state_difference(
            differences,
            semantic_key={
                **unit_semantic_key,
                "kind": "percent_reporting",
            },
            left=_require_mapping(
                "left.percent_reporting",
                left_record["percent_reporting"],
            ),
            right=_require_mapping(
                "right.percent_reporting",
                right_record["percent_reporting"],
            ),
            value_key="value",
        )

        left_methods = tuple(str(v) for v in left_record["vote_methods"])
        right_methods = tuple(str(v) for v in right_record["vote_methods"])
        all_methods = ordered_vote_methods(
            tuple(dict.fromkeys((*left_methods, *right_methods)))
        )

        left_method_totals = _w8j_method_map(
            [
                _require_mapping("left.method_total", value)
                for value in left_record["method_totals"]
            ]
        )
        right_method_totals = _w8j_method_map(
            [
                _require_mapping("right.method_total", value)
                for value in right_record["method_totals"]
            ]
        )

        for method in all_methods:
            left_present = method in left_methods
            right_present = method in right_methods
            if left_present != right_present:
                differences.append(
                    _w8j_difference(
                        category=(
                            "missing_left" if not left_present else "missing_right"
                        ),
                        semantic_key={
                            **unit_semantic_key,
                            "kind": "vote_method",
                            "method": method,
                        },
                        left_value=method if left_present else None,
                        right_value=method if right_present else None,
                        left_state="value" if left_present else "missing",
                        right_state="value" if right_present else "missing",
                    )
                )
            _w8j_append_state_difference(
                differences,
                semantic_key={
                    **unit_semantic_key,
                    "kind": "method_total",
                    "method": method,
                },
                left=left_method_totals.get(method),
                right=right_method_totals.get(method),
                value_key="votes",
            )

        left_candidates = {
            _w8j_candidate_key(_require_mapping("left.candidate", candidate)):
                candidate
            for candidate in left_record["candidates"]
        }
        right_candidates = {
            _w8j_candidate_key(_require_mapping("right.candidate", candidate)):
                candidate
            for candidate in right_record["candidates"]
        }

        for candidate_key in sorted(
            set(left_candidates) | set(right_candidates),
            key=_w8j_candidate_sort_key,
        ):
            left_candidate_raw = left_candidates.get(candidate_key)
            right_candidate_raw = right_candidates.get(candidate_key)
            candidate_semantic_key = {
                **unit_semantic_key,
                "kind": "candidate",
                "party": candidate_key[0],
                "candidate": candidate_key[1],
            }
            if left_candidate_raw is None:
                differences.append(
                    _w8j_difference(
                        category="missing_left",
                        semantic_key=candidate_semantic_key,
                        left_value=None,
                        right_value=right_candidate_raw,
                        left_state="missing",
                        right_state="value",
                    )
                )
                continue
            if right_candidate_raw is None:
                differences.append(
                    _w8j_difference(
                        category="missing_right",
                        semantic_key=candidate_semantic_key,
                        left_value=left_candidate_raw,
                        right_value=None,
                        left_state="value",
                        right_state="missing",
                    )
                )
                continue

            left_candidate = _require_mapping(
                "left.candidate",
                left_candidate_raw,
            )
            right_candidate = _require_mapping(
                "right.candidate",
                right_candidate_raw,
            )
            left_votes = _w8j_method_map(
                [
                    _require_mapping("left.candidate.method", value)
                    for value in left_candidate["method_votes"]
                ]
            )
            right_votes = _w8j_method_map(
                [
                    _require_mapping("right.candidate.method", value)
                    for value in right_candidate["method_votes"]
                ]
            )
            for method in all_methods:
                _w8j_append_state_difference(
                    differences,
                    semantic_key={
                        **candidate_semantic_key,
                        "kind": "candidate_method",
                        "method": method,
                    },
                    left=left_votes.get(method),
                    right=right_votes.get(method),
                    value_key="votes",
                )
            _w8j_append_state_difference(
                differences,
                semantic_key={
                    **candidate_semantic_key,
                    "kind": "candidate_total",
                },
                left=_require_mapping(
                    "left.candidate.total_votes",
                    left_candidate["total_votes"],
                ),
                right=_require_mapping(
                    "right.candidate.total_votes",
                    right_candidate["total_votes"],
                ),
                value_key="votes",
            )

        _w8j_append_state_difference(
            differences,
            semantic_key={
                **unit_semantic_key,
                "kind": "grand_total",
            },
            left=_require_mapping("left.grand_total", left_record["grand_total"]),
            right=_require_mapping("right.grand_total", right_record["grand_total"]),
            value_key="votes",
        )

    return differences


#!/usr/bin/env python3
"""ElectionPulse W24C3D pure capacity projection prototype (offline, NOT a route).

Scope: accepted immutable historical SNAPSHOT only. No webapp imports, side effects,
IO apart from invoking an INJECTED bytes-loader after the synthetic gate, Azure,
DB, parser, CLI, telemetry writes, or mutation privileges.

SECURITY: ReadContext is a pure integration contract, NOT a verified credential.
The future server must bind every field from CURRENT REQUEST authenticated
server-owned authority, NEVER client JSON/headers/query strings. This is not
proof of production authentication. In production, use a separately reviewed
route+authorization integration with a fail-closed identity resolver.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Callable, Any

CONTRACT = "w24c3d_operator_capacity_read_v1"
EXPECTED_SNAPSHOT_ID = "ELECTIONPULSE_W24C3B_READ_ONLY_PREVIEW_REV1"
APPROVED_SNAPSHOT_SHA256 = "04ece5376591c572d7da22fbf016fc34eb8edc2b9d78705055170104308dc0bd"
MAX_SNAPSHOT_BYTES = 256 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

class CapacityReadDenied(PermissionError):
    """No operator capability: never disclose sensitive details in response."""

class CapacityEvidenceRejected(ValueError):
    """Evidence bytes were unapproved, malformed, or semantically inconsistent."""

@dataclass(frozen=True)
class ReadContext:
    """Server-provided synthetic inputs, never a source of actual authentication."""
    method: str
    principal_authenticated: bool
    operator_capability_server_verified: bool
    authority_state: str
    session_unexpired: bool
    principal_restricted: bool
    development_bypass: bool


def _authorize(context: ReadContext) -> None:
    if not isinstance(context, ReadContext):
        raise CapacityReadDenied("CAPACITY_READ_AUTHORITY_DENIED")
    if context.method != "GET":
        raise CapacityReadDenied("CAPACITY_READ_METHOD_DENIED")
    # 'is True' is intentional: reject None, 1, and untrusted truthy strings.
    if (context.principal_authenticated is not True
            or context.operator_capability_server_verified is not True
            or context.session_unexpired is not True
            or context.principal_restricted is not False
            or context.development_bypass is not False
            or context.authority_state not in ("fresh_certificate", "certificate_session")):
        raise CapacityReadDenied("CAPACITY_READ_AUTHORITY_DENIED")


def _dict(obj: Any, key: str) -> dict:
    v = obj.get(key) if isinstance(obj, dict) else None
    if not isinstance(v, dict):
        raise CapacityEvidenceRejected("SOURCE_SHAPE_MISMATCH:" + key)
    return v


def _nonnull_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise CapacityEvidenceRejected("UNTRUSTED_NUMERIC_FIELD:" + name)
    return value


def _nullable_positive_float(value: Any, name: str, upper: float | None = None) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise CapacityEvidenceRejected("UNTRUSTED_NUMERIC_FIELD:" + name)
    if upper is not None and value > upper:
        raise CapacityEvidenceRejected("UNTRUSTED_NUMERIC_FIELD:" + name)
    return float(value)


def _known_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise CapacityEvidenceRejected("UNTRUSTED_BOOLEAN_FIELD:" + name)
    return value


def _known_nullable_null(value: Any, name: str) -> None:
    if value is not None:
        raise CapacityEvidenceRejected("UNVERIFIED_MEASUREMENT_MUST_REMAIN_NULL:" + name)
    return None


def _safe_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise CapacityEvidenceRejected("BAD_EVIDENCE_DIGEST:" + name)
    return value


def _safe_utc(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value):
        raise CapacityEvidenceRejected("UNSUPPORTED_TIME_FORMAT:" + name)
    return value


def _validated_snapshot(raw: bytes, expected_sha256: str) -> tuple[dict, str]:
    if not isinstance(raw, bytes) or len(raw) < 100 or len(raw) > MAX_SNAPSHOT_BYTES:
        raise CapacityEvidenceRejected("SNAPSHOT_SIZE_OR_TYPE_INVALID")
    expected = _safe_sha(expected_sha256, "approved_snapshot_sha256")
    if expected != APPROVED_SNAPSHOT_SHA256:
        raise CapacityEvidenceRejected("UNAPPROVED_SNAPSHOT_REFERENCE")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected:
        raise CapacityEvidenceRejected("APPROVED_SNAPSHOT_DIGEST_MISMATCH")
    try:
        data = json.loads(raw.decode("utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(ValueError("not finite")))
    except (UnicodeDecodeError, ValueError) as e:
        raise CapacityEvidenceRejected("INVALID_SNAPSHOT_JSON") from e
    if not isinstance(data, dict) or data.get("id") != EXPECTED_SNAPSHOT_ID or data.get("design_contract") != "W24C3_REV1" or data.get("classification") != "ACCEPTED_HISTORICAL_EVIDENCE_PRESENTATION_NOT_LIVE":
        raise CapacityEvidenceRejected("UNSUPPORTED_SNAPSHOT_AUTHORITY_OR_VERSION")
    return data, digest


def _project(data: dict, digest: str) -> dict:
    if data.get("as_of_utc") != "2026-10-22":
        raise CapacityEvidenceRejected("SNAPSHOT_AS_OF_CHANGED")
    scope = _dict(data, "scope")
    admission = _dict(data, "admission")
    baseline = _dict(data, "service_baseline")
    plan = _dict(baseline, "app_service_plan")
    processing = _dict(data, "processing_evidence")
    decoder = _dict(processing, "local_json_decode")
    oracle = _dict(processing, "source_only_oracle")
    cost = _dict(data, "historical_cost")
    future = _dict(data, "future_ledger_state")
    qualification = _dict(processing, "qualification")

    for name in ("full_parser_executed_for_preview", "new_parser_admission_authorized", "runtime_admission_policy_deployed", "new_routes_added", "production_deployment_authorized", "production_mutation_authorized"):
        if _known_bool(scope.get(name), name):
            raise CapacityEvidenceRejected("SNAPSHOT_NO_LONGER_DESIGN_ONLY:" + name)
    if admission.get("control_status") != "PROPOSED_NOT_DEPLOYED" or admission.get("full_parser_execution_authority") != "NOT_GRANTED":
        raise CapacityEvidenceRejected("UNEXPECTED_ADMISSION_STATE")
    if _nonnull_int(admission.get("production_b1_new_heavy_job_policy_slots"), "b1_policy_slots") != 0:
        raise CapacityEvidenceRejected("UNAUTHORIZED_NONZERO_POLICY_SLOTS")
    _known_nullable_null(admission.get("actual_b1_heavy_job_capacity"), "physical_capacity")
    _known_nullable_null(admission.get("unknown_safe_concurrency"), "safe_concurrency")
    _known_nullable_null(admission.get("unknown_worker_ram_bytes"), "worker_ram")
    _known_nullable_null(admission.get("unknown_worker_temp_bytes"), "worker_temp")
    if decoder.get("one_time_authority") != "PERMANENTLY_SPENT" or _known_bool(decoder.get("full_parser_extraction"), "decode_full_parser"):
        raise CapacityEvidenceRejected("DECODE_AUTHORITY_OR_SCOPE_DRIFT")
    _known_nullable_null(decoder.get("os_process_tree_peak_bytes"), "process_tree_peak")
    _known_nullable_null(decoder.get("temporary_disk_peak_bytes"), "temp_peak")
    if _known_bool(qualification.get("full_parser_invoked"), "qualified_full_parser") or _known_bool(qualification.get("process_tree_peak_qualified"), "process_peak_qualified"):
        raise CapacityEvidenceRejected("QUALIFICATION_EVIDENCE_SCOPE_DRIFT")
    if _known_bool(oracle.get("finalized_csv_validated"), "finalized_csv") or _known_bool(oracle.get("source_only"), "source_only") is not True:
        raise CapacityEvidenceRejected("SOURCE_ORACLE_WRONGLY_PROMOTED")
    if _known_bool(future.get("live_events_available"), "live_events") or _known_bool(future.get("operator_job_cost_records_available"), "cost_records"):
        raise CapacityEvidenceRejected("LIVE_LEDGER_DATA_UNAUTHORIZED")
    _known_nullable_null(cost.get("marginal_cost_per_job_usd"), "marginal_job_cost")
    _known_nullable_null(cost.get("current_monthly_bill_usd"), "current_month_bill")
    _known_nullable_null(cost.get("telemetry_ingestion_cost_usd"), "telemetry_ingestion_cost")

    if cost.get("currency") != "USD" or cost.get("month") != "2026-09":
        raise CapacityEvidenceRejected("BILLING_SOURCE_CHANGED")
    invoice = _nullable_positive_float(cost.get("total_usd"), "invoice_total")
    cpu = _nullable_positive_float(plan.get("cpu_mean_percent"), "cpu_mean_pct", 100)
    memory = _nullable_positive_float(plan.get("memory_p95_percent"), "memory_p95_pct", 100)
    begin = _safe_utc(baseline.get("window_start_utc"), "metric_window_start")
    end = _safe_utc(baseline.get("window_end_utc"), "metric_window_end")
    if end <= begin:
        raise CapacityEvidenceRejected("REVERSED_METRIC_WINDOW")
    original_precincts = _nonnull_int(oracle.get("precinct_unit_count"), "precinct_count")
    missing = oracle.get("expected_method_not_explicit")
    if missing != ["Provisional"]:
        raise CapacityEvidenceRejected("METHOD_AVAILABILITY_CONTRACT_DRIFT")
    _safe_sha(oracle.get("result_zip_sha256"), "oracle_receipt")
    _safe_sha(decoder.get("result_zip_sha256"), "decode_receipt")

    # Explicit allowlist: never copy arbitrary source keys, even when source
    # includes credentials, URLs, hostnames, principal identifiers or vote data.
    return {
        "contract": CONTRACT,
        "as_of_utc": "2026-10-22",
        "observation_class": "ACCEPTED_HISTORICAL_SUMMARY_NOT_LIVE_NOT_RAW_EVIDENCE_REVERIFIED",
        "basis_digest": digest,
        "capacity": {
            "app_service_plan_cpu_mean_pct": cpu,
            "app_service_plan_memory_p95_pct": memory,
            "observed_window_start_utc": begin,
            "observed_window_end_utc": end,
            "aggregate_granularity": "HOURLY_AGGREGATES_NOT_INSTANTANEOUS",
            "instantaneous_headroom_bytes": None,
        },
        "admission": {
            "design_policy_state": "PROPOSED_NOT_DEPLOYED",
            "new_heavy_jobs_allowed": False,
            "policy_validated_heavy_slots": 0,
            "physical_maximum_concurrency": None,
            "reason_code": "UNVERIFIED_FULL_WORKER_CAPACITY_AND_NO_NEW_AUTHORITY",
        },
        "quality": {
            "source_oracle_present": True,
            "finalized_csv_reconciled": False,
            "source_precinct_count": original_precincts,
            "method_availability_class": "PROVISIONAL_NOT_EXPLICIT_IN_SOURCE",
            "evidence_class": "SOURCE_ONLY_ORACLE",
        },
        "cost": {
            "billing_period": "2026-09",
            "historical_invoice_total": invoice,
            "currency": "USD",
            "amount_class": "HISTORICAL_INVOICE_TOTAL_NOT_PER_JOB_COST",
            "job_marginal_cost": None,
            "cost_source_digest": None,  # invoice receipt itself is not embedded or independently verified here
        },
        "freshness": {
            "metrics_stale": True,
            "billing_finality": "NOT_REVALIDATED_IN_C3D",
            "ledger_event_time_utc": None,
            "source_evidence_digest": oracle["result_zip_sha256"],
            "worker_measurements_unproven": True,
        },
    }


def project_authorized_capacity_read(
    *,
    context: ReadContext,
    snapshot_loader: Callable[[], bytes],
    approved_snapshot_sha256: str,
) -> dict:
    """Design-only GET projection; auth gate runs BEFORE invoking loader.

    This is not a Flask endpoint, a permission grant, or a full job ledger.
    The snapshot loader must itself be fixed to a server-owned verified artifact.
    """
    _authorize(context)
    if not callable(snapshot_loader):
        raise CapacityEvidenceRejected("SNAPSHOT_LOADER_NOT_CALLABLE")
    data, digest = _validated_snapshot(snapshot_loader(), approved_snapshot_sha256)
    return _project(data, digest)

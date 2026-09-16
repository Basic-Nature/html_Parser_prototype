"""Pure transport-neutral composition of noncanonical parser observations.

This service combines existing observation projectors for the same
``TablePipelineResult``. It adds no handler, store, socket, runtime, route,
persistence, database, timestamp, canonical output, or finalizer behavior.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from webapp.parser.contracts.table_pipeline import TablePipelineResult
from webapp.parser.services.election_structure_observation import (
    ELECTION_STRUCTURE_OBSERVATION_CONTRACT,
    project_election_structure_observation,
)
from webapp.parser.services.pipeline_inspection import (
    INSPECTION_AUTHORITY,
    INSPECTION_CONTRACT,
    project_pipeline_inspection,
)

PARSER_OBSERVATION_BUNDLE_CONTRACT = "parser_observation_bundle_v1"
PARSER_OBSERVATION_BUNDLE_AUTHORITY = INSPECTION_AUTHORITY


def _validate_common_subpayload(
    payload: Any,
    *,
    label: str,
    expected_contract: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError(f"{label} observation must be a dict")
    if payload.get("contract") != expected_contract:
        raise ValueError(f"{label} observation contract mismatch")

    authority = payload.get("authority")
    if not isinstance(authority, Mapping):
        raise ValueError(f"{label} observation authority is required")
    if authority.get("inspection") != INSPECTION_AUTHORITY:
        raise ValueError(f"{label} observation authority mismatch")
    if authority.get("canonical") is not False:
        raise ValueError(f"{label} observation must remain noncanonical")
    if payload.get("automatic_timestamp") is not False:
        raise ValueError(f"{label} observation must not add timestamps")
    return payload


def _validate_pipeline_inspection(payload: Any) -> dict[str, Any]:
    checked = _validate_common_subpayload(
        payload,
        label="pipeline",
        expected_contract=INSPECTION_CONTRACT,
    )
    if checked.get("rows_included") is not False or "rows" in checked:
        raise ValueError("pipeline observation must not contain raw rows")
    if checked.get("headers_included") is not False or "headers" in checked:
        raise ValueError("pipeline observation must not contain raw headers")

    provenance = checked.get("source_provenance")
    if isinstance(provenance, Mapping):
        if provenance.get("source_uri_included") is not False:
            raise ValueError("pipeline observation must not expose source URI")
        if provenance.get("source_metadata_included") is not False:
            raise ValueError("pipeline observation must not expose source metadata")
        if "source_uri" in provenance or "metadata" in provenance:
            raise ValueError(
                "pipeline observation contains forbidden source provenance fields"
            )
    return checked


def _validate_election_structure(payload: Any) -> dict[str, Any]:
    checked = _validate_common_subpayload(
        payload,
        label="election_structure",
        expected_contract=ELECTION_STRUCTURE_OBSERVATION_CONTRACT,
    )
    if checked.get("raw_rows_included") is not False or "rows" in checked:
        raise ValueError("election-structure observation must not contain raw rows")
    if checked.get("raw_headers_included") is not False or "headers" in checked:
        raise ValueError(
            "election-structure observation must not contain raw header sequence"
        )
    return checked


def project_parser_observation_bundle(
    result: TablePipelineResult,
) -> dict[str, Any]:
    """Compose existing noncanonical observations for one typed parser result."""
    if not isinstance(result, TablePipelineResult):
        raise TypeError("result must be a TablePipelineResult")

    pipeline = _validate_pipeline_inspection(
        project_pipeline_inspection(result)
    )
    election_structure = _validate_election_structure(
        project_election_structure_observation(result)
    )

    return {
        "contract": PARSER_OBSERVATION_BUNDLE_CONTRACT,
        "authority": {
            "inspection": PARSER_OBSERVATION_BUNDLE_AUTHORITY,
            "canonical": False,
        },
        "source_stage": result.stage.value,
        "pipeline_inspection": pipeline,
        "election_structure": election_structure,
        "raw_rows_included": False,
        "raw_headers_included": False,
        "automatic_timestamp": False,
    }

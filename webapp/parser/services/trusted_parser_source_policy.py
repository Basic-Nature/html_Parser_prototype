"""Shared trusted parser-source policy for governed Workflow execution.

Trust requires BOTH prior QC-backed provenance and the source's current exact
maintained-registry authority. Neither a historical verification-status string
nor registry membership alone is sufficient.

This module is pure policy: no database, Flask, parser, filesystem or network
access occurs here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

TRUSTED_PARSER_SOURCE_POLICY = "trusted_parser_source_policy_v1"

PROVENANCE_WORKFLOW_QC = "workflow_qc_approved_v1"
PROVENANCE_LEGACY_PRODUCTION = "legacy_verified_production_v1"
QC_BACKED_PROVENANCE_CLASSES = frozenset({
    PROVENANCE_WORKFLOW_QC,
    PROVENANCE_LEGACY_PRODUCTION,
})


class TrustedParserSourceDenied(PermissionError):
    pass


@dataclass(frozen=True)
class TrustedParserSourceAuthority:
    source_url: str
    provenance_class: str
    registry_category: str
    policy: str = TRUSTED_PARSER_SOURCE_POLICY

    def safe_projection(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "trusted": True,
            "provenance_class": self.provenance_class,
            "registry_category": self.registry_category,
            "source_url_disclosed": False,
        }


def _text(value: object) -> str:
    return str(value or "").strip()


def assert_trusted_parser_source(
    *,
    source_url: object,
    qc_evidence: Mapping[str, object],
    registry_state: Mapping[str, object],
) -> TrustedParserSourceAuthority:
    exact_source = _text(source_url)
    if not exact_source:
        raise TrustedParserSourceDenied("Trusted parser source denied.")

    evidence_url = _text(qc_evidence.get("source_url"))
    provenance_class = _text(qc_evidence.get("provenance_class"))
    if (
        qc_evidence.get("qc_backed") is not True
        or provenance_class not in QC_BACKED_PROVENANCE_CLASSES
        or evidence_url != exact_source
    ):
        raise TrustedParserSourceDenied("Trusted parser source denied.")

    registry_url = _text(registry_state.get("source_url"))
    registry_category = _text(
        registry_state.get("registry_category")
    ).lower()
    review_status = _text(
        registry_state.get("review_status")
    ).lower()

    if (
        registry_state.get("exact_registry_identity") is not True
        or registry_url != exact_source
        or registry_category != "curated"
        or review_status != "approved"
        or registry_state.get("parser_eligible") is not True
        or registry_state.get("quarantined") is True
        or registry_state.get("deprecated") is True
    ):
        raise TrustedParserSourceDenied("Trusted parser source denied.")

    return TrustedParserSourceAuthority(
        source_url=exact_source,
        provenance_class=provenance_class,
        registry_category=registry_category,
    )

from __future__ import annotations

import pytest

from webapp.parser.services.trusted_parser_source_policy import (
    PROVENANCE_LEGACY_PRODUCTION,
    PROVENANCE_WORKFLOW_QC,
    TrustedParserSourceDenied,
    assert_trusted_parser_source,
)

URL = "https://sos.example.gov/results.pdf"


def evidence(provenance=PROVENANCE_WORKFLOW_QC):
    return {
        "source_url": URL,
        "qc_backed": True,
        "provenance_class": provenance,
        "verification_status": "not_authority_by_itself",
    }


def registry():
    return {
        "source_url": URL,
        "exact_registry_identity": True,
        "registry_category": "curated",
        "review_status": "approved",
        "parser_eligible": True,
        "quarantined": False,
        "deprecated": False,
    }


@pytest.mark.parametrize(
    "provenance",
    [PROVENANCE_WORKFLOW_QC, PROVENANCE_LEGACY_PRODUCTION],
)
def test_combined_trust_accepts_explicit_qc_classes(provenance):
    authority = assert_trusted_parser_source(
        source_url=URL,
        qc_evidence=evidence(provenance),
        registry_state=registry(),
    )
    safe = authority.safe_projection()
    assert safe["trusted"] is True
    assert safe["source_url_disclosed"] is False
    assert URL not in repr(safe)


@pytest.mark.parametrize(
    ("side", "key", "value"),
    [
        ("evidence", "qc_backed", False),
        ("evidence", "provenance_class", "verified_string_only"),
        ("evidence", "source_url", "https://other.example/results.pdf"),
        ("registry", "exact_registry_identity", False),
        ("registry", "source_url", "https://other.example/results.pdf"),
        ("registry", "registry_category", "backlog"),
        ("registry", "review_status", "quarantined"),
        ("registry", "parser_eligible", False),
        ("registry", "quarantined", True),
        ("registry", "deprecated", True),
    ],
)
def test_combined_trust_fails_closed(side, key, value):
    qc = evidence()
    reg = registry()
    (qc if side == "evidence" else reg)[key] = value
    with pytest.raises(
        TrustedParserSourceDenied,
        match=r"^Trusted parser source denied\.$",
    ):
        assert_trusted_parser_source(
            source_url=URL,
            qc_evidence=qc,
            registry_state=reg,
        )

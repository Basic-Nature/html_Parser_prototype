# Contracts for Tranche 1F-B integrity trust delegation.

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from webapp.parser.trust_authority import (
    adjust_integrity_contamination,
    uses_strict_verified_anomaly_policy,
)
from webapp.parser.utils.privilege_tiers import PrivilegeTier


ROOT = Path(__file__).resolve().parents[2]
INTEGRITY_PATH = (
    ROOT
    / "webapp"
    / "parser"
    / "Context_Integration"
    / "Integrity_check.py"
)
TRUST_AUTHORITY_PATH = (
    ROOT / "webapp" / "parser" / "trust_authority.py"
)


def test_strict_verified_integrity_policy_matrix():
    assert not uses_strict_verified_anomaly_policy(
        None,
        verified_domain=True,
    )
    assert not uses_strict_verified_anomaly_policy(
        PrivilegeTier.STANDARD_USER,
        verified_domain=True,
    )
    assert not uses_strict_verified_anomaly_policy(
        PrivilegeTier.ADMIN_REVIEWER,
        verified_domain=True,
    )
    assert uses_strict_verified_anomaly_policy(
        PrivilegeTier.ADMIN_FULL_TRUST,
        verified_domain=True,
    )
    assert uses_strict_verified_anomaly_policy(
        PrivilegeTier.ROOT_ADMIN,
        verified_domain=True,
    )
    assert not uses_strict_verified_anomaly_policy(
        PrivilegeTier.ROOT_ADMIN,
        verified_domain=False,
    )


def test_integrity_contamination_ceiling_preserves_existing_semantics():
    assert adjust_integrity_contamination(
        0.05,
        PrivilegeTier.ADMIN_FULL_TRUST,
        verified_domain=True,
    ) == 0.01

    assert adjust_integrity_contamination(
        0.20,
        PrivilegeTier.ROOT_ADMIN,
        verified_domain=True,
    ) == 0.01

    assert adjust_integrity_contamination(
        0.005,
        PrivilegeTier.ADMIN_FULL_TRUST,
        verified_domain=True,
    ) == 0.005

    assert adjust_integrity_contamination(
        0.05,
        PrivilegeTier.ADMIN_REVIEWER,
        verified_domain=True,
    ) == 0.05

    assert adjust_integrity_contamination(
        0.05,
        PrivilegeTier.ADMIN_FULL_TRUST,
        verified_domain=False,
    ) == 0.05


def test_integrity_module_delegates_both_full_trust_decisions():
    source = INTEGRITY_PATH.read_text(encoding="utf-8")

    assert "adjust_integrity_contamination(" in source
    assert "uses_strict_verified_anomaly_policy(" in source
    assert (
        "privilege_tier >= PrivilegeTier.ADMIN_FULL_TRUST"
        not in source
    )


def test_integrity_summary_presentation_stays_in_integrity_module():
    integrity_source = INTEGRITY_PATH.read_text(encoding="utf-8")
    trust_source = TRUST_AUTHORITY_PATH.read_text(encoding="utf-8")

    strict_label = (
        "strict_verified (only severe anomalies flagged)"
    )
    root_label = (
        "all_anomalies_reviewed (root admin bypass)"
    )
    standard_label = (
        "standard (default thresholds)"
    )

    for label in (
        strict_label,
        root_label,
        standard_label,
    ):
        assert label in integrity_source
        assert label not in trust_source


def test_integrity_public_function_signatures_remain_stable():
    import webapp.parser.Context_Integration.Integrity_check as integrity

    detect_sig = inspect.signature(
        integrity.detect_anomalies_with_ml
    )
    analyze_sig = inspect.signature(
        integrity.analyze_contests
    )

    assert list(detect_sig.parameters) == [
        "contexts",
        "contamination",
        "n_estimators",
        "random_state",
        "embedding_model",
        "trust_factors",
        "privilege_tier",
    ]

    assert list(analyze_sig.parameters) == [
        "contests",
        "expected_year",
        "context_library_path",
        "trust_factors",
        "privilege_tier",
    ]


def test_trust_authority_stays_pure_of_ml_and_monitor_lifecycle():
    source = TRUST_AUTHORITY_PATH.read_text(encoding="utf-8")

    forbidden = (
        "IsolationForest",
        "DBSCAN",
        "monitor_db_for_alerts",
        "threading",
        "get_session",
        "safe_execute",
        "socketio",
        "session_manager",
    )

    for token in forbidden:
        assert token not in source


def test_integrity_monitor_function_still_exists_locally():
    source = INTEGRITY_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    functions = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }

    assert "monitor_db_for_alerts" in functions

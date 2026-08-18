# Structural and behavioral contract for Tranche 1E-A authorization comparator.

from __future__ import annotations

import ast
from pathlib import Path

from webapp.parser.auth.authorization import tier_satisfies
from webapp.parser.utils.privilege_tiers import PrivilegeTier


ROOT = Path(__file__).resolve().parents[2]

AUTHORIZATION_PATH = ROOT / "webapp" / "parser" / "auth" / "authorization.py"
MONOLITH_PATH = ROOT / "webapp" / "Smart_Elections_Parser_Webapp.py"
QA_PATH = ROOT / "webapp" / "parser" / "quality_assurance" / "qa_endpoints.py"
VERIFICATION_PATH = ROOT / "webapp" / "parser" / "verification_endpoints.py"
PRIVILEGE_PATH = ROOT / "webapp" / "parser" / "utils" / "privilege_tiers.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_tier_satisfies_preserves_ordering_semantics():
    assert tier_satisfies(
        PrivilegeTier.ADMIN_REVIEWER,
        PrivilegeTier.STANDARD_USER,
    )
    assert tier_satisfies(
        PrivilegeTier.ADMIN_REVIEWER,
        PrivilegeTier.ADMIN_REVIEWER,
    )
    assert not tier_satisfies(
        PrivilegeTier.STANDARD_USER,
        PrivilegeTier.ADMIN_REVIEWER,
    )
    assert tier_satisfies(
        PrivilegeTier.ROOT_ADMIN,
        PrivilegeTier.ADMIN_FULL_TRUST,
    )


def test_tier_satisfies_supports_integer_and_enum_inputs():
    assert tier_satisfies(2, 1)
    assert tier_satisfies(2, PrivilegeTier.ADMIN_FULL_TRUST)
    assert tier_satisfies(PrivilegeTier.ADMIN_FULL_TRUST, 2)
    assert not tier_satisfies(1, 2)


def test_missing_actual_tier_is_not_authorized():
    assert not tier_satisfies(None, PrivilegeTier.STANDARD_USER)


def test_authorization_module_does_not_absorb_identity_or_trust_policy():
    source = _source(AUTHORIZATION_PATH)

    forbidden = (
        "get_principal_tier",
        "get_tier_trust_thresholds",
        "should_apply_admin_boost",
        "should_quarantine",
        "should_reject",
        "extract_client_principal",
        "PrivilegeTier",
        "flask",
        "jsonify",
    )

    for token in forbidden:
        assert token not in source


def test_callers_delegate_comparison_but_keep_policy_ownership():
    monolith = _source(MONOLITH_PATH)
    qa = _source(QA_PATH)
    verification = _source(VERIFICATION_PATH)
    privilege = _source(PRIVILEGE_PATH)

    assert "tier_satisfies(actual_tier, required_tier)" in monolith
    assert "tier_satisfies(actual_tier, minimum_tier)" in monolith
    assert (
        "tier_satisfies(principal_tier, PrivilegeTier.ADMIN_REVIEWER)"
        in monolith
    )
    assert "tier_satisfies(actual, required)" in qa
    assert "tier_satisfies(actual_tier, required_tier)" in verification
    assert "tier_satisfies(tier, minimum_tier)" in privilege

    assert "int(actual) < int(required)" not in qa
    assert "int(actual_tier) < int(required_tier)" not in verification
    assert (
        "int(principal_tier) < int(PrivilegeTier.ADMIN_REVIEWER)"
        not in monolith
    )


def test_legacy_require_minimum_tier_keeps_explicit_none_behavior():
    source = _source(PRIVILEGE_PATH)
    tree = ast.parse(source)

    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }

    node = functions["require_minimum_tier"]
    segment = ast.get_source_segment(source, node)

    assert segment is not None
    assert "tier is not None" in segment
    assert "not tier_satisfies(tier, minimum_tier)" in segment

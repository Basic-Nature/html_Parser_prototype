
# Structural contract for Tranche 1D tier normalization.

from __future__ import annotations

import ast
from pathlib import Path

from webapp.parser.auth.tiers import (
    normalize_required_tier,
    tier_display,
    tier_level,
    tier_name,
)
from webapp.parser.utils.privilege_tiers import PrivilegeTier


ROOT = Path(__file__).resolve().parents[2]

TIERS_PATH = ROOT / "webapp" / "parser" / "auth" / "tiers.py"
PRIVILEGE_PATH = (
    ROOT / "webapp" / "parser" / "utils" / "privilege_tiers.py"
)
QA_PATH = (
    ROOT
    / "webapp"
    / "parser"
    / "quality_assurance"
    / "qa_endpoints.py"
)
VERIFICATION_PATH = (
    ROOT
    / "webapp"
    / "parser"
    / "verification_endpoints.py"
)


def _top_level_functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def test_privilege_enum_identity_and_values_remain_stable():
    assert PrivilegeTier.__module__ == (
        "webapp.parser.utils.privilege_tiers"
    )

    assert int(PrivilegeTier.STANDARD_USER) == 0
    assert int(PrivilegeTier.ADMIN_REVIEWER) == 1
    assert int(PrivilegeTier.ADMIN_FULL_TRUST) == 2
    assert int(PrivilegeTier.ROOT_ADMIN) == 3

    assert PrivilegeTier.STANDARD_USER.name_display == "Standard User"
    assert PrivilegeTier.ADMIN_REVIEWER.name_display == "Admin Reviewer"
    assert PrivilegeTier.ADMIN_FULL_TRUST.name_display == "Admin Full Trust"
    assert PrivilegeTier.ROOT_ADMIN.name_display == "Root Admin"


def test_required_tier_normalization_preserves_existing_aliases():
    assert normalize_required_tier("reviewer") is PrivilegeTier.STANDARD_USER
    assert normalize_required_tier("standard_user") is PrivilegeTier.STANDARD_USER
    assert normalize_required_tier("admin_reviewer") is PrivilegeTier.ADMIN_REVIEWER
    assert normalize_required_tier("admin_full_trust") is PrivilegeTier.ADMIN_FULL_TRUST
    assert normalize_required_tier("root_admin") is PrivilegeTier.ROOT_ADMIN
    assert normalize_required_tier(" ADMIN_REVIEWER ") is PrivilegeTier.ADMIN_REVIEWER
    assert normalize_required_tier("admin-reviewer") is PrivilegeTier.ADMIN_REVIEWER


def test_unknown_required_tier_remains_fail_closed():
    assert normalize_required_tier("unknown-tier") is PrivilegeTier.ROOT_ADMIN
    assert normalize_required_tier("") is PrivilegeTier.ROOT_ADMIN


def test_tier_presentation_helpers_preserve_contract():
    assert tier_level(PrivilegeTier.ADMIN_REVIEWER) == 1
    assert tier_name(PrivilegeTier.ADMIN_FULL_TRUST) == "ADMIN_FULL_TRUST"
    assert tier_display(PrivilegeTier.ROOT_ADMIN) == "Root Admin"

    assert tier_level(2) == 2
    assert tier_name(2) == "ADMIN_FULL_TRUST"
    assert tier_display(2) == "Admin Full Trust"


def test_qa_and_verification_keep_compatibility_wrappers():
    for path in (QA_PATH, VERIFICATION_PATH):
        source = path.read_text(encoding="utf-8")
        functions = _top_level_functions(path)
        node = functions["_normalize_required_tier"]
        segment = ast.get_source_segment(source, node)

        assert segment is not None
        assert "return normalize_required_tier(tier)" in segment
        assert '"admin_reviewer": PrivilegeTier.ADMIN_REVIEWER' not in segment
        assert '"root_admin": PrivilegeTier.ROOT_ADMIN' not in segment


def test_tier_vocabulary_does_not_absorb_resolution_authorization_or_trust():
    source = TIERS_PATH.read_text(encoding="utf-8")

    assert "def get_principal_tier" not in source
    assert "def get_tier_trust_thresholds" not in source
    assert "def should_apply_admin_boost" not in source
    assert "def require_minimum_tier" not in source
    assert "should_quarantine" not in source
    assert "should_reject" not in source


def test_legacy_privilege_module_still_owns_later_tranche_functions():
    functions = _top_level_functions(PRIVILEGE_PATH)

    for name in (
        "get_tier_trust_thresholds",
        "get_principal_tier",
        "should_apply_admin_boost",
        "require_minimum_tier",
    ):
        assert name in functions

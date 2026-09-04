from __future__ import annotations

from pathlib import Path


ROOT = Path(".")
MAIN = ROOT / "webapp/Smart_Elections_Parser_Webapp.py"
F2_TEMPLATE = ROOT / "webapp/templates/ballot_lens_f2.html"
QUALITY_DASHBOARD = ROOT / "webapp/templates/quality_dashboard.html"
URL_STATUS_DASHBOARD = ROOT / "webapp/templates/url_status_dashboard.html"
WORKFLOW = ROOT / ".github/workflows/main_ballotlens.yml"
ADVANCED_TOOL = ROOT / "tools/advanced_live_validation.py"
COMPREHENSIVE_TOOL = ROOT / "tools/comprehensive_ui_validation.py"

RETIRED_DEPENDENCY_PATHS = (
    ROOT / "webapp/static/js/__tests__/ballot_lens_modern.chip-transitions.test.js",
    ROOT / "webapp/static/js/__tests__/ballot_lens_modern.placeholder.test.js",
    ROOT / "webapp/static/js/__tests__/placeholder.migration.test.js",
    ROOT / "webapp/tests/test_ballot_lens_render_pressure_contract.py",
    ROOT / "webapp/tests/test_ballot_lens_socket_initialization_contract.py",
    ROOT / "webapp/tests/test_pipeline_inspection_frontend_contract.py",
    ROOT / "webapp/tests/test_pipeline_inspection_why_panel_contract.py",
    ROOT / "webapp/tools/debug_headless_output/final_page.html",
)

MIGRATED_MIXED_CONTRACTS = (
    ROOT / "webapp/tests/test_ballot_lens_csp_inline_style_contract.py",
    ROOT / "webapp/tests/test_ballot_lens_f2_foundation_contract.py",
    ROOT / "webapp/tests/test_ballot_lens_f2j_j2a_alias_extinction_contract.py",
    ROOT / "webapp/tests/test_ballot_lens_f2j_j2b_shared_css_consumer_contract.py",
    ROOT / "webapp/tests/test_ballot_lens_f2j_route_cutover_contract.py",
    ROOT / "webapp/tests/test_ballot_lens_public_registry_ui_projection_contract.py",
    ROOT / "webapp/tests/test_canonical_consumer_adapter_contract.py",
    ROOT / "webapp/tests/test_prelaunch_ui_security_contract.py",
    ROOT / "webapp/tests/test_public_read_privileged_mutation_authority_contract.py",
    ROOT / "webapp/tests/test_url_registry_contract.py",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def test_j2c_primary_f2_and_j2b_dashboard_cutovers_remain_closed():
    main = _read(MAIN)
    start = main.index("def ballot_lens():")
    end = main.index("def worklist():", start)
    body = main[start:end]
    f2 = _read(F2_TEMPLATE)
    quality = _read(QUALITY_DASHBOARD)
    status = _read(URL_STATUS_DASHBOARD)

    assert '"ballot_lens_f2.html"' in body
    assert '"ballot_lens.html"' not in body
    assert "/ballot_lens_modern" not in body
    assert 'id="ballotLensF2Root"' in f2
    assert "ballot_lens_modern.css" not in quality
    assert "ballot_lens_modern.css" not in status
    assert "quality_dashboard.css" in quality
    assert "quality_dashboard.css" in status


def test_j2c_obsolete_legacy_only_test_and_debug_dependencies_are_extinct():
    assert all(not path.exists() for path in RETIRED_DEPENDENCY_PATHS)


def test_j2c_workflow_no_longer_invokes_retired_ballot_lens_jest_suite():
    workflow = _read(WORKFLOW)
    assert "ballot_lens_modern.placeholder.test.js" not in workflow
    assert "ballot_lens_modern.chip-transitions.test.js" not in workflow
    assert "placeholder.migration.test.js" not in workflow
    assert "auth_utils.contract.test.js" in workflow
    assert "data_framework.contract.test.js" in workflow
    assert "Ballot Lens F2 is enforced by the blocking frontend job" in workflow


def test_j2c_live_validation_tools_target_f2_bootstrap_and_assets():
    for tool in (ADVANCED_TOOL, COMPREHENSIVE_TOOL):
        source = _read(tool)
        assert "/ballot_lens" in source
        assert "ballotLensF2Root" in source
        assert "data-public-registry-api" in source
        assert "ballot_lens_modern.js" not in source
        assert "ballot_lens_modern.css" not in source
        assert "ballot_lens_public_registry.js" not in source


def test_j2c_mixed_contracts_do_not_hold_retired_file_path_dependencies():
    forbidden_paths = (
        "webapp/templates/ballot_lens.html",
        "webapp/static/js/ballot_lens_modern.js",
        "webapp/static/js/ballot_lens_public_registry.js",
        "webapp/static/css/ballot_lens_modern.css",
    )
    for path in MIGRATED_MIXED_CONTRACTS:
        source = _read(path)
        for forbidden in forbidden_paths:
            assert forbidden not in source

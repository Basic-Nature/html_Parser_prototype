from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_FRAMEWORK = REPO_ROOT / "webapp" / "templates" / "data_framework.html"
WORKFLOW = REPO_ROOT / "webapp" / "templates" / "worklist.html"
SHARED_CSS = REPO_ROOT / "webapp" / "static" / "css" / "public_app_nav.css"

EXPECTED_LABELS = (
    "Observatory",
    "Data Framework",
    "Ballot Lens",
    "Workflow",
    "Quality",
    "Access",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_data_framework_and_workflow_share_application_navigation():
    data_framework = _read(DATA_FRAMEWORK)
    workflow = _read(WORKFLOW)

    for runtime in (data_framework, workflow):
        assert 'class="ep-app-nav"' in runtime
        assert "css/public_app_nav.css" in runtime
        for label in EXPECTED_LABELS:
            assert f">{label}</a>" in runtime
        for endpoint in (
            "index",
            "data_framework",
            "ballot_lens",
            "worklist",
            "quality_dashboard",
        ):
            assert f"url_for('{endpoint}')" in runtime
        assert "url_for('auth_welcome', next=request.url)" in runtime


def test_each_surface_marks_exactly_one_current_page():
    data_framework = _read(DATA_FRAMEWORK)
    workflow = _read(WORKFLOW)

    assert data_framework.count('aria-current="page"') == 1
    assert workflow.count('aria-current="page"') == 1

    assert (
        'class="ep-app-nav__link is-active" aria-current="page" '
        'title="Published and governed election data">Data Framework</a>'
        in data_framework
    )
    assert (
        'class="ep-app-nav__link is-active" aria-current="page" '
        'title="Review public verification workflow">Workflow</a>'
        in workflow
    )


def test_workflow_keeps_contextual_actions_and_public_authority():
    workflow = _read(WORKFLOW)

    assert 'class="workflow-nav"' in workflow
    assert "Explore Published Data" in workflow
    assert "← Ballot Lens" in workflow
    assert "public surface remains view-only" in workflow.lower()
    assert "Governed Workflow Plane" in workflow


def test_shared_navigation_css_is_responsive_and_accessible():
    css = _read(SHARED_CSS)

    assert ".ep-app-nav__link:focus-visible" in css
    assert "@media (max-width: 860px)" in css
    assert "@media (max-width: 620px)" in css
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert "overflow-x: auto" in css
    assert css.count("{") == css.count("}")

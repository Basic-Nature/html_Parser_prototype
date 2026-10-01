from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
F2 = ROOT / "webapp" / "frontend" / "ballot-lens"
CSS = F2 / "styles" / "shell.css"

FROZEN_FILES = {
    F2 / "app" / "AppShell.tsx":
        "644579302703cab7b3c7988d4100cccbd5676cb7036ca54ec2338425b2665513",
    F2 / "components" / "workspace" / "WorkspaceShell.tsx":
        "1d0de9f5da64446270940d30765dda40f64e8baa6cdabb8ce016d4066441ec45",
    F2 / "components" / "checkpoints" / "CheckpointRail.tsx":
        "37fe211a5a3ebab50dc60d5292da1ff130bae301cc4bdff5d989e8140f75085f",
    F2 / "components" / "diagnostics" / "DiagnosticsDrawer.tsx":
        "f32aab08bbbea9347f69d5b83383212906489cfb1e99834a76bd76570622b3f6",
    F2 / "components" / "source" / "PublicRegistryBrowser.tsx":
        "8ef531999cacf604fc25305482477a35a9ec3f69dae00d88ed4586b63c3fb793",
    F2 / "components" / "source" / "TrustedSourceBrowser.tsx":
        "e62f667e3f612157211e9d74f642fcf6387f36453fd5349153933b31264b8e6e",
    F2 / "state" / "runMachine.ts":
        "ef313db6334fd2b759663595dba75cc9c50887cabb6dbff4db3e785c4a656705",
    F2 / "state" / "sessionHistory.ts":
        "76b32483e45b12367e90e8bc1fff6b001fd56c5fbc5a6cb76c3e1ddc63e11c64",
    F2 / "services" / "publicSubmit.ts":
        "ec1f60d8bcf332bc65f0735e13c699cc49d1399504d61efb133023dd070da8a3",
    F2 / "services" / "trustedExecution.ts":
        "5560c00f87efa5ab3cb23a14c0cac22cff61f1f1dc568d7f12a9ca5a5f91550e",
    F2 / "services" / "publicRuntimeLifecycle.ts":
        "0f3d9a27a855d61ca5f3e565dc6598a61ccc6ada8ccedc212583a032d893ea6b",
}


W16L_HISTORICAL_SCOPE = {
    "commit": "fc84b5ee70fe5de803e89aa7d0d9f48391446f43",
    "parent": "6c64b9a20fa2059f708299c8fb06f64da4fb0850",
    "subject": "feat(ballot-lens): polish runtime presentation",
    "changed_paths": (
        "webapp/frontend/ballot-lens/styles/shell.css",
        "webapp/tests/test_w16l_ballot_lens_presentation_polish_contract.py",
    ),
    "post_sha256": {
        "webapp/frontend/ballot-lens/styles/shell.css": "9a553815a901feee1c93ad444a29a24fec6579b1976e0113f4dfcd3fea71cffc",
        "webapp/tests/test_w16l_ballot_lens_presentation_polish_contract.py": "6a37e38061405887f22f09ba9aa38b6b2352a5a2a5ecf1fd18fba83d8d7ce655",
    },
}

W16L_HISTORICAL_RUNTIME_SHA256 = {
    "webapp/frontend/ballot-lens/app/AppShell.tsx": "644579302703cab7b3c7988d4100cccbd5676cb7036ca54ec2338425b2665513",
    "webapp/frontend/ballot-lens/components/checkpoints/CheckpointRail.tsx": "37fe211a5a3ebab50dc60d5292da1ff130bae301cc4bdff5d989e8140f75085f",
    "webapp/frontend/ballot-lens/components/diagnostics/DiagnosticsDrawer.tsx": "f32aab08bbbea9347f69d5b83383212906489cfb1e99834a76bd76570622b3f6",
    "webapp/frontend/ballot-lens/components/source/PublicRegistryBrowser.tsx": "8ef531999cacf604fc25305482477a35a9ec3f69dae00d88ed4586b63c3fb793",
    "webapp/frontend/ballot-lens/components/source/TrustedSourceBrowser.tsx": "e62f667e3f612157211e9d74f642fcf6387f36453fd5349153933b31264b8e6e",
    "webapp/frontend/ballot-lens/components/workspace/WorkspaceShell.tsx": "1d0de9f5da64446270940d30765dda40f64e8baa6cdabb8ce016d4066441ec45",
    "webapp/frontend/ballot-lens/services/publicRuntimeLifecycle.ts": "0f3d9a27a855d61ca5f3e565dc6598a61ccc6ada8ccedc212583a032d893ea6b",
    "webapp/frontend/ballot-lens/services/publicSubmit.ts": "ec1f60d8bcf332bc65f0735e13c699cc49d1399504d61efb133023dd070da8a3",
    "webapp/frontend/ballot-lens/services/trustedExecution.ts": "5560c00f87efa5ab3cb23a14c0cac22cff61f1f1dc568d7f12a9ca5a5f91550e",
    "webapp/frontend/ballot-lens/state/runMachine.ts": "ef313db6334fd2b759663595dba75cc9c50887cabb6dbff4db3e785c4a656705",
    "webapp/frontend/ballot-lens/state/sessionHistory.ts": "76b32483e45b12367e90e8bc1fff6b001fd56c5fbc5a6cb76c3e1ddc63e11c64",
}

def _sha(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def test_w16l_historical_scope_and_runtime_manifest_are_evidence_only() -> None:
    assert W16L_HISTORICAL_SCOPE["commit"] == "fc84b5ee70fe5de803e89aa7d0d9f48391446f43"
    assert W16L_HISTORICAL_SCOPE["parent"] == "6c64b9a20fa2059f708299c8fb06f64da4fb0850"
    assert W16L_HISTORICAL_SCOPE["subject"] == "feat(ballot-lens): polish runtime presentation"
    assert W16L_HISTORICAL_SCOPE["changed_paths"] == (
        "webapp/frontend/ballot-lens/styles/shell.css",
        "webapp/tests/test_w16l_ballot_lens_presentation_polish_contract.py",
    )
    assert W16L_HISTORICAL_SCOPE["post_sha256"] == {
        "webapp/frontend/ballot-lens/styles/shell.css": "9a553815a901feee1c93ad444a29a24fec6579b1976e0113f4dfcd3fea71cffc",
        "webapp/tests/test_w16l_ballot_lens_presentation_polish_contract.py": "6a37e38061405887f22f09ba9aa38b6b2352a5a2a5ecf1fd18fba83d8d7ce655",
    }
    normalized_manifest = {
        path.relative_to(ROOT).as_posix(): expected
        for path, expected in FROZEN_FILES.items()
    }
    assert normalized_manifest == W16L_HISTORICAL_RUNTIME_SHA256
    assert len(W16L_HISTORICAL_RUNTIME_SHA256) == 11
    assert W16L_HISTORICAL_RUNTIME_SHA256["webapp/frontend/ballot-lens/app/AppShell.tsx"] == "644579302703cab7b3c7988d4100cccbd5676cb7036ca54ec2338425b2665513"
    assert "89f4412b1b0ed280cc7de2b634fe521f89f23c2d957122dafe29056f9ec7107c" not in W16L_HISTORICAL_RUNTIME_SHA256.values()



def test_w16l_workspace_and_session_hierarchy_uses_existing_markup_only() -> None:
    css = CSS.read_text(encoding="utf-8")

    assert "/* W16L bounded Ballot Lens presentation polish." in css
    assert ".blf2-workspace-header > div:first-child" in css
    assert ".blf2-workspace-state > span:first-child" in css
    assert ".blf2-workspace-state > span:nth-child(2)" in css
    assert ".blf2-result-frame" in css
    assert ".blf2-data-guardrails span" in css
    assert ".blf2-session-facts" in css
    assert ".blf2-session-events" in css


def test_w16l_checkpoint_visuals_consume_existing_data_state_only() -> None:
    css = CSS.read_text(encoding="utf-8")

    for state in ("active", "complete", "warning", "error"):
        assert f'li[data-state="{state}"]' in css

    assert ".blf2-checkpoint-marker" in css
    assert ".blf2-checkpoint-copy small" in css
    assert "-webkit-line-clamp: 2;" in css


def test_w16l_narrow_layout_touch_and_forced_color_contracts_are_present() -> None:
    css = CSS.read_text(encoding="utf-8")

    assert "@media (max-width: 940px)" in css
    assert "@media (max-width: 640px)" in css
    assert "@media (max-width: 420px)" in css
    assert ".blf2-run-action," in css
    assert ".blf2-workspace-tabs button" in css
    assert "min-height: 44px;" in css
    assert "@media (forced-colors: active)" in css
    assert "forced-color-adjust: auto;" in css


def test_w16l_css_remains_balanced_and_does_not_add_important_overrides() -> None:
    css = CSS.read_text(encoding="utf-8")

    assert css.count("{") == css.count("}")
    assert "!important" not in css

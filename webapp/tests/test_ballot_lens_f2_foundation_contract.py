from __future__ import annotations

import json
from pathlib import Path

import pytest

from webapp.parser.frontend_assets import load_ballot_lens_f2_assets


ROOT = Path(".")
MAIN = ROOT / "webapp/Smart_Elections_Parser_Webapp.py"
LEGACY_TEMPLATE = ROOT / "webapp/templates/ballot_lens.html"
F2_TEMPLATE = ROOT / "webapp/templates/ballot_lens_f2.html"
F2_MAIN = ROOT / "webapp/frontend/ballot-lens/main.tsx"
F2_APP = ROOT / "webapp/frontend/ballot-lens/app/App.tsx"
F2_PACKAGE = ROOT / "webapp/frontend/ballot-lens/package.json"
ROOT_PACKAGE = ROOT / "package.json"
DOCKERFILE = ROOT / "Dockerfile"
WORKFLOW = ROOT / ".github/workflows/main_ballotlens.yml"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def test_f2_variant_is_server_controlled_and_legacy_default():
    source = _read(MAIN)
    start = source.index("def ballot_lens():")
    end = source.index("def ballot_lens_modern():", start)
    body = source[start:end]

    assert 'os.environ.get("BALLOT_LENS_UI_VARIANT", "legacy")' in body
    assert '{"legacy", "f2"}' in body
    assert '"ballot_lens.html"' in body
    assert '"ballot_lens_f2.html"' in body
    assert 'ballot_lens_ui_variant == "f2"' in body
    assert "load_ballot_lens_f2_assets()" in body


def test_legacy_ballot_lens_template_remains_separate():
    legacy = _read(LEGACY_TEMPLATE)
    f2 = _read(F2_TEMPLATE)

    assert 'id="btnRunParser2"' in legacy
    assert 'id="ballotLensF2Root"' not in legacy
    assert 'id="ballotLensF2Root"' in f2
    assert "ballot_lens_modern.js" not in f2
    assert "ballot_lens_public_registry.js" not in f2
    assert "ballot_lens_modern.css" not in f2


def test_f2_foundation_has_no_parser_or_socket_execution():
    source = (_read(F2_MAIN) + "\n" + _read(F2_APP)).lower()
    assert "socket.emit" not in source
    assert "socket.on" not in source
    assert "fetch(" not in source
    assert "registry_source_id" not in source
    assert "direct_urls" not in source


def test_f2_isolated_package_does_not_modify_root_tooling_contract():
    root = json.loads(_read(ROOT_PACKAGE))
    f2 = json.loads(_read(F2_PACKAGE))

    assert "f2:typecheck" not in root["scripts"]
    assert "react" not in root.get("dependencies", {})
    assert f2["name"] == "@electionpulse/ballot-lens-f2"
    assert f2["engines"]["node"] == ">=24.0.0"
    assert f2["dependencies"]["react"] == "19.2.8"
    assert f2["dependencies"]["react-dom"] == "19.2.8"
    assert f2["devDependencies"]["@vitejs/plugin-react"] == "6.1.1"
    assert f2["devDependencies"]["vite"] == "8.2.2"
    assert f2["devDependencies"]["typescript"] == "5.9.3"


def test_f2_asset_loader_accepts_entry_css_list(tmp_path: Path):
    dist = tmp_path / "dist/ballot-lens-f2"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (assets / "main-abc.js").write_text("export {};", encoding="utf-8")
    (assets / "main-abc.css").write_text(":root{}", encoding="utf-8")
    (dist / "manifest.json").write_text(
        json.dumps(
            {
                "main.tsx": {
                    "file": "assets/main-abc.js",
                    "css": ["assets/main-abc.css"],
                    "isEntry": True,
                }
            }
        ),
        encoding="utf-8",
    )
    assert load_ballot_lens_f2_assets(static_root=tmp_path) == {
        "script": "dist/ballot-lens-f2/assets/main-abc.js",
        "styles": ["dist/ballot-lens-f2/assets/main-abc.css"],
    }


def test_f2_asset_loader_accepts_vite8_standalone_css_chunk(tmp_path: Path):
    dist = tmp_path / "dist/ballot-lens-f2"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (assets / "main-abc.js").write_text("export {};", encoding="utf-8")
    (assets / "style-def.css").write_text(":root{}", encoding="utf-8")
    (dist / "manifest.json").write_text(
        json.dumps(
            {
                "main.tsx": {
                    "file": "assets/main-abc.js",
                    "isEntry": True,
                },
                "style.css": {
                    "file": "assets/style-def.css",
                    "src": "style.css",
                },
            }
        ),
        encoding="utf-8",
    )
    assert load_ballot_lens_f2_assets(static_root=tmp_path) == {
        "script": "dist/ballot-lens-f2/assets/main-abc.js",
        "styles": ["dist/ballot-lens-f2/assets/style-def.css"],
    }


def test_f2_vite_config_intentionally_emits_single_css_bundle():
    config = _read(
        ROOT / "webapp/frontend/ballot-lens/vite.config.mts"
    )
    assert "cssCodeSplit: false" in config


def test_f2_asset_loader_rejects_escape(tmp_path: Path):
    dist = tmp_path / "dist/ballot-lens-f2"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (assets / "main-abc.js").write_text("export {};", encoding="utf-8")
    (dist / "manifest.json").write_text(
        json.dumps(
            {
                "main.tsx": {
                    "file": "assets/main-abc.js",
                    "isEntry": True,
                },
                "style.css": {
                    "file": "../escape.css",
                    "src": "style.css",
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escaped"):
        load_ballot_lens_f2_assets(static_root=tmp_path)


def test_docker_uses_isolated_f2_package():
    source = _read(DOCKERFILE)
    assert "FROM node:24.20.0-bookworm-slim AS frontend-builder" in source
    assert "WORKDIR /build/webapp/frontend/ballot-lens" in source
    assert "COPY webapp/frontend/ballot-lens/package.json" in source
    assert "RUN npm ci --no-audit --no-fund" in source
    assert "RUN npm run verify" in source
    assert (
        "COPY --from=frontend-builder "
        "/build/webapp/static/dist/ballot-lens-f2 "
        "/app/webapp/static/dist/ballot-lens-f2"
    ) in source


def test_f2_ci_is_separate_blocking_job():
    source = _read(WORKFLOW)
    assert "f2-frontend-foundation:" in source
    assert 'name: "Pre-Deploy QA: Ballot Lens F2 frontend (blocking)"' in source
    assert "node-version: '24.20.0'" in source
    assert "working-directory: webapp/frontend/ballot-lens" in source
    assert "run: npm ci --no-audit --no-fund" in source
    assert "run: npm run verify" in source
    assert "      - f2-frontend-foundation" in source
    # Existing legacy contract job remains explicitly non-blocking.
    assert "contract-checks-nonblocking:" in source
    assert "continue-on-error: true" in source

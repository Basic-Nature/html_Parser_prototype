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
F2_BOOTSTRAP = ROOT / "webapp/frontend/ballot-lens/contracts/bootstrap.ts"
F2_CHECKPOINTS = ROOT / "webapp/frontend/ballot-lens/contracts/checkpoints.ts"
F2_RUNTIME = ROOT / "webapp/frontend/ballot-lens/contracts/runtime.ts"
F2_RUN_MACHINE = ROOT / "webapp/frontend/ballot-lens/state/runMachine.ts"
F2_SELECTORS = ROOT / "webapp/frontend/ballot-lens/state/selectors.ts"
F2_RUN_TEST = ROOT / "webapp/frontend/ballot-lens/tests/runMachine.test.ts"
F2_APP_SHELL = ROOT / "webapp/frontend/ballot-lens/app/AppShell.tsx"
F2_HEADER = ROOT / "webapp/frontend/ballot-lens/components/common/HeaderBar.tsx"
F2_SOURCE_PANEL = ROOT / "webapp/frontend/ballot-lens/components/source/SourcePanel.tsx"
F2_WORKSPACE = ROOT / "webapp/frontend/ballot-lens/components/workspace/WorkspaceShell.tsx"
F2_CHECKPOINT_RAIL = ROOT / "webapp/frontend/ballot-lens/components/checkpoints/CheckpointRail.tsx"
F2_DIAGNOSTICS = ROOT / "webapp/frontend/ballot-lens/components/diagnostics/DiagnosticsDrawer.tsx"
F2_TOKENS = ROOT / "webapp/frontend/ballot-lens/styles/tokens.css"
F2_SHELL_CSS = ROOT / "webapp/frontend/ballot-lens/styles/shell.css"
F2_REGISTRY = ROOT / "webapp/frontend/ballot-lens/contracts/registry.ts"
F2_REGISTRY_API = ROOT / "webapp/frontend/ballot-lens/services/registryApi.ts"
F2_REGISTRY_BROWSER = ROOT / "webapp/frontend/ballot-lens/components/source/PublicRegistryBrowser.tsx"
F2_REGISTRY_TEST = ROOT / "webapp/frontend/ballot-lens/tests/registry.test.ts"
ROOT_PACKAGE = ROOT / "package.json"
DOCKERFILE = ROOT / "Dockerfile"
WORKFLOW = ROOT / ".github/workflows/main_ballotlens.yml"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def test_f2_variant_is_server_controlled_f2_default_and_legacy_override():
    source = _read(MAIN)
    start = source.index("def ballot_lens():")
    end = source.index("def ballot_lens_modern():", start)
    body = source[start:end]

    assert 'os.environ.get("BALLOT_LENS_UI_VARIANT", "f2")' in body
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
    source = "\n".join(
        _read(path)
        for path in (
            F2_MAIN,
            F2_APP,
            F2_APP_SHELL,
            F2_HEADER,
            F2_SOURCE_PANEL,
            F2_REGISTRY,
            F2_REGISTRY_BROWSER,
            F2_WORKSPACE,
            F2_CHECKPOINT_RAIL,
            F2_DIAGNOSTICS,
        )
    ).lower()
    assert "socket.emit" not in source
    assert "socket.on" not in source
    assert "direct_urls" not in source
    assert "target_url" not in source
    assert "executable_url" not in source
    assert "style={{" not in source

    registry_api = _read(F2_REGISTRY_API).lower()
    assert registry_api.count("fetch(") == 1
    assert "method: 'get'" in registry_api
    assert "credentials: 'same-origin'" in registry_api
    assert "socket." not in registry_api



def test_f2_isolated_package_does_not_modify_root_tooling_contract():
    root = json.loads(_read(ROOT_PACKAGE))
    f2 = json.loads(_read(F2_PACKAGE))

    assert "f2:typecheck" not in root["scripts"]
    assert "react" not in root.get("dependencies", {})
    assert f2["name"] == "@electionpulse/ballot-lens-f2"
    assert f2["engines"]["node"] == ">=24.0.0"
    assert f2["dependencies"]["react"] == "19.2.8"
    assert f2["dependencies"]["react-dom"] == "19.2.8"
    assert f2["dependencies"]["xstate"] == "5.32.6"
    assert f2["dependencies"]["socket.io-client"] == "4.7.5"
    assert f2["devDependencies"]["@vitejs/plugin-react"] == "6.1.1"
    assert f2["devDependencies"]["vite"] == "8.2.2"
    assert f2["devDependencies"]["typescript"] == "5.9.3"
    assert f2["devDependencies"]["@types/node"] == "24.10.0"
    assert f2["devDependencies"]["vitest"] == "4.1.11"
    assert f2["scripts"]["test:contracts"] == (
        "vitest run tests/runMachine.test.ts tests/registry.test.ts "
        "tests/publicRuntime.test.ts tests/socketAdapter.test.ts "
        "tests/publicRuntimeLifecycle.test.ts "
        "--environment node"
    )
    assert f2["scripts"]["verify"] == (
        "npm run typecheck && npm run test:contracts && npm run build"
    )


def test_f2_package_owns_node_types_for_vite_config():
    package = json.loads(_read(F2_PACKAGE))
    tsconfig = json.loads(
        _read(ROOT / "webapp/frontend/ballot-lens/tsconfig.json")
    )
    lock = json.loads(
        _read(ROOT / "webapp/frontend/ballot-lens/package-lock.json")
    )
    assert package["devDependencies"]["@types/node"] == "24.10.0"
    assert tsconfig["compilerOptions"]["types"] == ["vite/client", "node"]
    assert lock["packages"][""]["devDependencies"]["@types/node"] == "24.10.0"
    assert lock["packages"]["node_modules/@types/node"]["version"] == "24.10.0"



def test_f2b_phase_and_run_state_contracts_are_dormant_but_present():
    template = _read(F2_TEMPLATE)
    app = _read(F2_APP)
    bootstrap = _read(F2_BOOTSTRAP)
    checkpoints = _read(F2_CHECKPOINTS)
    runtime = _read(F2_RUNTIME)
    run_machine = _read(F2_RUN_MACHINE)
    selectors = _read(F2_SELECTORS)
    run_test = _read(F2_RUN_TEST)

    assert 'data-f2-phase="F2-E4"' in template
    assert "readonly phase: 'F2-E4'" in bootstrap
    assert "phase !== 'F2-E4'" in bootstrap
    assert "AppShell" in app

    assert "CHECKPOINT_DEFINITIONS" in checkpoints
    assert "source.resolve" in checkpoints
    assert "preview.publish" in checkpoints

    assert "registrySourceId?: string" in runtime
    assert "targetUrl" not in runtime
    assert "executableUrl" not in runtime
    assert "downloadAvailable: false" in runtime

    assert "SESSION_CORRELATED" in run_machine
    assert "ownsSession" in run_machine
    assert "incoming.sequence <= current.sequence" in run_machine
    assert "output.persistence === 'memory_only'" in run_machine
    assert "fromTransition" in run_machine

    assert "ownsActiveSession" in selectors
    assert "hasUnresolvedAction" in selectors

    assert "foreign session events" in run_test
    assert "memory-only output" in run_test
    assert "disconnect and reconnect" in run_test


def test_f2b_contract_layer_has_no_raw_socket_or_parser_command_authority():
    source = "\n".join(
        _read(path)
        for path in (
            F2_RUNTIME,
            F2_RUN_MACHINE,
            F2_SELECTORS,
            F2_RUN_TEST,
        )
    ).lower()

    assert "socket.emit" not in source
    assert "socket.on" not in source
    assert "fetch(" not in source
    assert "direct_urls" not in source
    assert "http://" not in source
    assert "https://" not in source


def test_f2c_app_shell_is_componentized_presentation_only_and_honest():
    app = _read(F2_APP)
    shell = _read(F2_APP_SHELL)
    header = _read(F2_HEADER)
    source = _read(F2_SOURCE_PANEL)
    workspace = _read(F2_WORKSPACE)
    checkpoints = _read(F2_CHECKPOINT_RAIL)
    diagnostics = _read(F2_DIAGNOSTICS)
    tokens = _read(F2_TOKENS)
    css = _read(F2_SHELL_CSS)
    template = _read(F2_TEMPLATE)

    assert "AppShell" in app
    assert "HeaderBar" in shell
    assert "SourcePanel" in shell
    assert "WorkspaceShell" in shell
    assert "CheckpointRail" in shell
    assert "DiagnosticsDrawer" in shell

    assert "F2-E3/E4 runtime" in header
    assert "Submit ready" in header
    assert "Approved public sources" in source
    assert "PublicRegistryBrowser" in source
    assert "No parser result yet" in workspace
    assert "does not fabricate preview rows or vote" in workspace
    assert "NULL preserved" in workspace
    assert "No precinct inference" in workspace
    assert "Provenance retained" in workspace
    assert "0 / 9" in checkpoints
    assert "Awaiting run" in checkpoints
    assert "Diagnostics &amp; audit trail" in diagnostics
    assert "No correlated runtime events yet" in diagnostics

    assert "--blf2-accent-soft" in tokens
    assert "--blf2-focus" in tokens
    assert "grid-template-columns:" in css
    assert "@media (max-width: 940px)" in css
    assert "@media (max-width: 640px)" in css
    assert ":focus-visible" in css
    assert "!important" not in css
    assert 'data-f2-phase="F2-E4"' in template
    assert "Ballot Lens — F2 Public Runtime Lifecycle" in template


def test_f2c_visual_shell_preserves_f2b_run_state_contracts():
    assert "SESSION_CORRELATED" in _read(F2_RUN_MACHINE)
    assert "incoming.sequence <= current.sequence" in _read(F2_RUN_MACHINE)
    assert "output.persistence === 'memory_only'" in _read(F2_RUN_MACHINE)
    assert "registrySourceId?: string" in _read(F2_RUNTIME)
    assert "downloadAvailable: false" in _read(F2_RUNTIME)
    assert "ownsActiveSession" in _read(F2_SELECTORS)
    assert "foreign session events" in _read(F2_RUN_TEST)
    assert "source.resolve" in _read(F2_CHECKPOINTS)



def test_f2d1_registry_discovery_preserves_public_registry_security_boundary():
    registry = _read(F2_REGISTRY)
    api = _read(F2_REGISTRY_API)
    browser = _read(F2_REGISTRY_BROWSER)
    source_panel = _read(F2_SOURCE_PANEL)
    package = json.loads(_read(F2_PACKAGE))

    for field in (
        "contest",
        "format",
        "registry_category",
        "registry_source_id",
        "scope",
        "state",
        "year",
    ):
        assert f"'{field}'" in registry

    assert "PUBLIC_REGISTRY_SOURCE_KEYS" in registry
    assert "hasExactSafeSourceKeys" in registry
    assert "hasExactSafeSourceKeys(entry)" in registry
    assert "Unsafe public registry source projection" in registry
    assert "registry_category: 'curated'" in registry
    assert "target_url" not in registry.lower()
    assert "executable_url" not in registry.lower()
    assert "direct_urls" not in registry.lower()

    assert "same-origin relative" in api
    assert "method: 'GET'" in api
    assert "credentials: 'same-origin'" in api
    assert "fetch(" in api
    assert "socket." not in api.lower()
    assert "execution_source_id" not in api
    assert "execution_enabled" not in api

    assert "getRegistryFacetOptions" in browser
    assert "Search approved sources" in browser
    assert "Scope / county" in browser
    assert "Select the one execution-authorized source" in browser
    assert "Selection is app-owned" in browser
    assert "PublicRegistryBrowser" in source_panel

    assert package["scripts"]["test:contracts"] == (
        "vitest run tests/runMachine.test.ts tests/registry.test.ts "
        "tests/publicRuntime.test.ts tests/socketAdapter.test.ts "
        "tests/publicRuntimeLifecycle.test.ts "
        "--environment node"
    )



def test_f2e1_registry_execution_metadata_adds_no_parser_command_authority():
    registry = _read(F2_REGISTRY)
    non_registry_command_surfaces = "\n".join(
        _read(path)
        for path in (
            F2_REGISTRY_API,
            F2_REGISTRY_BROWSER,
            F2_SOURCE_PANEL,
        )
    )

    # E1 deliberately types only SAFE ROOT registry execution authority.
    assert "PUBLIC_REGISTRY_ROOT_KEYS" in registry
    assert "execution_enabled" in registry
    assert "execution_source_id" in registry

    # Individual source records remain exact seven-field projections.
    assert "PUBLIC_REGISTRY_SOURCE_KEYS" in registry
    assert "hasExactSafeSourceKeys" in registry
    assert "hasExactSafeSourceKeys(entry)" in registry

    # Root execution metadata is not a parser-command surface.
    assert "execution_enabled" not in non_registry_command_surfaces
    assert "execution_source_id" not in non_registry_command_surfaces

    combined = registry + "\n" + non_registry_command_surfaces
    assert "public_registry_runtime_started" not in combined
    assert "public_registry_result" not in combined
    assert "socket.emit" not in combined
    assert "btnRunParser" not in combined
    assert "btnRunParser2" not in combined

    for forbidden in (
        "direct_urls",
        "file_source",
        "warehouse_override",
        "executable_url",
    ):
        assert forbidden not in non_registry_command_surfaces


def test_f2d1_registry_facets_are_availability_aware():
    registry = _read(F2_REGISTRY)
    browser = _read(F2_REGISTRY_BROWSER)
    tests = _read(F2_REGISTRY_TEST)

    assert "available: count > 0" in registry
    assert "disabled={!option.available" in browser
    assert "keeps unavailable facet values visible with zero counts" in tests

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

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
F2 = ROOT / "webapp/frontend/ballot-lens"
PACKAGE = F2 / "package.json"
REGISTRY = F2 / "contracts/registry.ts"
REGISTRY_API = F2 / "services/registryApi.ts"
PUBLIC_RUNTIME = F2 / "contracts/publicRuntime.ts"
SOCKET_CLIENT = F2 / "services/socketClient.ts"
SOCKET_ADAPTER = F2 / "services/socketAdapter.ts"
TEMPLATE = ROOT / "webapp/templates/ballot_lens_f2.html"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def test_f2e1_socket_dependency_is_exact_and_isolated():
    package = json.loads(_read(PACKAGE))
    assert package["dependencies"]["socket.io-client"] == "4.7.5"
    assert "socket.io-client" in _read(PACKAGE)


def test_f2e1_preserves_exact_seven_field_source_projection():
    source = _read(REGISTRY)
    for token in (
        "'contest',",
        "'format',",
        "'registry_category',",
        "'registry_source_id',",
        "'scope',",
        "'state',",
        "'year',",
    ):
        assert token in source
    assert "PUBLIC_REGISTRY_ROOT_KEYS" in source
    assert "'execution_enabled'," in source
    assert "'execution_source_id'," in source
    assert "hasExactSafeSourceKeys" in source
    assert "hasExactSafeSourceKeys(entry)" in source
    assert "Unsafe public registry root projection" in source


def test_f2e1_registry_api_remains_get_only_and_same_origin():
    source = _read(REGISTRY_API)
    assert "method: 'GET'" in source
    assert "credentials: 'same-origin'" in source
    assert "loadPublicRegistryEnvelope" in source
    for forbidden in (
        "POST",
        "direct_urls",
        "file_source",
        "warehouse_override",
        "executable_url",
    ):
        assert forbidden not in source


def test_f2e1_public_runtime_contract_preserves_no_write_semantics():
    source = _read(PUBLIC_RUNTIME)
    assert "ballot_lens_public_memory_preview_v1" in source
    assert "ballot_lens_public_runtime_result_v1" in source
    assert "MEMORY_PREVIEW_ONLY" in source
    assert "download_available: false" in source
    assert "persistent_output: false" in source
    assert "JsonValue" in source


def test_f2e1_socket_transport_is_dormant_and_has_no_emit_authority():
    client = _read(SOCKET_CLIENT)
    adapter = _read(SOCKET_ADAPTER)
    assert "socket.io-client" in client
    assert "autoConnect: false" in client
    assert "DormantBallotLensSocket" in client
    assert "emit(" not in client
    assert ".emit" not in client
    assert "emit(" not in adapter
    assert ".emit" not in adapter
    for forbidden in (
        "direct_urls",
        "file_source",
        "warehouse_override",
        "manual_upload",
    ):
        assert forbidden not in client
        assert forbidden not in adapter


def test_f2e1_template_reuses_server_owned_socket_config_without_legacy_global():
    template = _read(TEMPLATE)
    assert "data-socketio-config=" in template
    assert "socketio_client_config | tojson | forceescape" in template
    assert "socket.io-4.7.5.min.js" not in template
    assert "ballot_lens_init.js" not in template


def test_f2e1_has_no_public_run_submit_wiring():
    combined = "\n".join(
        _read(path)
        for path in (
            REGISTRY,
            REGISTRY_API,
            PUBLIC_RUNTIME,
            SOCKET_CLIENT,
            SOCKET_ADAPTER,
        )
    )
    assert "SUBMIT_REQUESTED" not in combined
    assert "SUBMISSION_ACCEPTED" not in combined
    assert "socket.emit" not in combined
    assert "registry_source_id: source.registry_source_id" not in combined

def test_f2e1_root_execution_metadata_does_not_create_command_authority():
    registry = _read(REGISTRY)
    api = _read(REGISTRY_API)

    assert "execution_enabled" in registry
    assert "execution_source_id" in registry
    assert "PUBLIC_REGISTRY_ROOT_KEYS" in registry

    for forbidden in (
        "execution_source_id",
        "execution_enabled",
        "socket.emit",
        "direct_urls",
        "file_source",
        "warehouse_override",
        "executable_url",
    ):
        assert forbidden not in api


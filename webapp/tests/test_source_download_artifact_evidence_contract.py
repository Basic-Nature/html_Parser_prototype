from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from webapp.parser.services.source_download_artifact_evidence import (
    CONTRACT,
    SourceDownloadArtifactObservationError,
    observe_source_download_artifact_if_requested,
)

ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_UTILS = ROOT / "webapp/parser/utils/download_utils.py"
FORMAT_ROUTER = ROOT / "webapp/parser/utils/format_router.py"
PUBLIC_POLICY = ROOT / "webapp/parser/services/public_ballot_lens_policy.py"
PUBLIC_EXECUTION = ROOT / "webapp/parser/services/public_ballot_lens_execution.py"

FORBIDDEN_OBSERVATION_KEYS = {
    "payload_bytes",
    "raw_requested_url",
    "raw_effective_request_url",
    "raw_final_response_url",
    "raw_redirect_chain",
    "request_headers",
    "response_headers",
    "cookies",
    "raw_filename",
    "raw_local_path",
    "raw_error_text",
}


class _ExplodesOnTouch:
    def __str__(self):
        raise AssertionError("dormant observer touched an input")

    def __fspath__(self):
        raise AssertionError("dormant observer touched a path")


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name} in {path}")


def _default_for_arg(fn: ast.FunctionDef, arg_name: str):
    positional = [*fn.args.posonlyargs, *fn.args.args]
    names = [arg.arg for arg in positional]
    index = names.index(arg_name)
    defaults_start = len(positional) - len(fn.args.defaults)
    if index < defaults_start:
        raise AssertionError(f"{arg_name} has no default")
    return ast.literal_eval(fn.args.defaults[index - defaults_start])


def test_dormant_none_returns_before_touching_inputs() -> None:
    bomb = _ExplodesOnTouch()
    result = observe_source_download_artifact_if_requested(
        emit_func=None,
        persisted_path=bomb,
        requested_url=bomb,  # type: ignore[arg-type]
        effective_request_url=bomb,  # type: ignore[arg-type]
        final_response_url=bomb,  # type: ignore[arg-type]
        transport=bomb,  # type: ignore[arg-type]
    )
    assert result is None


def test_active_observation_is_hash_only(tmp_path: Path) -> None:
    payload = b"source-download-payload\n"
    artifact = tmp_path / "payload.bin"
    artifact.write_bytes(payload)
    emitted: list[dict[str, object]] = []

    observation = observe_source_download_artifact_if_requested(
        emit_func=emitted.append,
        persisted_path=artifact,
        requested_url="https://example.test/requested?secret=one",
        effective_request_url="https://example.test/effective?secret=two",
        final_response_url="https://example.test/final?secret=three",
        transport="playwright_api_request",
    )
    assert observation is not None
    assert emitted == [observation]
    assert observation["contract"] == CONTRACT
    assert observation["payload_sha256"] == hashlib.sha256(payload).hexdigest()
    assert observation["payload_size"] == len(payload)
    assert not (FORBIDDEN_OBSERVATION_KEYS & set(observation))
    serialized = json.dumps(observation, sort_keys=True)
    assert "secret=one" not in serialized
    assert "secret=two" not in serialized
    assert "secret=three" not in serialized
    assert str(artifact) not in serialized


def test_precomputed_identity_avoids_path_read() -> None:
    emitted: list[dict[str, object]] = []
    observation = observe_source_download_artifact_if_requested(
        emit_func=emitted.append,
        persisted_path=_ExplodesOnTouch(),
        requested_url="https://example.test/a",
        effective_request_url="https://example.test/b",
        final_response_url=None,
        transport="requests_stream",
        precomputed_payload_sha256="a" * 64,
        precomputed_payload_size=123,
    )
    assert observation is not None
    assert observation["payload_sha256"] == "a" * 64
    assert observation["payload_size"] == 123


def test_callback_failure_is_wrapped() -> None:
    def explode(_observation):
        raise TypeError("raw callback secret")

    with pytest.raises(
        SourceDownloadArtifactObservationError,
        match="observation callback failed",
    ) as exc_info:
        observe_source_download_artifact_if_requested(
            emit_func=explode,
            persisted_path=_ExplodesOnTouch(),
            requested_url="https://example.test/a",
            effective_request_url="https://example.test/b",
            final_response_url=None,
            transport="requests_stream",
            precomputed_payload_sha256="b" * 64,
            precomputed_payload_size=7,
        )
    assert "raw callback secret" not in str(exc_info.value)


def test_source_seams_are_explicit_and_default_none() -> None:
    download_fn = _function(DOWNLOAD_UTILS, "download_file")
    router_fn = _function(FORMAT_ROUTER, "prompt_and_handle_download")
    assert _default_for_arg(download_fn, "source_download_observation_emit_func") is None
    assert _default_for_arg(router_fn, "source_download_observation_emit_func") is None

    download_text = DOWNLOAD_UTILS.read_text(encoding="utf-8")
    router_text = FORMAT_ROUTER.read_text(encoding="utf-8")
    assert "precomputed_payload_sha256=filehash" in download_text
    assert "precomputed_payload_size=total" in download_text
    assert 'source_download_transport = "playwright_api_request"' in router_text
    assert 'source_download_transport = "raw_requests_temp_last_resort"' in router_text
    assert "transport=source_download_transport" in router_text


def test_public_runtime_download_boundaries_remain_fail_closed() -> None:
    policy = PUBLIC_POLICY.read_text(encoding="utf-8")
    execution = PUBLIC_EXECUTION.read_text(encoding="utf-8")
    assert "public_file_download: bool = False" in policy
    assert "shared_download_manifest_as_public_authority: bool = False" in policy
    assert "caller_supplied_url: bool = False" in policy
    assert "download_manifest_write: bool = False" in execution
    assert "persistent_output_write: bool = False" in execution

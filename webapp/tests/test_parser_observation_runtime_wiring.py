from __future__ import annotations

import ast
from pathlib import Path

import pytest

import webapp.parser.socket_ballot_lens_orchestration as orchestration


REPO_ROOT = Path(__file__).resolve().parents[2]
SOCKET_PATH = REPO_ROOT / "webapp/parser/socket_ballot_lens_orchestration.py"
WEB_PATH = REPO_ROOT / "webapp/parser/web_pipeline.py"
HTML_PATH = REPO_ROOT / "webapp/parser/html_election_parser.py"
SHARED_PATH = REPO_ROOT / "webapp/parser/utils/shared_logic.py"


class FakeSocketIO:
    def __init__(self):
        self.calls = []

    def emit(self, event, payload, room=None):
        self.calls.append((event, payload, room))


def _payload():
    return {
        "contract": "parser_observation_bundle_v1",
        "authority": {
            "inspection": "noncanonical_parser_evidence",
            "canonical": False,
        },
        "source_stage": "interpreted",
        "pipeline_inspection": {
            "contract": "pipeline_inspection_v1",
            "authority": {
                "inspection": "noncanonical_parser_evidence",
                "canonical": False,
            },
        },
        "election_structure": {
            "contract": "election_structure_observation_v1",
            "authority": {
                "inspection": "noncanonical_parser_evidence",
                "canonical": False,
            },
        },
        "raw_rows_included": False,
        "raw_headers_included": False,
        "automatic_timestamp": False,
    }


def _tree(path: Path):
    source = path.read_text(encoding="utf-8-sig")
    return source, ast.parse(source, filename=str(path))


def _fn(tree: ast.AST, name: str):
    rows = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(rows) == 1
    return rows[0]


def _nested_fn(parent: ast.FunctionDef, name: str):
    rows = [
        node for node in ast.walk(parent)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(rows) == 1
    return rows[0]


def _call_name(node: ast.AST):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _call_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    if isinstance(node, ast.Subscript):
        try:
            return ast.unparse(node)
        except Exception:
            return ""
    return ""


def _calls(parent: ast.AST):
    return [node for node in ast.walk(parent) if isinstance(node, ast.Call)]


def _keyword(call: ast.Call, name: str):
    rows = [kw for kw in call.keywords if kw.arg == name]
    assert len(rows) == 1
    return rows[0]


def test_parser_observation_socket_emitter_is_session_scoped_and_hides_principal():
    socketio = FakeSocketIO()
    emit_observation = orchestration._make_parser_observation_emitter(
        "session-obs-1",
        "principal-secret-1",
        {"socketio": socketio},
    )

    payload = _payload()
    emit_observation(payload)

    assert len(socketio.calls) == 1
    event, envelope, room = socketio.calls[0]
    assert event == "parser_observation"
    assert room == "session-obs-1"
    assert envelope["contract"] == "parser_observation_socket_v1"
    assert envelope["authority"] == {
        "canonical": False,
        "transport": "same_run_socket",
    }
    assert envelope["session_id"] == "session-obs-1"
    assert envelope["observation"] is payload
    assert "principal-secret-1" not in repr(envelope)


def test_parser_observation_socket_emitter_rejects_missing_ownership():
    for session_id, principal in (
        ("", "principal"),
        ("session", None),
        ("session", ""),
    ):
        with pytest.raises(ValueError):
            orchestration._make_parser_observation_emitter(
                session_id,
                principal,
                {"socketio": FakeSocketIO()},
            )


def test_parser_observation_socket_emitter_fails_closed_on_invalid_payloads():
    socketio = FakeSocketIO()
    emit_observation = orchestration._make_parser_observation_emitter(
        "session-obs-2",
        "principal-obs-2",
        {"socketio": socketio},
    )

    invalid_payloads = []

    wrong_contract = _payload()
    wrong_contract["contract"] = "wrong"
    invalid_payloads.append(wrong_contract)

    canonical = _payload()
    canonical["authority"]["canonical"] = True
    invalid_payloads.append(canonical)

    raw_rows = _payload()
    raw_rows["raw_rows_included"] = True
    invalid_payloads.append(raw_rows)

    rows_key = _payload()
    rows_key["rows"] = [{"Precinct": "P-1"}]
    invalid_payloads.append(rows_key)

    raw_headers = _payload()
    raw_headers["raw_headers_included"] = True
    invalid_payloads.append(raw_headers)

    headers_key = _payload()
    headers_key["headers"] = ["Precinct"]
    invalid_payloads.append(headers_key)

    timestamp = _payload()
    timestamp["automatic_timestamp"] = True
    invalid_payloads.append(timestamp)

    for payload in invalid_payloads:
        with pytest.raises((TypeError, ValueError)):
            emit_observation(payload)

    assert socketio.calls == []


def test_trusted_worker_forwards_private_parser_observation_emitter():
    _, tree = _tree(SOCKET_PATH)
    worker = _nested_fn(_fn(tree, "_start_pipeline_worker"), "worker_wrapper")
    process_calls = [
        call for call in _calls(worker)
        if "process_urls_for_web" in _call_name(call.func)
    ]
    assert len(process_calls) == 1
    kw = _keyword(process_calls[0], "parser_observation_emit_func")
    assert isinstance(kw.value, ast.Call)
    assert _call_name(kw.value.func).split(".")[-1] == "_make_parser_observation_emitter"
    assert [ast.unparse(arg) for arg in kw.value.args] == [
        "session_id",
        "principal",
        "h",
    ]


def test_public_worker_explicitly_disables_parser_observation_emitter():
    _, tree = _tree(SOCKET_PATH)
    worker = _nested_fn(_fn(tree, "_start_public_registry_runtime"), "worker_wrapper")
    process_calls = [
        call for call in _calls(worker)
        if "process_urls_for_web" in _call_name(call.func)
    ]
    assert len(process_calls) == 1
    kw = _keyword(process_calls[0], "parser_observation_emit_func")
    assert isinstance(kw.value, ast.Constant)
    assert kw.value.value is None


def test_public_web_pipeline_forbids_private_parser_observation_callback():
    _, tree = _tree(WEB_PATH)
    fn = _fn(tree, "_process_public_registry_url_for_web")
    text = ast.unparse(fn)
    assert "'parser_observation_emit_func'" in text
    assert "Public registry runtime forbids" in text


def test_html_main_explicitly_hands_parser_observation_callback_to_format_override():
    _, tree = _tree(HTML_PATH)
    fn = _fn(tree, "main")
    calls = [
        call for call in _calls(fn)
        if _call_name(call.func).split(".")[-1] == "process_format_override"
    ]
    assert len(calls) == 1
    kw = _keyword(calls[0], "parser_observation_emit_func")
    assert isinstance(kw.value, ast.Call)
    assert _call_name(kw.value.func) == "kwargs.get"
    assert len(kw.value.args) == 1
    assert isinstance(kw.value.args[0], ast.Constant)
    assert kw.value.args[0].value == "parser_observation_emit_func"


def test_format_override_existing_generic_kwargs_conduit_remains_intact():
    _, tree = _tree(HTML_PATH)
    fn = _fn(tree, "process_format_override")
    assert fn.args.kwarg is not None
    assert fn.args.kwarg.arg == "kwargs"
    calls = [
        call for call in _calls(fn)
        if _call_name(call.func).split(".")[-1] == "safe_parse"
    ]
    assert len(calls) == 1
    assert any(
        kw.arg is None
        and isinstance(kw.value, ast.Name)
        and kw.value.id == "kwargs"
        for kw in calls[0].keywords
    )


def test_safe_parse_existing_generic_kwargs_conduit_remains_intact():
    _, tree = _tree(SHARED_PATH)
    fn = _fn(tree, "safe_parse")
    text = ast.unparse(fn)
    assert "call_kwargs = dict(kwargs)" in text
    calls = [
        call for call in _calls(fn)
        if _call_name(call.func) == "parse_method"
    ]
    assert len(calls) == 1
    assert any(
        kw.arg is None
        and isinstance(kw.value, ast.Name)
        and kw.value.id == "call_kwargs"
        for kw in calls[0].keywords
    )


def test_projection_does_not_expand_into_public_or_canonical_surfaces():
    socket_source = SOCKET_PATH.read_text(encoding="utf-8-sig")
    web_source = WEB_PATH.read_text(encoding="utf-8-sig")

    assert '"parser_observation"' in socket_source
    assert '"parser_observation_emit_func"' in web_source

    for rel in (
        "webapp/parser/handlers/formats/csv_handler.py",
        "webapp/parser/handlers/formats/xlsx_handler.py",
        "webapp/parser/handlers/formats/pdf_handler.py",
        "webapp/parser/navigator/dom_snapshot.py",
        "webapp/parser/services/parser_observation_callback.py",
        "webapp/parser/services/parser_observation_bundle.py",
        "webapp/parser/utils/output_utils.py",
    ):
        path = REPO_ROOT / rel
        assert path.is_file()

    public_worker = _fn(_tree(SOCKET_PATH)[1], "_start_public_registry_runtime")
    public_text = ast.unparse(public_worker)
    assert "parser_observation_emit_func=None" in public_text

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

ORCH_PATH = (
    REPO_ROOT
    / "webapp"
    / "parser"
    / "socket_ballot_lens_orchestration.py"
)
WEB_PIPELINE_PATH = (
    REPO_ROOT
    / "webapp"
    / "parser"
    / "web_pipeline.py"
)
HTML_PATH = (
    REPO_ROOT
    / "webapp"
    / "parser"
    / "html_election_parser.py"
)
SHARED_LOGIC_PATH = (
    REPO_ROOT
    / "webapp"
    / "parser"
    / "utils"
    / "shared_logic.py"
)
MISC_PATH = (
    REPO_ROOT
    / "webapp"
    / "parser"
    / "utils"
    / "misc_utils.py"
)
PDF_PATH = (
    REPO_ROOT
    / "webapp"
    / "parser"
    / "handlers"
    / "formats"
    / "pdf_handler.py"
)


def _tree(path: Path):
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(path))


def _fn(tree: ast.AST, name: str):
    rows = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == name
    ]
    assert len(rows) == 1
    return rows[0]


def _nested_fn(parent: ast.FunctionDef, name: str):
    rows = [
        node
        for node in ast.walk(parent)
        if isinstance(node, ast.FunctionDef)
        and node.name == name
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


def _calls(fn: ast.FunctionDef):
    return [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
    ]


def _keyword(call: ast.Call, name: str):
    rows = [
        kw
        for kw in call.keywords
        if kw.arg == name
    ]
    assert len(rows) == 1
    return rows[0]


def test_worker_producer_is_inside_worker_wrapper():
    _, tree = _tree(ORCH_PATH)
    outer = _fn(tree, "_start_pipeline_worker")
    worker = _nested_fn(outer, "worker_wrapper")
    text = ast.unparse(worker)

    assert "artifact_identity = None" in text
    assert "requested_source == 'uploads' and force_parse_input_file" in text

    outer_text = ast.unparse(outer)
    prefix = outer_text.split("def worker_wrapper", 1)[0]
    assert "ArtifactIdentityHandoff" not in prefix
    assert "file_hash(" not in prefix


def test_worker_revalidates_exact_upload_path_before_hash():
    _, tree = _tree(ORCH_PATH)
    worker = _nested_fn(
        _fn(tree, "_start_pipeline_worker"),
        "worker_wrapper",
    )
    text = ast.unparse(worker)

    assert "h['os'].path.abspath(h['uploads_dir'])" in text
    assert (
        "is_path_within_root(artifact_path, artifact_uploads_dir, "
        "path_module=h['os'].path)"
        in text
    )
    assert "h['os'].path.isfile(artifact_path)" in text

    containment_index = text.index("is_path_within_root(")
    hash_index = text.index("file_hash(")
    assert containment_index < hash_index


def test_worker_reuses_misc_file_hash_with_explicit_sha256():
    _, tree = _tree(ORCH_PATH)
    worker = _nested_fn(
        _fn(tree, "_start_pipeline_worker"),
        "worker_wrapper",
    )

    imports = [
        node
        for node in ast.walk(worker)
        if isinstance(node, ast.ImportFrom)
    ]

    assert any(
        node.level == 1
        and node.module == "utils.misc_utils"
        and any(alias.name == "file_hash" for alias in node.names)
        for node in imports
    )

    calls = [
        node
        for node in _calls(worker)
        if _call_name(node.func).split(".")[-1] == "file_hash"
    ]
    assert len(calls) == 1
    call = calls[0]

    assert len(call.args) == 1
    assert isinstance(call.args[0], ast.Name)
    assert call.args[0].id == "artifact_path"

    algo = _keyword(call, "algo")
    assert isinstance(algo.value, ast.Constant)
    assert algo.value.value == "sha256"


def test_worker_constructs_validated_handoff_only_from_digest():
    _, tree = _tree(ORCH_PATH)
    worker = _nested_fn(
        _fn(tree, "_start_pipeline_worker"),
        "worker_wrapper",
    )

    imports = [
        node
        for node in ast.walk(worker)
        if isinstance(node, ast.ImportFrom)
    ]

    assert any(
        node.level == 1
        and node.module == "contracts.artifact_identity"
        and any(
            alias.name == "ArtifactIdentityHandoff"
            for alias in node.names
        )
        for node in imports
    )

    calls = [
        node
        for node in _calls(worker)
        if _call_name(node.func).split(".")[-1]
        == "ArtifactIdentityHandoff"
    ]
    assert len(calls) == 1
    call = calls[0]

    kw = _keyword(call, "document_sha256")
    assert isinstance(kw.value, ast.Name)
    assert kw.value.id == "document_sha256"


def test_worker_identity_production_fails_closed_before_dispatch():
    _, tree = _tree(ORCH_PATH)
    worker = _nested_fn(
        _fn(tree, "_start_pipeline_worker"),
        "worker_wrapper",
    )
    text = ast.unparse(worker)

    required = [
        (
            "if not is_path_within_root(artifact_path, artifact_uploads_dir, "
            "path_module=h['os'].path):"
        ),
        (
            "raise RuntimeError("
            "'Manual upload escaped uploads root during worker revalidation.'"
            ")"
        ),
        "if not h['os'].path.isfile(artifact_path):",
        (
            "raise RuntimeError("
            "'Manual upload is no longer a file during worker revalidation.'"
            ")"
        ),
        "if not document_sha256:",
        (
            "raise RuntimeError("
            "'Manual upload SHA-256 identity computation failed.'"
            ")"
        ),
    ]

    dispatch_index = text.index("h['process_urls_for_web'](")
    for fragment in required:
        assert fragment in text
        assert text.index(fragment) < dispatch_index


def test_worker_forwards_identity_to_web_pipeline():
    _, tree = _tree(ORCH_PATH)
    worker = _nested_fn(
        _fn(tree, "_start_pipeline_worker"),
        "worker_wrapper",
    )

    calls = [
        node
        for node in _calls(worker)
        if "process_urls_for_web" in _call_name(node.func)
    ]
    assert len(calls) == 1
    call = calls[0]

    kw = _keyword(call, "artifact_identity")
    assert isinstance(kw.value, ast.Name)
    assert kw.value.id == "artifact_identity"


def test_identity_is_not_stored_in_run_cfg():
    _, tree = _tree(ORCH_PATH)
    prepare = _fn(tree, "_prepare_run_inputs")
    worker = _fn(tree, "_start_pipeline_worker")

    prepare_text = ast.unparse(prepare)
    worker_text = ast.unparse(worker)

    assert "'artifact_identity'" not in prepare_text
    assert '"artifact_identity"' not in prepare_text
    assert "run_cfg['artifact_identity']" not in worker_text
    assert 'run_cfg["artifact_identity"]' not in worker_text


def test_existing_web_pipeline_kwargs_conduit_needs_no_mutation():
    _, tree = _tree(WEB_PIPELINE_PATH)
    fn = _fn(tree, "process_urls_for_web")

    assignments = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "main_kwargs"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    value = assignments[0].value
    assert isinstance(value, ast.Call)
    assert _call_name(value.func) == "dict"
    assert any(kw.arg is None for kw in value.keywords)

    main_calls = [
        node
        for node in _calls(fn)
        if _call_name(node.func).split(".")[-1] == "main"
    ]
    assert len(main_calls) >= 1
    assert all(
        any(
            kw.arg is None
            and isinstance(kw.value, ast.Name)
            and kw.value.id == "main_kwargs"
            for kw in call.keywords
        )
        for call in main_calls
    )


def test_main_explicitly_hands_identity_to_format_override():
    _, tree = _tree(HTML_PATH)
    fn = _fn(tree, "main")

    calls = [
        node
        for node in _calls(fn)
        if _call_name(node.func).split(".")[-1]
        == "process_format_override"
    ]
    assert len(calls) == 1
    call = calls[0]

    kw = _keyword(call, "artifact_identity")
    assert isinstance(kw.value, ast.Call)
    assert _call_name(kw.value.func) == "kwargs.get"
    assert len(kw.value.args) == 1
    assert isinstance(kw.value.args[0], ast.Constant)
    assert kw.value.args[0].value == "artifact_identity"


def test_format_override_existing_kwargs_reaches_safe_parse():
    _, tree = _tree(HTML_PATH)
    fn = _fn(tree, "process_format_override")

    assert fn.args.kwarg is not None
    assert fn.args.kwarg.arg == "kwargs"

    calls = [
        node
        for node in _calls(fn)
        if _call_name(node.func).split(".")[-1] == "safe_parse"
    ]
    assert len(calls) == 1
    call = calls[0]

    assert any(
        kw.arg is None
        and isinstance(kw.value, ast.Name)
        and kw.value.id == "kwargs"
        for kw in call.keywords
    )


def test_safe_parse_existing_handler_dispatch_preserves_kwargs():
    _, tree = _tree(SHARED_LOGIC_PATH)
    fn = _fn(tree, "safe_parse")
    text = ast.unparse(fn)

    assert "call_kwargs = dict(kwargs)" in text

    calls = [
        node
        for node in _calls(fn)
        if _call_name(node.func) == "parse_method"
    ]
    assert len(calls) == 1
    assert any(
        kw.arg is None
        and isinstance(kw.value, ast.Name)
        and kw.value.id == "call_kwargs"
        for kw in calls[0].keywords
    )


def test_pdf_optional_identity_conduit_remains_exact():
    _, tree = _tree(PDF_PATH)
    wrapper = _fn(tree, "parse")
    inner = _fn(tree, "parse_pdf_election_results")

    wrapper_kwonly = [
        arg.arg
        for arg in wrapper.args.kwonlyargs
    ]
    inner_kwonly = [
        arg.arg
        for arg in inner.args.kwonlyargs
    ]

    assert wrapper_kwonly.count("artifact_identity") == 1
    assert inner_kwonly.count("artifact_identity") == 1

    calls = [
        node
        for node in _calls(wrapper)
        if _call_name(node.func).split(".")[-1]
        == "parse_pdf_election_results"
    ]
    assert len(calls) == 1

    kw = _keyword(calls[0], "artifact_identity")
    assert isinstance(kw.value, ast.Name)
    assert kw.value.id == "artifact_identity"


def test_misc_file_hash_is_streaming_binary_sha256_capable():
    _, tree = _tree(MISC_PATH)
    fn = _fn(tree, "file_hash")
    text = ast.unparse(fn)

    assert "hashlib.new(algo)" in text
    assert "open(filepath, 'rb')" in text
    assert "for chunk in iter(" in text
    assert "h.update(chunk)" in text
    assert "return h.hexdigest()" in text

    defaults = list(fn.args.defaults)
    positional = list(fn.args.args)
    default_map = {
        arg.arg: default
        for arg, default in zip(
            positional[-len(defaults):],
            defaults,
        )
    }

    assert isinstance(default_map["algo"], ast.Constant)
    assert default_map["algo"].value == "sha256"


def test_hashing_is_upstream_not_parser_side():
    orch_text = ORCH_PATH.read_text(encoding="utf-8")
    html_text = HTML_PATH.read_text(encoding="utf-8")
    pdf_text = PDF_PATH.read_text(encoding="utf-8")

    assert orch_text.count("file_hash(") == 1
    assert "file_hash(" not in html_text
    assert "file_hash(" not in pdf_text

    assert "ArtifactIdentityHandoff(" in orch_text
    assert "ArtifactIdentityHandoff(" not in html_text

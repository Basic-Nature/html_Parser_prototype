from __future__ import annotations

import ast
import importlib.util
import ntpath
import posixpath
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = REPO_ROOT / "webapp" / "parser" / "utils" / "path_safety.py"
ORCH_PATH = (
    REPO_ROOT
    / "webapp"
    / "parser"
    / "socket_ballot_lens_orchestration.py"
)
HTML_PATH = (
    REPO_ROOT
    / "webapp"
    / "parser"
    / "html_election_parser.py"
)


def _load_helper():
    spec = importlib.util.spec_from_file_location(
        "_electionpulse_path_safety_test",
        HELPER_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_posix_child_is_allowed():
    helper = _load_helper()
    assert helper.is_path_within_root(
        "/srv/app/uploads/a.pdf",
        "/srv/app/uploads",
        path_module=posixpath,
    )


def test_posix_sibling_prefix_is_rejected():
    helper = _load_helper()
    assert not helper.is_path_within_root(
        "/srv/app/uploads_evil/a.pdf",
        "/srv/app/uploads",
        path_module=posixpath,
    )


def test_posix_parent_traversal_is_rejected():
    helper = _load_helper()
    assert not helper.is_path_within_root(
        "/srv/app/uploads/../secret/a.pdf",
        "/srv/app/uploads",
        path_module=posixpath,
    )


def test_windows_child_is_allowed():
    helper = _load_helper()
    assert helper.is_path_within_root(
        r"C:\app\uploads\a.pdf",
        r"C:\app\uploads",
        path_module=ntpath,
    )


def test_windows_sibling_prefix_is_rejected():
    helper = _load_helper()
    assert not helper.is_path_within_root(
        r"C:\app\uploads_evil\a.pdf",
        r"C:\app\uploads",
        path_module=ntpath,
    )


def test_windows_parent_traversal_is_rejected():
    helper = _load_helper()
    assert not helper.is_path_within_root(
        r"C:\app\uploads\..\secret\a.pdf",
        r"C:\app\uploads",
        path_module=ntpath,
    )


def test_windows_cross_drive_fails_closed():
    helper = _load_helper()
    assert not helper.is_path_within_root(
        r"D:\secret\a.pdf",
        r"C:\app\uploads",
        path_module=ntpath,
    )


def test_helper_contract_is_realpath_commonpath_fail_closed():
    tree = ast.parse(
        HELPER_PATH.read_text(encoding="utf-8"),
        filename=str(HELPER_PATH),
    )
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "is_path_within_root"
    )
    text = ast.unparse(fn)
    assert ".realpath(" in text
    assert ".commonpath(" in text
    assert ".normcase(" in text
    assert "ValueError" in text
    assert "TypeError" in text
    assert "OSError" in text


def test_socket_guard_uses_shared_path_boundary():
    text = ORCH_PATH.read_text(encoding="utf-8")
    assert "candidate_path.startswith(abs_uploads_dir)" not in text
    assert (
        "is_path_within_root("
        "candidate_path, abs_uploads_dir, "
        "path_module=h['os'].path)"
        in text
    )


def test_format_override_guard_uses_shared_path_boundary():
    text = HTML_PATH.read_text(encoding="utf-8")
    assert "forced_path.startswith(input_folder)" not in text
    assert (
        "is_path_within_root("
        "forced_path, input_folder, "
        "path_module=os.path)"
        in text
    )


def test_path_is_not_artifact_identity():
    text = HELPER_PATH.read_text(encoding="utf-8")
    assert "ArtifactIdentityHandoff" not in text
    assert "sha256" not in text.lower()
    assert "hashlib" not in text

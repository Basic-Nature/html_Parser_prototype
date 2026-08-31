from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WEBAPP_PATH = REPO_ROOT / "webapp" / "Smart_Elections_Parser_Webapp.py"


def _helper():
    tree = ast.parse(
        WEBAPP_PATH.read_text(encoding="utf-8"),
        filename=str(WEBAPP_PATH),
    )
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_save_uploaded_file"
    ]
    assert len(matches) == 1
    return matches[0]


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _call_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return ""


def test_exclusive_xb_open_is_preserved():
    helper = _helper()
    opens = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and _call_name(node.func) == "open"
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "save_path"
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "xb"
    ]
    assert len(opens) == 1


def test_created_path_flag_is_initialized_false():
    helper = _helper()
    rows = [
        node
        for node in helper.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "created_upload_path"
        and isinstance(node.value, ast.Constant)
        and node.value.value is False
    ]
    assert len(rows) == 1


def test_created_path_flag_becomes_true_before_save():
    helper = _helper()
    target_try = next(
        node
        for node in helper.body
        if isinstance(node, ast.Try)
        and any(isinstance(child, ast.With) for child in node.body)
    )
    with_node = next(
        child for child in target_try.body if isinstance(child, ast.With)
    )

    true_index = None
    save_index = None

    for index, statement in enumerate(with_node.body):
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == "created_upload_path"
            and isinstance(statement.value, ast.Constant)
            and statement.value.value is True
        ):
            true_index = index

        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and _call_name(statement.value.func) == "file_obj.save"
        ):
            save_index = index

    assert true_index is not None
    assert save_index is not None
    assert true_index < save_index


def test_fileexists_handler_does_not_delete():
    helper = _helper()
    target_try = next(
        node
        for node in helper.body
        if isinstance(node, ast.Try)
        and any(
            isinstance(handler.type, ast.Name)
            and handler.type.id == "FileExistsError"
            for handler in node.handlers
            if handler.type is not None
        )
    )
    handler = next(
        handler
        for handler in target_try.handlers
        if isinstance(handler.type, ast.Name)
        and handler.type.id == "FileExistsError"
    )
    text = ast.unparse(handler)
    assert "os.remove" not in text
    assert "created_upload_path" not in text


def test_generic_exception_cleanup_is_guarded():
    helper = _helper()
    target_try = next(
        node
        for node in helper.body
        if isinstance(node, ast.Try)
        and any(
            isinstance(handler.type, ast.Name)
            and handler.type.id == "Exception"
            for handler in node.handlers
            if handler.type is not None
        )
    )
    handler = next(
        handler
        for handler in target_try.handlers
        if isinstance(handler.type, ast.Name)
        and handler.type.id == "Exception"
    )
    guarded = [
        node
        for node in handler.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "created_upload_path"
    ]
    assert len(guarded) == 1


def test_generic_exception_removes_exact_save_path():
    helper = _helper()
    text = ast.unparse(helper)
    assert "if created_upload_path:" in text
    assert "os.remove(save_path)" in text


def test_cleanup_failure_is_best_effort_oserror():
    helper = _helper()
    text = ast.unparse(helper)
    assert "except OSError:" in text


def test_success_path_still_saves_destination():
    helper = _helper()
    text = ast.unparse(helper)
    assert "file_obj.save(destination)" in text


def test_collision_message_is_preserved():
    helper = _helper()
    text = ast.unparse(helper)
    assert "Unable to allocate a unique upload filename; please retry." in text


def test_no_hash_or_identity_behavior_is_added():
    helper = _helper()
    text = ast.unparse(helper)
    assert "hashlib" not in text
    assert "sha256" not in text.lower()
    assert "ArtifactIdentityHandoff" not in text

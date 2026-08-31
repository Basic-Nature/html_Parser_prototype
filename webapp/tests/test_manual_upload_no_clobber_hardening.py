from __future__ import annotations
import ast
from pathlib import Path
TARGET_REL = "webapp/Smart_Elections_Parser_Webapp.py"

def _helper():
    tree = ast.parse(Path(TARGET_REL).read_text(encoding="utf-8"), filename=TARGET_REL)
    xs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_save_uploaded_file"]
    assert len(xs) == 1
    return xs[0]

def _name(n):
    if isinstance(n, ast.Name): return n.id
    if isinstance(n, ast.Attribute):
        left = _name(n.value)
        return f"{left}.{n.attr}" if left else n.attr
    return ""

def test_exclusive_xb_open_exact():
    h = _helper()
    xs = [n for n in ast.walk(h) if isinstance(n, ast.Call) and _name(n.func) == "open"]
    assert len(xs) == 1
    c = xs[0]
    assert len(c.args) >= 2
    assert isinstance(c.args[0], ast.Name) and c.args[0].id == "save_path"
    assert isinstance(c.args[1], ast.Constant) and c.args[1].value == "xb"

def test_save_receives_destination():
    h = _helper()
    xs = [n for n in ast.walk(h) if isinstance(n, ast.Call) and _name(n.func) == "file_obj.save"]
    assert len(xs) == 1
    assert len(xs[0].args) == 1
    assert isinstance(xs[0].args[0], ast.Name) and xs[0].args[0].id == "destination"

def test_direct_save_path_removed():
    text = ast.unparse(_helper())
    assert "file_obj.save(save_path)" not in text

def test_file_exists_handler_once():
    h = _helper()
    xs = [n for n in ast.walk(h) if isinstance(n, ast.ExceptHandler) and isinstance(n.type, ast.Name) and n.type.id == "FileExistsError"]
    assert len(xs) == 1

def test_collision_handler_precedes_generic():
    h = _helper()
    hits = []
    for n in ast.walk(h):
        if isinstance(n, ast.Try):
            names = [x.type.id for x in n.handlers if isinstance(x.type, ast.Name)]
            if "FileExistsError" in names: hits.append(names)
    assert len(hits) == 1
    assert hits[0].index("FileExistsError") < hits[0].index("Exception")

def test_success_return_preserved():
    vals = {ast.unparse(n.value) for n in ast.walk(_helper()) if isinstance(n, ast.Return) and n.value is not None}
    assert "(True, filename, save_path)" in vals

def test_generic_failure_return_preserved():
    vals = {ast.unparse(n.value) for n in ast.walk(_helper()) if isinstance(n, ast.Return) and n.value is not None}
    assert "(False, f'Failed to save upload: {exc}', None)" in vals

def test_validation_preserved_once():
    xs = [n for n in ast.walk(_helper()) if isinstance(n, ast.Call) and _name(n.func) == "_validate_uploaded_file"]
    assert len(xs) == 1

def test_failure_cleanup_preserved_once():
    xs = [n for n in ast.walk(_helper()) if isinstance(n, ast.Call) and _name(n.func) == "os.remove"]
    assert len(xs) == 1

def test_filename_generator_preserved_once():
    xs = [n for n in ast.walk(_helper()) if isinstance(n, ast.Call) and _name(n.func) == "_generate_upload_filename"]
    assert len(xs) == 1

def test_no_retry_loop_added():
    assert [n for n in ast.walk(_helper()) if isinstance(n, (ast.For, ast.While))] == []

def test_no_hash_or_identity_logic():
    text = ast.unparse(_helper()).lower()
    assert "sha256" not in text and "hashlib" not in text
    assert "artifact_identity" not in text and "artifactidentityhandoff" not in text

def test_no_parser_geometry_logic():
    text = ast.unparse(_helper()).lower()
    for token in ("parse_pdf_election_results", "_get_page_orientation_map", "_collect_page_orientation", "pdf_structure_profile"):
        assert token not in text

def test_builtin_open_not_os_open():
    helper = _helper()
    builtin_opens = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and _name(node.func) == "open"
    ]
    os_opens = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and _name(node.func) == "os.open"
    ]
    assert len(builtin_opens) == 1
    call = builtin_opens[0]
    assert len(call.args) >= 2
    assert isinstance(call.args[0], ast.Name)
    assert call.args[0].id == "save_path"
    assert isinstance(call.args[1], ast.Constant)
    assert call.args[1].value == "xb"
    assert os_opens == []

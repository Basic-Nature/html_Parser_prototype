"""W24PUBINGEST-AZ1B3A REC1B portable integration contracts.

Run from webapp/tests after exact-file integration, or from an identically laid-out
external shadow tree. No original builder, source downloads, DB, or live URLs.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest

WEBAPP = Path(__file__).resolve().parents[1]
JSON_HANDLER = WEBAPP / "parser/handlers/formats/json_handler.py"
OUTPUT_UTILS = WEBAPP / "parser/utils/output_utils.py"
ADAPTER = WEBAPP / "parser/utils/enhanced_voting_precinct_adapter.py"


def load_adapter():
    spec = importlib.util.spec_from_file_location("w24az_adapter_contract", ADAPTER)
    assert spec and spec.loader, "Candidate adapter is not importable from the target tree"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class IntegrationContract(unittest.TestCase):
    def test_deployed_five_file_layout_and_syntax(self):
        for target in (JSON_HANDLER, OUTPUT_UTILS, ADAPTER):
            with self.subTest(target=str(target.relative_to(WEBAPP))):
                self.assertTrue(target.is_file(), f"Missing five-file candidate: {target}")
                ast.parse(target.read_text(encoding="utf-8"), filename=str(target))
        # The test files must work from their exact proposed webapp/tests home.
        self.assertEqual(ADAPTER.parent.parent, WEBAPP / "parser")

    def test_handler_uses_explicit_opt_in_selection(self):
        src = JSON_HANDLER.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(JSON_HANDLER))
        self.assertIn("enhanced_voting_precinct_selection", src)
        self.assertIn("enhanced_voting_selection", src)
        self.assertIn("contest_id", src)
        self.assertIn("preformatted_smart_elections_wide", src)
        self.assertIn("finalize_election_output", src)
        self.assertTrue(any(isinstance(n, ast.FunctionDef) and n.name == "parse" for n in ast.walk(tree)))

    def test_legacy_inner_parser_unchanged_and_new_branch_explicit(self):
        source = JSON_HANDLER.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(JSON_HANDLER))
        inner = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse_json_election_results")
        seg = ast.get_source_segment(source, inner)
        self.assertEqual(hashlib.sha256(seg.encode("utf-8")).hexdigest(),
            "4b619ce0479eab2658044c41b5204e0f975a7ecf66123a0ef5685042b310438e")
        self.assertNotIn("enhanced_voting_selection", seg)
        wrapper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse")
        body = ast.get_source_segment(source, wrapper)
        self.assertIn("if ev_selection is not None:", body)
        self.assertIn("result = parse_json_election_results(\n            manual_file, session_id=session_id, coordinator=coordinator,\n        )", body)
        helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_parse_enhanced_voting_precinct_json"]
        self.assertEqual(len(helpers), 1)
        self.assertIn("build_precinct_table", ast.get_source_segment(source, helpers[0]))

    def test_finalizer_compatibility_switch_is_explicit(self):
        src = OUTPUT_UTILS.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(OUTPUT_UTILS))
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "finalize_election_output"]
        self.assertEqual(len(functions), 1)
        fn_source = ast.get_source_segment(src, functions[0]) or ""
        self.assertIn('preformatted_smart_elections_wide', fn_source)
        self.assertIn('transform_wide_to_smart_standard', fn_source)
        self.assertIn('smart_applied', fn_source)

    def test_adapter_fails_closed_for_nonmatching_json(self):
        mod = load_adapter()
        self.assertFalse(mod.matches_enhanced_voting_shape({}))
        self.assertFalse(mod.matches_enhanced_voting_shape({"localResults": []}))
        self.assertTrue(callable(mod.build_precinct_table))


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Isolated J2B candidate contract tests; stdlib only, no DB/network."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import unittest
from uuid import uuid4

SOURCE = Path(__file__).resolve().parents[2] / "webapp/parser/services/project_run_intent_contract.py"
spec = importlib.util.spec_from_file_location("candidate_project_run_intent", SOURCE)
module = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class IntentContractTests(unittest.TestCase):
    def setUp(self):
        self.body = {"source_ref_id": str(uuid4()), "workflow_item_id": str(uuid4()),
                     "expected_project_version": 4, "idempotency_key": str(uuid4())}

    def test_valid_intent_only_selectors(self):
        obj = module.parse_browser_run_intent(self.body)
        self.assertEqual(obj.expected_project_version, 4)
        self.assertEqual(len(obj.safe_request_fingerprint_fields()), 3)

    def test_reject_any_unexpected_fields_including_url_or_workflow_pass(self):
        for key in ("source_url", "direct_urls", "workflow_pass_id", "execution_mode", "is_admin"):
            with self.subTest(key=key), self.assertRaises(module.InvalidRunIntent):
                module.parse_browser_run_intent({**self.body, key: "untrusted"})

    def test_missing_keys_and_non_mapping(self):
        for bad in (None, {}, [], {k:v for k,v in self.body.items() if k != "idempotency_key"}):
            with self.assertRaises(module.InvalidRunIntent):
                module.parse_browser_run_intent(bad)

    def test_reject_bool_and_invalid_versions(self):
        for v in (True, False, 0, -1, 2147483648, "4", 4.0):
            with self.subTest(version=v), self.assertRaises(module.InvalidRunIntent):
                module.parse_browser_run_intent({**self.body, "expected_project_version": v})

    def test_reject_noncanonical_and_malformed_uuid(self):
        for v in ("bad", "{" + self.body["source_ref_id"] + "}", "", 123):
            with self.subTest(value=v), self.assertRaises(module.InvalidRunIntent):
                module.parse_browser_run_intent({**self.body, "source_ref_id": v})

    def test_idempotency_only_one_selector(self):
        obj = module.parse_browser_run_intent(self.body)
        self.assertNotIn(str(obj.idempotency_key), obj.safe_request_fingerprint_fields())

    def test_no_self_authorizing_state_transition(self):
        for to in (module.RunState.RUNNING, module.RunState.COMPLETED):
            with self.assertRaises(module.InvalidRunIntent):
                module.require_state_transition(module.RunState.ADMITTED, to)

    def test_valid_lifecycle_syntax(self):
        for current, next_state in (("admitted", "dispatch_pending"),
                                    ("dispatch_pending", "running"),
                                    ("running", "completed")):
            module.require_state_transition(module.RunState(current), module.RunState(next_state))

    def test_terminal_states_are_terminal(self):
        for state in (module.RunState.COMPLETED, module.RunState.FAILED, module.RunState.STALE):
            with self.assertRaises(module.InvalidRunIntent):
                module.require_state_transition(state, module.RunState.ADMITTED)

    def test_invalid_raw_state(self):
        with self.assertRaises(module.InvalidRunIntent):
            module.require_state_transition("admitted", module.RunState.RUNNING)


if __name__ == "__main__": unittest.main()

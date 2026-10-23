"""W24C3E isolated synthetic tests for proposed *pure* service module.

No webapp entrypoint/config, Flask, Azure, DB, or parser imports. Synthetic
fixture data deliberately do NOT represent extracted election observations.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

APPROVED_MODEL_SHA256 = "f0ff44a09880101cb4a6de64f119df196b25ad09ec09279ee926eed231f5704e"
APPROVED_SNAPSHOT_SHA256 = "04ece5376591c572d7da22fbf016fc34eb8edc2b9d78705055170104308dc0bd"


def _module_path() -> Path:
    here = Path(__file__).resolve().parent
    candidate = here / "operator_capacity_read_model.py"  # flat review kit
    if candidate.is_file():
        return candidate
    return here.parent / "parser" / "services" / "operator_capacity_read_model.py"  # proposed repo layout


def _load_pure_module():
    path = _module_path()
    spec = importlib.util.spec_from_file_location("w24c3e_pure_capacity_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("PURE_MODEL_MODULE_NOT_LOADABLE")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# SOURCE-FREE MOCK. Values are synthetic, used only to test projection semantics.
def _synthetic_snapshot():
    return {
        "id": "ELECTIONPULSE_W24C3B_READ_ONLY_PREVIEW_REV1",
        "classification": "ACCEPTED_HISTORICAL_EVIDENCE_PRESENTATION_NOT_LIVE",
        "design_contract": "W24C3_REV1",
        "as_of_utc": "2026-10-22",
        "scope": {
            "full_parser_executed_for_preview": False,
            "new_parser_admission_authorized": False,
            "runtime_admission_policy_deployed": False,
            "new_routes_added": False,
            "production_deployment_authorized": False,
            "production_mutation_authorized": False,
        },
        "admission": {
            "control_status": "PROPOSED_NOT_DEPLOYED",
            "full_parser_execution_authority": "NOT_GRANTED",
            "production_b1_new_heavy_job_policy_slots": 0,
            "actual_b1_heavy_job_capacity": None,
            "unknown_safe_concurrency": None,
            "unknown_worker_ram_bytes": None,
            "unknown_worker_temp_bytes": None,
        },
        "service_baseline": {
            "window_start_utc": "2026-10-01T00:00:00Z",
            "window_end_utc": "2026-10-02T00:00:00Z",
            "app_service_plan": {"cpu_mean_percent": 12.5, "memory_p95_percent": 70.0},
        },
        "processing_evidence": {
            "local_json_decode": {"one_time_authority": "PERMANENTLY_SPENT", "full_parser_extraction": False,
                "os_process_tree_peak_bytes": None, "temporary_disk_peak_bytes": None, "result_zip_sha256": "a"*64},
            "source_only_oracle": {"finalized_csv_validated": False, "source_only": True,
                "precinct_unit_count": 3, "expected_method_not_explicit": ["Provisional"], "result_zip_sha256": "b"*64},
            "qualification": {"full_parser_invoked": False, "process_tree_peak_qualified": False},
        },
        "historical_cost": {"currency": "USD", "month": "2026-09", "total_usd": 99.0,
            "marginal_cost_per_job_usd": None, "current_monthly_bill_usd": None,
            "telemetry_ingestion_cost_usd": None},
        "future_ledger_state": {"live_events_available": False, "operator_job_cost_records_available": False},
    }


def _mock_bytes(payload=None):
    return (json.dumps(_synthetic_snapshot() if payload is None else payload,
                       sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


class W24C3EPureCapacityModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_pure_module()

    def ctx(self, **changes):
        fields = dict(method="GET", principal_authenticated=True,
                      operator_capability_server_verified=True,
                      authority_state="fresh_certificate", session_unexpired=True,
                      principal_restricted=False, development_bypass=False)
        fields.update(changes)
        return self.mod.ReadContext(**fields)

    def project(self, *, ctx=None, raw=None, approved=None, spy=None):
        raw = _mock_bytes() if raw is None else raw
        digest = hashlib.sha256(raw).hexdigest()
        with patch.object(self.mod, "APPROVED_SNAPSHOT_SHA256", digest):
            return self.mod.project_authorized_capacity_read(
                context=self.ctx() if ctx is None else ctx,
                snapshot_loader=(lambda:raw) if spy is None else spy,
                approved_snapshot_sha256=digest if approved is None else approved)

    def test_01_exact_historical_snapshot_constant(self):
        self.assertEqual(self.mod.APPROVED_SNAPSHOT_SHA256, APPROVED_SNAPSHOT_SHA256)

    def test_02_base_module_matches_accepted_c3d_hash(self):
        self.assertEqual(hashlib.sha256(_module_path().read_bytes()).hexdigest(), APPROVED_MODEL_SHA256)

    def test_03_projection_only_allowlisted_fields(self):
        p = self.project()
        self.assertEqual(set(p), {"contract","as_of_utc","observation_class","basis_digest","capacity","admission","quality","cost","freshness"})

    def test_04_unknown_capacity_not_converted_to_zero(self):
        p = self.project(); self.assertIsNone(p["admission"]["physical_maximum_concurrency"])
        self.assertIsNone(p["capacity"]["instantaneous_headroom_bytes"])

    def test_05_policy_slots_not_hardware(self):
        p=self.project();self.assertEqual(p["admission"]["policy_validated_heavy_slots"],0)
        self.assertFalse(p["admission"]["new_heavy_jobs_allowed"])

    def test_06_cost_not_job_price(self):
        p=self.project();self.assertIsNone(p["cost"]["job_marginal_cost"])
        self.assertEqual(p["cost"]["historical_invoice_total"],99.0)
        self.assertIn("NOT_PER_JOB_COST",p["cost"]["amount_class"])

    def test_07_oracle_not_final_csv(self):
        p=self.project();self.assertFalse(p["quality"]["finalized_csv_reconciled"])
        self.assertEqual(p["quality"]["source_precinct_count"],3)

    def test_08_method_missing_not_observed_zero(self):
        self.assertEqual(self.project()["quality"]["method_availability_class"],"PROVISIONAL_NOT_EXPLICIT_IN_SOURCE")

    def test_09_refuse_unauthenticated(self):
        with self.assertRaises(self.mod.CapacityReadDenied):self.project(ctx=self.ctx(principal_authenticated=False))

    def test_10_refuse_no_operator_capability(self):
        with self.assertRaises(self.mod.CapacityReadDenied):self.project(ctx=self.ctx(operator_capability_server_verified=False))

    def test_11_refuse_restricted(self):
        with self.assertRaises(self.mod.CapacityReadDenied):self.project(ctx=self.ctx(principal_restricted=True))

    def test_12_refuse_dev_bypass(self):
        with self.assertRaises(self.mod.CapacityReadDenied):self.project(ctx=self.ctx(development_bypass=True))

    def test_13_refuse_expired(self):
        with self.assertRaises(self.mod.CapacityReadDenied):self.project(ctx=self.ctx(session_unexpired=False))

    def test_14_refuse_unapproved_authority(self):
        with self.assertRaises(self.mod.CapacityReadDenied):self.project(ctx=self.ctx(authority_state="public_read"))

    def test_15_refuse_post(self):
        with self.assertRaises(self.mod.CapacityReadDenied):self.project(ctx=self.ctx(method="POST"))

    def test_16_refuse_head(self):
        with self.assertRaises(self.mod.CapacityReadDenied):self.project(ctx=self.ctx(method="HEAD"))

    def test_17_refuse_untrusted_dictionary_context(self):
        with self.assertRaises(self.mod.CapacityReadDenied):self.project(ctx={"principal_authenticated":True})

    def test_18_denial_before_evidence_loader(self):
        hits=[]
        def fail():hits.append(1);raise AssertionError("should not run")
        with self.assertRaises(self.mod.CapacityReadDenied):self.project(ctx=self.ctx(development_bypass=True),spy=fail)
        self.assertEqual(hits,[])

    def test_19_valid_evidence_loader_called_once(self):
        hits=[]
        def record():hits.append(1);return _mock_bytes()
        self.project(spy=record)
        self.assertEqual(hits,[1])

    def test_20_refuse_changed_source_hash(self):
        with self.assertRaises(self.mod.CapacityEvidenceRejected):
            self.project(raw=_mock_bytes(), spy=lambda: _mock_bytes()+b" ")

    def test_21_refuse_snapshot_fingerprint_change(self):
        with self.assertRaises(self.mod.CapacityEvidenceRejected):self.project(approved="f"*64)

    def test_22_refuse_production_admission_promotion(self):
        d=_synthetic_snapshot();d["admission"]["production_b1_new_heavy_job_policy_slots"]=1
        with self.assertRaises(self.mod.CapacityEvidenceRejected):self.project(raw=_mock_bytes(d))

    def test_23_refuse_fabricated_process_peak(self):
        d=_synthetic_snapshot();d["processing_evidence"]["local_json_decode"]["os_process_tree_peak_bytes"]=1024
        with self.assertRaises(self.mod.CapacityEvidenceRejected):self.project(raw=_mock_bytes(d))

    def test_24_refuse_promotion_of_source_only_oracle(self):
        d=_synthetic_snapshot();d["processing_evidence"]["source_only_oracle"]["finalized_csv_validated"]=True
        with self.assertRaises(self.mod.CapacityEvidenceRejected):self.project(raw=_mock_bytes(d))

    def test_25_prevent_secret_field_echo(self):
        d=_synthetic_snapshot();d.update({"principal":"cert:fictional", "password":"fictional-secret", "raw_votes":[99999]})
        p=self.project(raw=_mock_bytes(d));s=json.dumps(p)
        for v in ("cert:fictional","fictional-secret","raw_votes","99999"):
            self.assertNotIn(v,s)

    def test_26_refuse_invalid_boolean_truthy(self):
        with self.assertRaises(self.mod.CapacityReadDenied):self.project(ctx=self.ctx(operator_capability_server_verified="true"))

    def test_27_no_webapp_flask_network_database_import(self):
        source=_module_path().read_text(encoding="utf-8")
        for word in ("from webapp", "import flask", "import requests", "import sqlalchemy", "az.cmd"):
            self.assertNotIn(word,source)

    def test_28_repeatable_projection(self):
        self.assertEqual(self.project(),self.project())

    def test_29_unsupported_http_method_denied_before_load(self):
        called=[]
        with self.assertRaises(self.mod.CapacityReadDenied):self.project(ctx=self.ctx(method="OPTIONS"),spy=lambda: called.append(1))
        self.assertFalse(called)

    def test_30_reject_missing_provisional_source_marker(self):
        d=_synthetic_snapshot();d["processing_evidence"]["source_only_oracle"]["expected_method_not_explicit"]=[]
        with self.assertRaises(self.mod.CapacityEvidenceRejected):self.project(raw=_mock_bytes(d))

    def test_31_refuse_future_ledger_fabrication(self):
        d=_synthetic_snapshot();d["future_ledger_state"]["live_events_available"]=True
        with self.assertRaises(self.mod.CapacityEvidenceRejected):self.project(raw=_mock_bytes(d))

    def test_32_zero_cpu_is_a_measurement_not_unknown(self):
        d=_synthetic_snapshot();d["service_baseline"]["app_service_plan"]["cpu_mean_percent"]=0
        self.assertEqual(self.project(raw=_mock_bytes(d))["capacity"]["app_service_plan_cpu_mean_pct"],0.0)

    def test_33_no_implicit_route_grant(self):
        self.assertFalse(hasattr(self.mod,"app"))
        self.assertFalse(hasattr(self.mod,"Blueprint"))
        self.assertTrue(callable(self.mod.project_authorized_capacity_read))

    def test_34_python_dataclass_is_not_credential(self):
        self.assertTrue(self.ctx().operator_capability_server_verified)
        self.assertIn("NOT a verified credential",self.mod.__doc__ or "")


if __name__ == "__main__":
    unittest.main(verbosity=2)

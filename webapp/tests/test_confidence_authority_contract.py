# Characterization tests for the ElectionPulse confidence-authority boundary.
# Phase 1 tests ownership/delegation without changing scoring behavior.

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from webapp.parser.health.risk_gates import RiskGateEvaluator
from webapp.parser.health.risk_gates_calculus import CalculusRiskEvaluator


REPO_ROOT = Path(__file__).resolve().parents[2]
RISK_GATES_PATH = REPO_ROOT / "webapp" / "parser" / "health" / "risk_gates.py"
CALCULUS_PATH = REPO_ROOT / "webapp" / "parser" / "health" / "risk_gates_calculus.py"
AUTHORITY_DOC = REPO_ROOT / "docs" / "ARCHITECTURE" / "confidence_authority.md"


def _project_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("webapp."):
                    imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("webapp."):
                imports.add(module)

    return imports


def _sample_inputs() -> dict:
    return {
        "extraction_confidence": 0.84,
        "ground_truth_matches": 8,
        "total_records": 10,
        "suspicious_pattern_count": 1,
        "outlier_record_count": 0,
        "integrity_flags": [],
        "fallback_verification_score": None,
    }


def test_risk_gates_is_domain_agnostic_current_state_evaluator():
    assert _project_imports(RISK_GATES_PATH) == set()

    module = __import__("webapp.parser.health.risk_gates", fromlist=["*"])
    doc = inspect.getdoc(module)
    assert doc is not None
    normalized = " ".join(doc.split())
    assert "owns the normalized CURRENT risk state" in normalized
    assert "not, by themselves, authority" in normalized


def test_calculus_depends_on_base_risk_state_not_domain_modules():
    assert _project_imports(CALCULUS_PATH) == {
        "webapp.parser.health.risk_gates"
    }

    module = __import__(
        "webapp.parser.health.risk_gates_calculus",
        fromlist=["*"],
    )
    doc = inspect.getdoc(module)
    assert doc is not None
    assert "TRAJECTORY / BOUNDARY / CONVERGENCE" in doc
    assert "base current-state vector remains owned by risk_gates.py" in doc


def test_calculus_reuses_base_current_state_without_reinterpreting_it():
    base = RiskGateEvaluator()
    calculus = CalculusRiskEvaluator()

    expected = base.evaluate(**_sample_inputs())
    actual, _derivatives, _sub_tier = calculus.evaluate_with_derivatives(
        **_sample_inputs(),
        previous_scores=None,
        time_delta=1.0,
    )

    assert actual == expected


def test_authority_contract_is_provenance_based_not_gov_only():
    text = AUTHORITY_DOC.read_text(encoding="utf-8")

    assert "Official source is a provenance relationship, not a TLD classification." in text
    assert "officially delegated third-party source" in text
    assert "Vendor reputation alone is not equivalent to jurisdictional delegation." in text
    assert "Hard security/authorization constraints are not averaged away." in text


def test_authority_contract_separates_measurement_from_truth_promotion():
    text = AUTHORITY_DOC.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "A local algorithm may need thresholds for search, ranking" in normalized
    assert "do not automatically become authority to declare election truth" in normalized
    assert "risk_gates.py" in text
    assert "risk_gates_calculus.py" in text

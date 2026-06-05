from __future__ import annotations

from tools import smoke_webapp_api as smoke


def test_build_artifact_contains_summary_fields() -> None:
    results = [
        smoke.CheckResult(
            name="Health",
            method="GET",
            url="http://127.0.0.1:5000/health",
            ok=True,
            status=200,
            elapsed_ms=12,
            detail='{"status":"ok"}',
        ),
        smoke.CheckResult(
            name="Integrity Signal",
            method="POST",
            url="http://127.0.0.1:5000/api/integrity_signal",
            ok=False,
            status=403,
            elapsed_ms=24,
            detail='{"error":"forbidden"}',
        ),
    ]

    artifact = smoke.build_artifact("http://127.0.0.1:5000", 8.0, results)

    assert artifact["base_url"] == "http://127.0.0.1:5000"
    assert artifact["timeout_seconds"] == 8.0
    assert artifact["summary"]["passed"] == 1
    assert artifact["summary"]["failed"] == 1
    assert artifact["summary"]["total"] == 2
    assert len(artifact["results"]) == 2


def test_compare_results_detects_status_deltas() -> None:
    primary = [
        smoke.CheckResult("Health", "GET", "http://a/health", True, 200, 10, "ok"),
        smoke.CheckResult("Certificate Info", "GET", "http://a/api/auth/certificate_info", True, 401, 11, "cert required"),
    ]
    secondary = [
        smoke.CheckResult("Health", "GET", "http://b/health", True, 200, 9, "ok"),
        smoke.CheckResult("Certificate Info", "GET", "http://b/api/auth/certificate_info", True, 200, 8, "cert ok"),
    ]

    diffs = smoke.compare_results(primary, secondary)

    assert len(diffs) == 1
    assert "Certificate Info" in diffs[0]


def test_run_smoke_suite_uses_injected_checks(monkeypatch) -> None:
    checks = [
        smoke.EndpointCheck(name="Health", method="GET", path="/health", expected_statuses=(200,)),
        smoke.EndpointCheck(name="Heartbeat", method="GET", path="/heartbeat", expected_statuses=(200,)),
    ]

    def fake_run_check(base_url: str, timeout: float, check: smoke.EndpointCheck) -> smoke.CheckResult:
        return smoke.CheckResult(
            name=check.name,
            method=check.method,
            url=f"{base_url}{check.path}",
            ok=True,
            status=200,
            elapsed_ms=1,
            detail="ok",
        )

    monkeypatch.setattr(smoke, "_run_check", fake_run_check)

    results = smoke.run_smoke_suite("http://127.0.0.1:5000", 2.0, checks)

    assert len(results) == 2
    assert all(r.ok for r in results)
    assert {r.name for r in results} == {"Health", "Heartbeat"}

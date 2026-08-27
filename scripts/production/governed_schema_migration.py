from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

APP_NAME = "BallotLens"
RESOURCE_GROUP = "BallotLens_group"
EXPECTED_DB_NAME = "ballotlens-database"
EXPECTED_DB_HOST = "ballotlens-server.postgres.database.azure.com"
WEBJOB_NAME = "ElectionPulseGovernedSchemaMigration"
REGISTRY_FILENAME = "schema_migration_registry.json"
RESULT_MARKER = "EP_SCHEMA_MIGRATION_RESULT_JSON="
MAX_HISTORY_WAIT_SECONDS = 300


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_registry(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "electionpulse_production_schema_migration_registry_v1":
        raise RuntimeError("Unexpected schema migration registry version.")
    migrations = payload.get("migrations")
    if not isinstance(migrations, dict) or not migrations:
        raise RuntimeError("Schema migration registry has no migrations.")
    return payload


def get_spec(registry: dict[str, Any], target: str) -> dict[str, Any]:
    spec = registry["migrations"].get(target)
    if not isinstance(spec, dict):
        raise RuntimeError(f"Target revision {target!r} is not allow-listed.")
    if spec.get("target_revision") != target:
        raise RuntimeError("Registry target_revision does not match lookup key.")
    return spec


def expected_confirmation(spec: dict[str, Any]) -> str:
    return f"APPLY:{spec['from_revision']}:{spec['target_revision']}"


def run_process(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        argv,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(argv)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def find_az() -> str:
    az = shutil.which("az")
    if not az:
        raise RuntimeError("Azure CLI was not found.")
    return str(Path(az).resolve())


def run_az(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    az = find_az()
    if Path(az).suffix.lower() in {".cmd", ".bat"}:
        argv = ["cmd.exe", "/d", "/c", "call", az, *args]
    else:
        argv = [az, *args]
    return run_process(argv, check=check)


def az_json(*args: str) -> Any:
    proc = run_az(*args, "-o", "json")
    return json.loads(proc.stdout)


def az_tsv(*args: str) -> str:
    return run_az(*args, "-o", "tsv").stdout.strip()


def resolve_scm_base(site: dict[str, Any]) -> str:
    # Resolve the authoritative Kudu/SCM HTTPS origin from Azure metadata.
    candidates: list[str] = []

    ssl_states = site.get("hostNameSslStates")
    if isinstance(ssl_states, list):
        for item in ssl_states:
            if not isinstance(item, dict):
                continue
            if str(item.get("hostType") or "").strip().lower() != "repository":
                continue
            host = str(item.get("name") or "").strip().lower().rstrip(".")
            if host:
                candidates.append(host)

    # Some API/CLI projections can omit hostNameSslStates. Fall back to the
    # enabled SCM hostname, but still require a single unambiguous Azure host.
    if not candidates:
        enabled = site.get("enabledHostNames")
        if isinstance(enabled, list):
            for value in enabled:
                host = str(value or "").strip().lower().rstrip(".")
                if ".scm." in host and host.endswith(".azurewebsites.net"):
                    candidates.append(host)

    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise RuntimeError(
            "Could not resolve exactly one authoritative App Service SCM hostname "
            f"from Azure metadata: {candidates!r}"
        )

    host = candidates[0]
    if not host.endswith(".azurewebsites.net") or ".scm." not in host:
        raise RuntimeError(f"Resolved unexpected SCM hostname: {host!r}")

    return f"https://{host}"


def ensure_azure_target() -> dict[str, Any]:
    site = az_json(
        "webapp",
        "show",
        "--name",
        APP_NAME,
        "--resource-group",
        RESOURCE_GROUP,
    )
    config = az_json(
        "webapp",
        "config",
        "show",
        "--name",
        APP_NAME,
        "--resource-group",
        RESOURCE_GROUP,
    )

    state = {
        "name": site.get("name"),
        "resource_group": site.get("resourceGroup"),
        "state": site.get("state"),
        "client_cert_enabled": site.get("clientCertEnabled"),
        "client_cert_mode": site.get("clientCertMode"),
        "https_only": site.get("httpsOnly"),
        "http20_enabled": config.get("http20Enabled"),
        "min_tls_version": config.get("minTlsVersion"),
        "linux_fx_version": config.get("linuxFxVersion"),
        "scm_base": resolve_scm_base(site),
    }

    if state["name"] != APP_NAME or state["resource_group"] != RESOURCE_GROUP:
        raise RuntimeError(f"Resolved unexpected App Service target: {state!r}")
    if state["state"] != "Running":
        raise RuntimeError(f"App Service is not Running: {state['state']!r}")
    if state["client_cert_enabled"] is not True:
        raise RuntimeError("clientCertEnabled drifted from accepted True state.")
    if state["client_cert_mode"] != "OptionalInteractiveUser":
        raise RuntimeError(
            "clientCertMode drifted from accepted OptionalInteractiveUser state: "
            f"{state['client_cert_mode']!r}"
        )

    app_settings = az_json(
        "webapp",
        "config",
        "appsettings",
        "list",
        "--name",
        APP_NAME,
        "--resource-group",
        RESOURCE_GROUP,
    )
    settings = {
        str(item.get("name")): str(item.get("value") or "")
        for item in app_settings
        if isinstance(item, dict) and item.get("name")
    }
    if settings.get("WEBSITE_SKIP_RUNNING_KUDUAGENT", "").strip().lower() == "true":
        raise RuntimeError(
            "WEBSITE_SKIP_RUNNING_KUDUAGENT=true prevents governed WebJob execution."
        )

    return state


def require_deployed_commit(expected_sha: str, azure_state: dict[str, Any]) -> None:
    linux_fx = str(azure_state.get("linux_fx_version") or "")
    if expected_sha and expected_sha not in linux_fx:
        raise RuntimeError(
            "Production container image does not reference the workflow commit SHA. "
            f"expected={expected_sha} linuxFxVersion={linux_fx!r}"
        )


def acquire_kudu_token() -> str:
    token = az_tsv(
        "account",
        "get-access-token",
        "--query",
        "accessToken",
    )
    if not token:
        raise RuntimeError("Azure CLI returned an empty Microsoft Entra access token.")
    return token


def kudu_request(
    token: str,
    method: str,
    path_or_url: str,
    *,
    scm_base: str,
    body: bytes | None = None,
    content_type: str | None = None,
    content_disposition: str | None = None,
    timeout: int = 60,
) -> tuple[int, dict[str, str], bytes]:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        parsed = urllib.parse.urlparse(path_or_url)
        url = scm_base.rstrip("/") + parsed.path
        if parsed.query:
            url += "?" + parsed.query
    else:
        url = scm_base.rstrip("/") + "/" + path_or_url.lstrip("/")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": "ElectionPulse-GovernedSchemaMigration/1.0",
    }
    if content_type:
        headers["Content-Type"] = content_type
    if content_disposition:
        headers["Content-Disposition"] = content_disposition

    req = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return (
                int(resp.status),
                {k.lower(): v for k, v in resp.headers.items()},
                resp.read(),
            )
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Kudu HTTP {exc.code} for {method} "
            f"{urllib.parse.urlparse(url).path}: {body_text[:1200]}"
        ) from exc


def build_job_zip(script_path: Path, registry_path: Path) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("run.py", script_path.read_bytes())
        archive.writestr(REGISTRY_FILENAME, registry_path.read_bytes())
        archive.writestr(
            "settings.job",
            json.dumps({"is_singleton": True, "stopping_wait_time": 60}),
        )
    return payload.getvalue()


def upload_job(token: str, job_zip: bytes, *, scm_base: str) -> None:
    status, _, _ = kudu_request(
        token,
        "PUT",
        f"/api/triggeredwebjobs/{WEBJOB_NAME}",
        scm_base=scm_base,
        body=job_zip,
        content_type="application/zip",
        content_disposition=f"attachment; filename={WEBJOB_NAME}.zip",
        timeout=90,
    )
    if status not in {200, 201, 202, 204}:
        raise RuntimeError(f"Unexpected Kudu WebJob upload status: {status}")


def trigger_job(
    token: str,
    *,
    scm_base: str,
    mode: str,
    target: str,
    confirmation: str,
) -> str:
    args = [
        "--role",
        "worker",
        "--mode",
        mode,
        "--target",
        target,
    ]
    if confirmation:
        args.extend(["--confirmation", confirmation])
    query = urllib.parse.urlencode({"arguments": " ".join(args)})
    status, headers, _ = kudu_request(
        token,
        "POST",
        f"/api/triggeredwebjobs/{WEBJOB_NAME}/run?{query}",
        scm_base=scm_base,
        body=b"",
        timeout=60,
    )
    if status not in {200, 201, 202, 204}:
        raise RuntimeError(f"Unexpected Kudu trigger status: {status}")
    return headers.get("location", "")


def newest_history_url(token: str, *, scm_base: str) -> str:
    _, _, body = kudu_request(
        token,
        "GET",
        f"/api/triggeredwebjobs/{WEBJOB_NAME}/history",
        scm_base=scm_base,
        timeout=60,
    )
    payload = json.loads(body.decode("utf-8"))
    runs = payload.get("runs") if isinstance(payload, dict) else None
    if not runs:
        raise RuntimeError("Kudu WebJob history has no runs.")
    url = runs[0].get("url")
    if not url:
        raise RuntimeError("Newest Kudu WebJob run has no URL.")
    return str(url)


def wait_for_job(
    token: str,
    run_url: str,
    *,
    scm_base: str,
) -> tuple[dict[str, Any], str]:
    if not run_url:
        time.sleep(2)
        run_url = newest_history_url(token, scm_base=scm_base)

    deadline = time.time() + MAX_HISTORY_WAIT_SECONDS
    last: dict[str, Any] | None = None

    while time.time() < deadline:
        _, _, body = kudu_request(
            token,
            "GET",
            run_url,
            scm_base=scm_base,
            timeout=60,
        )
        payload = json.loads(body.decode("utf-8"))
        last = payload
        status = str(payload.get("status") or "").lower()
        if status not in {"running", "pending", "starting", "initializing"}:
            output = ""
            output_url = payload.get("output_url")
            if output_url:
                _, _, output_body = kudu_request(
                    token,
                    "GET",
                    str(output_url),
                    scm_base=scm_base,
                    timeout=60,
                )
                output = output_body.decode("utf-8", errors="replace")
            return payload, output
        time.sleep(3)

    raise RuntimeError(f"Timed out waiting for WebJob completion: {last!r}")


def delete_job(token: str, *, scm_base: str) -> None:
    status, _, _ = kudu_request(
        token,
        "DELETE",
        f"/api/triggeredwebjobs/{WEBJOB_NAME}",
        scm_base=scm_base,
        timeout=60,
    )
    if status not in {200, 202, 204}:
        raise RuntimeError(f"Unexpected Kudu WebJob delete status: {status}")


def parse_worker_result(output: str) -> dict[str, Any]:
    matches = [
        line[len(RESULT_MARKER):]
        for line in output.splitlines()
        if line.startswith(RESULT_MARKER)
    ]
    if not matches:
        raise RuntimeError("WebJob output did not contain governed result marker.")
    return json.loads(matches[-1])


def controller_main(args: argparse.Namespace) -> int:
    script_path = Path(__file__).resolve()
    registry_path = script_path.with_name(REGISTRY_FILENAME)
    registry = load_registry(registry_path)
    spec = get_spec(registry, args.target)

    expected = expected_confirmation(spec)
    if args.mode == "apply" and args.confirmation != expected:
        raise RuntimeError(
            "Apply confirmation mismatch. Expected exact value: "
            f"{expected}"
        )

    workflow_sha = os.environ.get("GITHUB_SHA", "").strip()
    evidence_root = Path("output/reports")
    evidence_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence_path = evidence_root / (
        f"production_schema_migration_{args.target}_{args.mode}_{stamp}.json"
    )

    evidence: dict[str, Any] = {
        "schema": "electionpulse_governed_schema_migration_controller_v1",
        "created_at_utc": utc_now(),
        "mode": args.mode,
        "target_revision": args.target,
        "from_revision": spec["from_revision"],
        "workflow_sha": workflow_sha or None,
        "result": "FAIL",
        "database_mutation": "NONE",
        "azure_config_mutation": "NONE",
        "source_mutation": "NONE",
        "git_mutation": "NONE",
        "temporary_webjob": "NOT_CREATED",
        "temporary_webjob_removed": False,
        "authentication": "microsoft_entra_bearer_to_kudu",
    }

    token = ""
    scm_base = ""
    uploaded = False
    try:
        azure_state = ensure_azure_target()
        evidence["azure_state_before"] = azure_state
        scm_base = str(azure_state["scm_base"])
        evidence["scm_base"] = scm_base
        if workflow_sha:
            require_deployed_commit(workflow_sha, azure_state)

        token = acquire_kudu_token()
        job_zip = build_job_zip(script_path, registry_path)
        evidence["job_zip_sha256"] = sha256_bytes(job_zip)

        upload_job(token, job_zip, scm_base=scm_base)
        uploaded = True
        evidence["temporary_webjob"] = "CREATED"

        run_url = trigger_job(
            token,
            scm_base=scm_base,
            mode=args.mode,
            target=args.target,
            confirmation=args.confirmation or "",
        )
        run_payload, output = wait_for_job(token, run_url, scm_base=scm_base)
        worker = parse_worker_result(output)

        evidence["webjob_status"] = run_payload.get("status")
        evidence["worker_result"] = worker

        if str(run_payload.get("status") or "").lower() != "success":
            raise RuntimeError(
                f"Governed migration WebJob status was {run_payload.get('status')!r}."
            )
        if worker.get("result") != "PASS":
            raise RuntimeError("Governed migration worker did not return PASS.")

        evidence["database_mutation"] = worker.get("database_mutation", "UNKNOWN")
        evidence["result"] = "PASS"

        print(output[-14000:])
        print(f"CONTROLLER_RESULT=PASS")
        print(f"MODE={args.mode}")
        print(f"TARGET_REVISION={args.target}")
        print(f"DATABASE_MUTATION={evidence['database_mutation']}")
        return 0

    except Exception as exc:
        evidence["error"] = str(exc)
        print(f"CONTROLLER_ERROR={exc}")
        return 1

    finally:
        if uploaded and token:
            try:
                delete_job(token, scm_base=scm_base)
                evidence["temporary_webjob_removed"] = True
                evidence["temporary_webjob"] = "CREATED_AND_REMOVED"
            except Exception as cleanup_exc:
                evidence["temporary_webjob_cleanup_error"] = str(cleanup_exc)

        evidence_path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"EVIDENCE={evidence_path}")


def worker_registry_path() -> Path:
    return Path(__file__).resolve().with_name(REGISTRY_FILENAME)


def find_app_root() -> Path:
    candidates = [
        Path("/app"),
        Path.cwd(),
    ]
    for candidate in candidates:
        root = candidate.resolve()
        if (root / "alembic.ini").is_file() and (root / "alembic").is_dir():
            return root
    raise RuntimeError("Could not locate application root containing alembic.ini.")


def connect_db():
    import psycopg2

    host = (os.environ.get("POSTGRES_HOST") or "").strip()
    port = int((os.environ.get("POSTGRES_PORT") or "5432").strip())
    db = (os.environ.get("POSTGRES_DB") or "").strip()
    user = (os.environ.get("POSTGRES_USER") or "").strip()
    password = os.environ.get("POSTGRES_PASSWORD") or ""

    if host.lower() != EXPECTED_DB_HOST:
        raise RuntimeError(f"Unexpected production DB host: {host!r}")
    if db != EXPECTED_DB_NAME:
        raise RuntimeError(f"Unexpected production DB name: {db!r}")
    if not user or not password:
        raise RuntimeError("POSTGRES_USER/POSTGRES_PASSWORD are required.")

    return psycopg2.connect(
        host=host,
        port=port,
        dbname=db,
        user=user,
        password=password,
        sslmode="require",
        connect_timeout=15,
        application_name="ElectionPulse-GovernedSchemaMigration",
    )


def build_database_url() -> str:
    host = (os.environ.get("POSTGRES_HOST") or "").strip()
    port = (os.environ.get("POSTGRES_PORT") or "5432").strip()
    db = (os.environ.get("POSTGRES_DB") or "").strip()
    user = (os.environ.get("POSTGRES_USER") or "").strip()
    password = os.environ.get("POSTGRES_PASSWORD") or ""

    if host.lower() != EXPECTED_DB_HOST or db != EXPECTED_DB_NAME:
        raise RuntimeError("Refusing Alembic URL for unexpected production database.")

    return (
        "postgresql+psycopg2://"
        + urllib.parse.quote_plus(user)
        + ":"
        + urllib.parse.quote_plus(password)
        + "@"
        + host
        + ":"
        + port
        + "/"
        + urllib.parse.quote_plus(db)
        + "?sslmode=require"
    )


def read_db_state(spec: dict[str, Any]) -> dict[str, Any]:
    expected_tables = list(spec.get("expected_new_tables") or [])
    support_tables = list(spec.get("required_existing_tables") or [])
    conn = connect_db()
    try:
        conn.autocommit = False
        cur = conn.cursor()
        cur.execute("SET TRANSACTION READ ONLY")

        cur.execute(
            """
            SELECT current_database(), current_user,
                   COALESCE(inet_server_addr()::text, ''),
                   inet_server_port(),
                   current_setting('transaction_read_only'),
                   version()
            """
        )
        (
            current_database,
            current_user,
            server_address_raw,
            server_port,
            transaction_read_only,
            server_version,
        ) = cur.fetchone()

        cur.execute("SELECT version_num FROM alembic_version")
        revisions = [str(row[0]) for row in cur.fetchall()]

        table_state: dict[str, bool] = {}
        for table in sorted(set(expected_tables + support_tables + ["alembic_version"])):
            cur.execute("SELECT to_regclass(%s)", (f"public.{table}",))
            table_state[table] = cur.fetchone()[0] is not None

        cur.execute("SELECT COUNT(*) FROM canonical_election_results")
        canonical_result_count = int(cur.fetchone()[0])

        cur.execute("SELECT COUNT(*) FROM canonical_election_races")
        canonical_race_count = int(cur.fetchone()[0])

        cur.execute(
            "SELECT COALESCE(SUM(total_votes), 0) FROM canonical_election_results"
        )
        canonical_total_votes_sum = int(cur.fetchone()[0])

        expected_table_counts: dict[str, int | None] = {}
        for table in expected_tables:
            if table_state.get(table):
                cur.execute(f'SELECT COUNT(*) FROM "{table}"')
                expected_table_counts[table] = int(cur.fetchone()[0])
            else:
                expected_table_counts[table] = None

        conn.rollback()
        return {
            "database": current_database,
            "user": current_user,
            "server_address_raw": server_address_raw,
            "server_address_normalized": (
                server_address_raw.split("/", 1)[0]
                if server_address_raw
                else server_address_raw
            ),
            "server_port": int(server_port),
            "transaction_read_only": transaction_read_only,
            "server_version": server_version,
            "alembic_revisions": revisions,
            "tables": table_state,
            "expected_table_counts": expected_table_counts,
            "canonical_result_count": canonical_result_count,
            "canonical_race_count": canonical_race_count,
            "canonical_total_votes_sum": canonical_total_votes_sum,
        }
    finally:
        conn.close()


def validate_source(app_root: Path, spec: dict[str, Any]) -> str:
    migration_path = app_root / spec["migration_path"]
    if not migration_path.is_file():
        raise RuntimeError(f"Migration file absent in deployed image: {migration_path}")
    actual = sha256_bytes(migration_path.read_bytes())
    expected = str(spec["migration_sha256"])
    if actual != expected:
        raise RuntimeError(
            f"Migration SHA mismatch. expected={expected} actual={actual}"
        )
    return actual


def classify_pre_state(
    state: dict[str, Any],
    spec: dict[str, Any],
) -> str:
    revisions = state["alembic_revisions"]
    expected_tables = list(spec["expected_new_tables"])

    missing_support = [
        table
        for table in spec["required_existing_tables"]
        if not state["tables"].get(table)
    ]
    if missing_support:
        raise RuntimeError(
            "Required pre-existing table(s) missing: " + ", ".join(missing_support)
        )

    if revisions == [spec["target_revision"]]:
        missing_expected = [
            table for table in expected_tables if not state["tables"].get(table)
        ]
        if missing_expected:
            raise RuntimeError(
                "Alembic is at target but expected table(s) are missing: "
                + ", ".join(missing_expected)
            )
        return "already_applied"

    if revisions != [spec["from_revision"]]:
        raise RuntimeError(
            f"Expected Alembic {spec['from_revision']} or already-applied "
            f"{spec['target_revision']}; found {revisions!r}."
        )

    partial = [
        table for table in expected_tables if state["tables"].get(table)
    ]
    if partial:
        raise RuntimeError(
            "Expected workflow tables absent before migration but found partial state: "
            + ", ".join(partial)
        )

    return "ready"


def run_exact_alembic(app_root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    env = dict(os.environ)
    env["DATABASE_URL"] = build_database_url()
    env["DEPLOY_ENV"] = "production"

    proc = run_process(
        [
            sys.executable,
            "-m",
            "alembic",
            "-c",
            str(app_root / "alembic.ini"),
            "upgrade",
            str(spec["target_revision"]),
        ],
        cwd=app_root,
        env=env,
        timeout=240,
        check=False,
    )

    password = os.environ.get("POSTGRES_PASSWORD") or ""
    stdout = proc.stdout.replace(password, "<REDACTED_PASSWORD>") if password else proc.stdout
    stderr = proc.stderr.replace(password, "<REDACTED_PASSWORD>") if password else proc.stderr

    return {
        "returncode": int(proc.returncode),
        "stdout_tail": stdout[-12000:],
        "stderr_tail": stderr[-12000:],
    }


def verify_post_state(
    before: dict[str, Any],
    after: dict[str, Any],
    spec: dict[str, Any],
) -> None:
    if after["alembic_revisions"] != [spec["target_revision"]]:
        raise RuntimeError(
            f"Post-migration Alembic mismatch: {after['alembic_revisions']!r}"
        )

    missing = [
        table
        for table in spec["expected_new_tables"]
        if not after["tables"].get(table)
    ]
    if missing:
        raise RuntimeError("Expected new table(s) missing: " + ", ".join(missing))

    if spec.get("expect_new_tables_empty"):
        nonempty = {
            table: count
            for table, count in after["expected_table_counts"].items()
            if count != 0
        }
        if nonempty:
            raise RuntimeError(
                f"Expected new workflow tables to be empty; found {nonempty!r}"
            )

    if spec.get("preserve_canonical_publication_metrics"):
        fields = (
            "canonical_result_count",
            "canonical_race_count",
            "canonical_total_votes_sum",
        )
        drift = {
            field: {"before": before[field], "after": after[field]}
            for field in fields
            if before[field] != after[field]
        }
        if drift:
            raise RuntimeError(
                f"Canonical publication metrics changed during migration: {drift!r}"
            )


def emit_worker(payload: dict[str, Any]) -> None:
    print(RESULT_MARKER + json.dumps(payload, sort_keys=True))


def worker_main(args: argparse.Namespace) -> int:
    registry = load_registry(worker_registry_path())
    spec = get_spec(registry, args.target)
    result: dict[str, Any] = {
        "schema": "electionpulse_governed_schema_migration_worker_v1",
        "created_at_utc": utc_now(),
        "mode": args.mode,
        "target_revision": args.target,
        "from_revision": spec["from_revision"],
        "result": "FAIL",
        "database_mutation": "NONE",
        "canonical_data_mutation_evidence": "NONE",
    }

    try:
        if args.mode == "apply" and args.confirmation != expected_confirmation(spec):
            raise RuntimeError("Worker apply confirmation mismatch.")

        app_root = find_app_root()
        result["migration_sha256"] = validate_source(app_root, spec)

        before = read_db_state(spec)
        result["before"] = before
        pre_state = classify_pre_state(before, spec)
        result["pre_state"] = pre_state

        if args.mode == "preflight":
            result["result"] = "PASS"
            result["database_mutation"] = "NONE"
            emit_worker(result)
            return 0

        if pre_state == "already_applied":
            result["result"] = "PASS"
            result["database_mutation"] = "NONE_ALREADY_AT_TARGET"
            result["after"] = before
            emit_worker(result)
            return 0

        upgrade = run_exact_alembic(app_root, spec)
        result["alembic_upgrade"] = upgrade
        result["database_mutation"] = (
            f"ATTEMPTED_EXACT_SCHEMA_MIGRATION_"
            f"{spec['from_revision']}_TO_{spec['target_revision']}"
        )
        if upgrade["returncode"] != 0:
            raise RuntimeError(
                "Exact Alembic upgrade failed: "
                + upgrade["stderr_tail"][-3000:]
            )

        after = read_db_state(spec)
        result["after"] = after
        verify_post_state(before, after, spec)

        result["database_mutation"] = (
            f"APPLIED_EXACT_SCHEMA_MIGRATION_"
            f"{spec['from_revision']}_TO_{spec['target_revision']}"
        )
        result["canonical_data_mutation_evidence"] = "NONE"
        result["result"] = "PASS"
        emit_worker(result)
        return 0

    except Exception as exc:
        result["error"] = str(exc)
        emit_worker(result)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Governed ElectionPulse production schema migration runner."
    )
    parser.add_argument(
        "--role",
        choices=("controller", "worker"),
        default="controller",
    )
    parser.add_argument(
        "--mode",
        choices=("preflight", "apply"),
        required=True,
    )
    parser.add_argument("--target", required=True)
    parser.add_argument("--confirmation", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.role == "worker":
        return worker_main(args)
    return controller_main(args)


if __name__ == "__main__":
    raise SystemExit(main())

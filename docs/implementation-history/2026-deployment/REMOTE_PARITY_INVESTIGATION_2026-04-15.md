# Remote Parity Investigation - 2026-04-15

## Scope

Investigate why `https://electionpulse.org/api/integrity_signal` returns `405 Method Not Allowed` while local code registers the route as POST and local tests pass.

## Verified Facts

- Local smoke suite: 6/6 passed.
- Local stress suite: 180/180 requests passed, 0 failures, 0 5xx.
- Remote compare against `https://electionpulse.org`:
  - 5/6 checks passed.
  - `GET /api/data_framework/preview?mode=active` returned `403` remotely; treated as expected auth variance.
  - `POST /api/integrity_signal` returned `405` remotely.
- Remote stress suite:
  - 120 total requests.
  - 20 failures, all from `POST /api/integrity_signal` returning `405`.
  - 0 server `5xx` responses.

## Code Path Verification

### Local route registration path

Azure/local production container starts Gunicorn with:

- `.Dockerfile` -> `gunicorn --config gunicorn.conf.py webapp.Smart_Elections_Parser_Webapp:app`

The Flask app registers the observability blueprint unconditionally in:

- `webapp/Smart_Elections_Parser_Webapp.py`
  - `app.register_blueprint(create_observability_blueprint())`

The observability blueprint declares:

- `webapp/parser/routes/observability_blueprint.py`
  - `/api/integrity_signal` with `methods=["POST"]`

The handler map binds the endpoint to:

- `webapp/Smart_Elections_Parser_Webapp.py`
  - `app.config["_OBSERVABILITY_ROUTE_HANDLERS"]["api_integrity_signal"] = api_integrity_signal`

## Interpretation

Because the deployed container startup path and local startup path are the same module/app object, the remote `405` is unlikely to be caused by the current repo structure itself.

Most plausible causes:

1. Azure is still serving an older image or stale deployment revision.
2. The deployed startup path is not actually using the current container/image tag expected by the workflow.
3. An upstream gateway/proxy/WAF rule is blocking or rewriting POST to `/api/integrity_signal`.
4. A deployment-specific registration/import failure occurs before or during observability blueprint setup, leaving a different handler surface active.

## Regression Protection Added

A local regression test now asserts route methods explicitly:

- `/api/integrity_trends` is GET-only.
- `/api/integrity_signal` is POST-only.

Test file:

- `webapp/tests/test_integrity_api_structure.py`

## Commit Readiness

### Ready now

- Session/auth hardening
- Data Framework fetch retry and auth-aware resilience
- Certificate welcome UX fixes
- Smoke testing utility
- Local-vs-remote compare utility
- Concurrent API stress testing utility
- TS config cleanup and JS typing fixes
- Integrity route contract regression test

### Not fully closed

- Remote deployment parity for `POST /api/integrity_signal`

## Deployment Follow-up Checklist

1. Confirm Azure Web App is running the latest pushed image digest/tag.
2. Check Azure App Service container startup logs for the line:
   - `Observability routes blueprint registered`
3. Confirm the container command is still:
   - `gunicorn --config gunicorn.conf.py webapp.Smart_Elections_Parser_Webapp:app`
4. Probe remote route methods from the deployed environment or App Service console:
   - `curl -i -X POST https://electionpulse.org/api/integrity_signal`
   - `curl -i -X GET https://electionpulse.org/api/integrity_signal`
5. Inspect any reverse proxy/WAF/front-door rules for method restrictions on `/api/*` or `/api/integrity_signal`.
6. After deployment confirmation, rerun:
   - `npm run smoke:api:compare -- --azure-base-url https://electionpulse.org`
7. Expect parity outcome:
   - `POST /api/integrity_signal` should return one of the app-defined responses, not `405`.

## Suggested Merge Note

Local code is stable and tested. Remote parity issue appears isolated to deployment/runtime handling of `/api/integrity_signal` and should be tracked as a deployment follow-up, not as a blocker on local code correctness.

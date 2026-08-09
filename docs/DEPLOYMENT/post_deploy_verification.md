---
layout: default
title: Post-Deployment Verification
---

# Post-Deployment Verification

Post-deployment verification determines whether a newly deployed Election Pulse
application is reachable, configured as intended, and healthy enough to remain
in service.

Deployment success and application health are separate conditions.

A workflow completing its deployment step does not by itself prove that the
application is operating correctly.

## Verification boundary

The verification sequence begins after application deployment:

```text
deployment completed
    |
    v
reachability
    |
    v
transport and HTTPS checks
    |
    v
application health
    |
    v
security configuration
    |
    v
critical runtime behavior
    |
    v
accept or investigate
```

The active deployment workflow remains authoritative for checks performed
automatically.

This document defines the durable verification expectations.

## 1. Confirm deployment completion

Identify the exact deployment being verified.

Record or retain:

- workflow run;
- source revision;
- deployment timestamp;
- target environment;
- application/container version where available.

Verification results should be traceable to a specific deployment.

## 2. Confirm public reachability

The live Election Pulse application should be reachable through its intended
public URL.

A successful reachability check should establish that:

- DNS resolves as intended;
- the expected HTTPS endpoint responds;
- the response comes from the intended application surface;
- obvious redirect loops are absent.

GitHub Pages must not be mistaken for the live application during this check.

## 3. Verify HTTPS behavior

Production traffic should use HTTPS.

Checks should confirm that the deployed surface behaves according to current
transport-security policy.

A redirect response may be acceptable when it intentionally moves HTTP traffic
to HTTPS.

Verification should distinguish expected redirects from routing failures.

## 4. Verify application health

A reachable page is not sufficient proof of application health.

Where supported by the current application, verify health or status endpoints
used by the deployment workflow.

Health checks should be lightweight and should not mutate election data.

Useful health evidence may include:

- successful application response;
- expected status code;
- dependency readiness where exposed safely;
- startup completion;
- absence of repeated fatal errors.

Health endpoints must not expose secrets or sensitive internal state.

## 5. Verify deployed configuration

Confirm security and runtime configuration that materially affects production
behavior.

Examples may include:

```text
CSP_MODE
ALLOW_STYLE_ATTR
```

The verification goal is to confirm intended behavior, not to print secret
configuration values.

Configuration checks should use safe observable behavior or redacted metadata
where possible.

## 6. Verify Content Security Policy

Production CSP behavior should match the deployment security model.

Current deployment policy uses strict CSP configuration.

Verification should confirm that:

- the CSP header is present where expected;
- the deployed policy matches the intended mode;
- required local assets load;
- unexpected CSP violations are investigated;
- temporary relaxed behavior is not silently left enabled in production.

See:

- [CSP model](security/csp_model.md)
- [CSP deployment checklist](security/csp_deployment_checklist.md)

## 7. Verify critical application behavior

Post-deployment verification should exercise a small set of non-destructive,
high-value behaviors.

Examples may include:

- loading the primary application surface;
- loading expected static assets;
- confirming required client-side initialization;
- checking a safe read-only API or health endpoint;
- confirming authentication boundaries render correctly.

The verification suite should avoid creating or modifying election records
unless the test environment explicitly supports that operation.

## 8. Verify logs

Review deployment and startup logs for failures that may not be visible from a
basic HTTP check.

Look for conditions such as:

- startup exceptions;
- repeated dependency failures;
- container restart loops;
- configuration parsing errors;
- authentication initialization failures;
- security-policy errors.

Logs should be reviewed using appropriate redaction.

Sensitive values must not be copied into public reports.

## 9. Verify database dependency carefully

If the deployed application requires a database connection, verification should
confirm dependency readiness without exposing credentials.

A database connectivity check should distinguish:

```text
application reachable
database reachable
database schema ready
data operation safe
```

These are separate conditions.

A database mutation should not be used as a generic health test unless the
operation is explicitly designed to be safe and reversible.

## 10. Verify documentation separately

GitHub Pages has its own delivery and routing checks.

Documentation deployment success does not prove live-application health, and
live-application health does not prove documentation publication success.

The two surfaces should be monitored separately.

## Failure classification

A failed verification should identify the failing layer where possible.

Useful categories include:

```text
BUILD
CONTAINER
REGISTRY
AZURE_DEPLOYMENT
ROUTING
HTTPS
APPLICATION_STARTUP
SECURITY_CONFIGURATION
DEPENDENCY
DATABASE
AUTHENTICATION
POST_DEPLOY_CHECK
```

A category identifies the failing boundary.

It does not automatically identify root cause.

## Response to failure

When a critical verification fails:

1. preserve the workflow and deployment evidence;
2. avoid additional unrelated changes;
3. identify whether the failure is build, configuration, runtime, or dependency
   related;
4. determine whether rollback or corrective deployment is safer;
5. verify again after remediation.

Do not treat a failed post-deployment check as successful merely because the
deployment command itself returned success.

## Verification evidence

Useful verification evidence may include:

- workflow run identifier;
- source revision;
- checked URL or endpoint;
- status code;
- timestamp;
- check name;
- pass/fail result;
- redacted diagnostic message.

This evidence can later support automated deployment health reporting.

## Relationship to the verification gate

The repository documentation gate:

```powershell
& .\scripts\maintenance\verification_gate.ps1
```

validates repository documentation quality.

It does not replace runtime post-deployment verification.

The two gates operate at different boundaries:

```text
verification_gate.ps1
    -> repository/documentation integrity

post-deployment verification
    -> deployed application integrity
```

## Related documentation

- [Application deployment](deployment.md)
- [CI/CD](ci_cd.md)
- [Deployment security](security/deployment_security.md)
- [Election operations](election_operations.md)

## Invariants

1. deployment completion is not equivalent to application health.
2. verification results are tied to a specific deployment.
3. health checks avoid unsafe data mutation.
4. security configuration is verified without exposing secrets.
5. GitHub Pages and the live application are verified separately.
6. failures retain enough evidence for diagnosis.
7. critical failed checks are not silently ignored.
8. rollback or remediation is followed by verification again.

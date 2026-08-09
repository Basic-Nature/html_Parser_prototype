---
layout: default
title: CSP Deployment Checklist
---

# CSP Deployment Checklist

Use this checklist when changing or verifying Content Security Policy behavior
for the live Election Pulse application.

The checklist intentionally avoids hard-coding transient library versions.

The application code and active deployment workflow remain authoritative.

## Before deployment

Confirm the intended environment and policy mode.

- [ ] Identify the deployment revision.
- [ ] Confirm the target environment.
- [ ] Confirm required local browser assets are present.
- [ ] Confirm CSP-related application settings are understood.
- [ ] Confirm the active workflow still applies the intended production mode.
- [ ] Confirm no secret values are being added to documentation or logs.

## Verify repository policy

Review current implementation sources:

```text
webapp/
.github/workflows/main_ballotlens.yml
```

Confirm that documentation still matches the behavior implemented there.

## Production mode

For the current Azure production path:

- [ ] `CSP_MODE` is configured for strict behavior.
- [ ] any style-attribute allowance is intentional.
- [ ] temporary compatibility changes are documented.
- [ ] broad external origins have a concrete requirement.

## Deploy

Deploy through the normal application workflow.

Do not treat manual changes in the Azure portal as permanent configuration
unless the workflow or documented deployment process is also updated.

Configuration drift between portal state and repository-defined deployment
behavior should be resolved explicitly.

## Verify response headers

After deployment:

- [ ] request the live application over HTTPS;
- [ ] confirm a Content Security Policy header is present where expected;
- [ ] confirm the observed policy matches the intended deployment mode;
- [ ] confirm unexpected external origins are absent;
- [ ] confirm required connection behavior remains available.

Do not expose authentication tokens or private headers in public reports.

## Verify browser behavior

Using the deployed application:

- [ ] load the primary interface;
- [ ] inspect the browser console for CSP violations;
- [ ] confirm required JavaScript executes;
- [ ] confirm required styles render;
- [ ] confirm critical interactive behavior;
- [ ] confirm required network connections succeed.

A visually loaded page is not sufficient if browser security errors prevent
critical functionality.

## Investigate violations

For each unexpected CSP violation:

1. identify the blocked resource or behavior;
2. determine whether it is required;
3. determine whether it can be served locally;
4. determine the narrowest safe policy change;
5. change application or deployment configuration;
6. deploy again;
7. verify again.

Do not solve violations by broadly relaxing the policy without understanding the
resource.

## Temporary relaxed mode

If relaxed behavior is required temporarily:

- [ ] record why strict mode cannot currently be used;
- [ ] limit the relaxation to the affected environment;
- [ ] identify the remediation path;
- [ ] restore strict production policy after remediation;
- [ ] rerun post-deployment verification.

Relaxed behavior must not silently become the new baseline.

## Style attributes

If `ALLOW_STYLE_ATTR` is changed:

- [ ] identify the UI requirement;
- [ ] verify whether the style can be moved to a stylesheet;
- [ ] confirm the resulting CSP header;
- [ ] check the browser console;
- [ ] restore the stricter setting when practical.

## Deployment failure

If the application becomes unhealthy after a CSP change:

1. preserve the deployment and browser evidence;
2. determine whether CSP is actually the failing layer;
3. use a known-safe configuration if recovery is required;
4. redeploy;
5. rerun verification.

Do not leave an emergency configuration undocumented.

## Post-deployment record

Useful CSP verification evidence includes:

```text
deployment revision
environment
timestamp
observed mode
header present
browser violations
critical UI status
review result
```

Do not store secret header values or credentials.

## Related documentation

- [CSP model](csp_model.md)
- [Deployment security model](deployment_security.md)
- [Application deployment](../deployment.md)
- [Post-deployment verification](../post_deploy_verification.md)

## Invariants

1. CSP changes are verified in the deployed application.
2. strict production policy is restored after temporary relaxation.
3. broad policy changes require a concrete reason.
4. browser behavior and response headers are both checked.
5. CSP failures do not justify exposing credentials.
6. emergency configuration does not silently become permanent policy.

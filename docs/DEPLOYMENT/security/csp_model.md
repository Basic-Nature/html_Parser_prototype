---
layout: default
title: Content Security Policy Model
---

# Content Security Policy Model

Election Pulse uses Content Security Policy as a browser-side defense against
unexpected script, style, and resource execution.

The application code is authoritative for the exact generated policy.

Deployment configuration determines which supported mode is used in a given
environment.

## Policy roles

CSP helps constrain browser behavior for resources such as:

- scripts;
- styles;
- images;
- fonts;
- connections;
- frames;
- forms;
- objects.

CSP does not replace:

- server-side authorization;
- authentication;
- input validation;
- output encoding;
- secret management;
- runtime isolation.

## Supported configuration

Current application code recognizes deployment configuration including:

```text
CSP_MODE
ALLOW_STYLE_ATTR
```

Repository evidence shows that application fallback behavior and production
deployment policy are intentionally distinguishable.

## Application fallback

The application currently contains fallback behavior in which `CSP_MODE` may
resolve to a relaxed mode when production configuration is not supplied.

This is application behavior.

It is not a statement that relaxed mode is the desired production policy.

## Production policy

The active Azure deployment workflow explicitly applies:

```text
CSP_MODE=STRICT
```

Therefore:

```text
application fallback
!=
production deployment policy
```

Current production documentation should describe strict mode as the intended
Azure deployment configuration while preserving the existence of application
fallback behavior.

## Inline style attributes

The application also reads:

```text
ALLOW_STYLE_ATTR
```

Current code defaults this control to a disabled value unless explicitly
enabled.

This setting affects whether style attributes receive additional allowance in
the generated policy.

The application code remains authoritative for exact directive construction.

## Strict mode

Strict mode should prefer resources that are intentionally part of the deployed
application.

A strict policy should minimize unnecessary external execution sources.

Any relaxation should have a concrete compatibility reason.

## Relaxed mode

Relaxed mode exists as compatibility or fallback behavior.

It should not silently become the permanent production configuration.

If relaxed mode is required temporarily:

1. document the reason;
2. identify the blocked dependency;
3. limit the change;
4. restore strict behavior after remediation;
5. verify the deployed policy again.

## Local assets

Where the application vendors required browser assets locally, CSP policy should
support those local resources without requiring unnecessary external hosts.

Documentation should not hard-code library versions unless the current
deployment requires that exact version as an operational contract.

## Connections

CSP connection policy must support the network behavior actually required by
the deployed application.

Unnecessary connection destinations should not be added preemptively.

If WebSocket or other live connections are required, the generated policy
should be verified against the deployed behavior.

## Nonces and dynamic script policy

Where application code uses request-specific script authorization such as
nonces, the exact implementation should be verified in the runtime code.

Documentation should not assume that a nonce exists merely because a historical
policy used one.

## Verification

CSP should be verified after deployment through observable response behavior.

Verification may include:

- inspecting the CSP response header;
- checking browser console violations;
- confirming required assets load;
- confirming critical UI behavior;
- confirming temporary relaxed configuration is not left enabled.

See:

- [CSP deployment checklist](csp_deployment_checklist.md)
- [Post-deployment verification](../post_deploy_verification.md)

## Change policy

Before changing CSP:

1. identify the blocked resource or behavior;
2. determine whether the resource should be local;
3. prefer the narrowest required directive change;
4. avoid adding broad external origins without evidence;
5. verify both security header and application behavior;
6. record material production-policy changes.

## Invariants

1. application CSP generation is the implementation authority.
2. production workflow policy is distinguished from application fallback.
3. strict mode is the intended Azure production configuration.
4. relaxed mode is not silently treated as permanent production policy.
5. CSP does not replace server-side security controls.
6. policy relaxation is narrow and evidence-driven.
7. deployed CSP behavior is verified after change.

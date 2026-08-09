---
layout: default
title: Implemented Contracts
---

# Implemented Contracts

This document summarizes implementation contracts that should remain verifiable
against source code and tests.

## Output finalization

Structured election output should pass through:

```python
finalize_election_output(headers, rows, metadata)
```

Handlers should not invent incompatible final CSV schemas.

## Smart Elections row structure

The default result uses one precinct per row and preserves candidate vote-method
columns across precincts.

Zero-vote candidates and methods remain represented.

## Context persistence

Runtime parser evidence must not be implicitly persisted as learned context.

Context writes should pass through explicit persistence/write policy.

## Browser behavior

Reusable DOM/browser behavior should use shared browser infrastructure rather
than being duplicated in every jurisdiction handler.

## Logging

Code should use shared logging/event infrastructure where runtime routing
matters.

Web-session output must remain session-scoped.

## Repository state

Runtime locks, telemetry, generated output, caches, and local databases should
not be committed unless an explicit fixture/reference contract requires them.

## Documentation authority

Generated reports under `docs/DEVELOPMENT/generated/` provide evidence but do
not override architecture or CORE contracts.

## Validation

Parser validation should expose unreconciled state rather than manufacture
agreement.

Important checks include candidate vote-method reconciliation, precinct
reconciliation where comparable totals exist, duplicate precinct detection,
missing vote methods, and candidate/method completeness.

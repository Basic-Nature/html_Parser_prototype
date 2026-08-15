---
layout: default
title: Confidence Authority
---

# Confidence Authority

This document defines the boundary between **evidence measurement** and
**decision authority** in ElectionPulse.

## Status

This is a Phase 1 architecture contract.

The contract defines target ownership without claiming that every existing
runtime branch has already migrated. Legacy local thresholds remain present and
will be moved incrementally after characterization and correctness work.

## Core boundary

The governing question is:

> Which code is allowed to measure evidence, and which code is allowed to turn
> that evidence into a decision?

ElectionPulse separates those responsibilities.

```text
domain measurement
        |
        v
typed / attributable evidence
        |
        v
risk_gates.py
current normalized state
        |
        v
risk_gates_calculus.py
trajectory / boundary / convergence
        |
        v
domain action policy
review / quarantine / continue / promote
```

A local algorithm may need thresholds for search, ranking, anomaly detection,
OCR, fuzzy matching, or candidate pruning. Those algorithmic thresholds do not
automatically become authority to declare election truth.

## Central state authority

`webapp/parser/health/risk_gates.py` is the central **current-state evaluator**.

It owns the normalized state dimensions:

- confidence;
- verification;
- anomaly;

and the resulting composite suspicion and primary risk tier.

It does not determine whether a candidate identity, source, office, county, or
other domain fact is true merely because a local score crossed a threshold.

## Equal gate weights and policy boundaries

The approximate one-third gate weights and the tier boundaries describe
different parts of the model.

The default gate weights (`0.33 / 0.33 / 0.34`) provide an approximately equal
three-dimension baseline for confidence, verification, and anomaly.

The current `0.45` and `0.72` suspicion values are **policy boundaries over the
resulting state**, not literal one-third partitions of the unit interval. They
may be calibrated independently from the gate weights as evidence about desired
workflow behavior improves.

This distinction preserves the useful one-third intuition without presenting
the current asymmetric LOG/WARN/BLOCK regions as equal mathematical thirds.

## Trajectory and convergence authority

`webapp/parser/health/risk_gates_calculus.py` is the
**trajectory/convergence evaluator**.

It extends normalized state with rate of change, boundary approach,
instability, convergence, and diminishing informational return.

It does not replace the base state evaluator and does not independently
canonicalize domain values.

ElectionPulse treats convergence as a computational boundary. Iterative
evaluation continues while new evidence materially changes the risk state. As
successive state changes approach zero, the expected informational benefit of
further evaluation also approaches zero. Processing may stop when change falls
below a defined convergence tolerance, reaches an iteration/cost budget, or
cannot improve with the available evidence.

The exact convergence implementation remains subject to later correctness and
calibration work.

## Evidence producers

Domain-specific code should increasingly behave as an evidence producer.

Examples include fuzzy name similarity, FEC candidate lookup, party/state/
office/cycle agreement, parser extraction quality, header or table similarity,
ML probabilities, DL1/DL2 comparison, anomaly statistics, source provenance,
and historical success.

A local score can remain useful for retrieval, ranking, filtering, or
diagnostics without having final decision authority.

## Official election sources and third-party services

**Official source is a provenance relationship, not a TLD classification.**

A `.gov` domain is strong evidence that a source is operated within a
government namespace, but `.gov` is neither required nor sufficient for every
official election artifact.

Election jurisdictions may publish results through third-party election
services or vendor-hosted result systems. A non-`.gov` source can therefore
carry official-source evidence when its delegation is established through
durable provenance, for example:

- an official jurisdiction page links to the vendor-hosted result;
- an official election office identifies the service as its publication path;
- a verified source registry records the jurisdiction-to-service relationship;
- an independently preserved source artifact establishes that relationship.

Vendor reputation alone is not equivalent to jurisdictional delegation.

The system should preserve the distinction between:

```text
first-party official source
officially delegated third-party source
secondary/reference source
unknown source
```

This taxonomy is architectural guidance in Phase 1; it is not yet a new runtime
schema.

## Security constraints are non-compensatory

Source provenance and security are different dimensions.

A source may be officially delegated and still fail a security constraint.

Examples include authorization failure, a prohibited private-network/SSRF
target, a known phishing or malware condition, a principal-isolation violation,
or an explicit provenance-integrity failure where required provenance cannot be
established.

A favorable confidence, `.gov`, vendor, or historical-success signal must not
mathematically cancel a hard constraint.

Likewise, an unfamiliar third-party domain should not be rejected merely
because it lacks `.gov` when official delegation can be established and
security constraints pass.

## Promotion boundary

Evidence evaluation is not finalization.

A domain component may produce strong evidence. `risk_gates.py` may classify
the resulting state. `risk_gates_calculus.py` may characterize trajectory.
Durable promotion to verified/canonical election truth remains a downstream
governance and domain-policy decision with provenance preserved.

## Compatibility during migration

Current code still contains independent scalar thresholds.

During migration they are classified as:

```text
EVIDENCE_ONLY
CENTRAL_POLICY
HARD_CONSTRAINT
RETAIN_LOCAL_ALGORITHM
PRESENTATION_ONLY
DOCUMENTATION_ONLY
```

A threshold is migrated based on **consequence**, not merely its numeric value.

For example, a fuzzy threshold used to prune a search list may remain local. A
fuzzy threshold used to declare a candidate identity authoritative must be
migrated into the evidence-to-policy path.

## Phase 1 invariants

1. `risk_gates.py` owns normalized current-state evaluation.
2. `risk_gates_calculus.py` owns trajectory/convergence evaluation.
3. Domain-specific scores are evidence unless a documented policy explicitly
   grants them a narrower operational role.
4. Local algorithm thresholds do not independently promote verified/canonical
   election truth.
5. Official-source evidence is provenance-based, not `.gov`-only.
6. Officially delegated third-party sources are representable without weakening
   security constraints.
7. Hard security/authorization constraints are not averaged away.
8. Confidence is uncertainty, not truth.
9. Verification remains distinct from confidence.
10. Phase 1 changes this contract and its tests, not runtime scoring behavior.

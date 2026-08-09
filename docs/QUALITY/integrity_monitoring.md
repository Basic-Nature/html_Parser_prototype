---
layout: default
title: Integrity Monitoring
---

# Integrity Monitoring

Integrity monitoring identifies changes, anomalies, and inconsistencies that may
require additional evidence or review.

An integrity signal is not proof of wrongdoing or proof that election data is
incorrect.

## Responsibilities

Integrity monitoring may evaluate:

- unexpected vote-method distributions;
- large precinct-to-precinct ratio changes;
- repeated or reused rows;
- source structure changes;
- parser behavior drift;
- reconciliation failures;
- unusual missing-data patterns;
- changes in source metadata;
- discrepancies between independently available representations.

## Signal model

```text
observation
    |
    v
integrity signal
    |
    +-- expected variation
    |
    +-- parser/source drift
    |
    +-- data discrepancy
    |
    `-- unresolved anomaly
            |
            v
          review
```

A signal identifies something worth examining.

It does not determine the explanation.

## Drift

Drift may occur when:

a vendor changes HTML;
a table layout changes;
labels change;
vote methods change;
navigation behavior changes;
a source changes download formats;
OCR characteristics change.

Parser drift should be distinguished from election-data anomalies whenever
possible.

## Election-result anomalies

Election Pulse may flag patterns such as:

absentee or other method concentration;
abrupt candidate-ratio changes;
repeated rows under different precinct names;
totals that fail reconciliation.

Thresholds should be configurable and explainable.

A statistical outlier is evidence for review, not a conclusion.

## Review escalation

Integrity signals may lead to:

record
-> collect evidence
-> compare
-> review
-> resolve / quarantine / retain discrepancy

The review result should retain provenance.

## Relationship to context

A source-specific integrity observation must not become a global parser rule
without explicit review and scope.

Approved recurring patterns may be promoted into learned context.

## Relationship to Ballot Lens

Ballot Lens may present integrity signals, comparisons, maps, and supporting
evidence.

Presentation does not determine the underlying integrity decision.

## Audit requirements

A durable integrity finding should be able to explain:

what triggered the signal;
which data was evaluated;
which source supplied the data;
which threshold or rule was used;
what review occurred;
what final disposition was chosen.

## Invariants

anomaly is not proof;
integrity monitoring preserves supporting evidence;
parser drift and election-data anomalies are distinguished where possible;
thresholds remain explicit;
signals do not silently rewrite canonical values;
review outcomes retain provenance.

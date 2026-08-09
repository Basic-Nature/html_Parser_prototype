---
layout: default
title: Quality and Integrity
---

# Quality and Integrity

The QUALITY domain defines how Election Pulse evaluates uncertainty,
consistency, integrity signals, quarantine state, and machine-assisted quality
observations.

Quality controls do not create election truth.

They evaluate evidence and canonical candidates for promotion, publication,
review, or quarantine.

## Quality documents

- [Verification](verification.md)
- [Confidence framework](confidence_framework.md)
- [Integrity monitoring](integrity_monitoring.md)
- [Quarantine system](quarantine_system.md)
- [ML quality](ml_quality.md)

## Quality model

```text
source evidence
    |
    v
extraction
    |
    v
normalization
    |
    v
verification
    |
    +-- reconciled --------------------+
    |                                  |
    +-- uncertain -> confidence -------+
    |                                  |
    +-- anomalous -> integrity --------+--> review / promotion
    |                                  |
    +-- unsafe -> quarantine ----------+
    |                                  |
    +-- ML signals --------------------+
```

No quality subsystem may silently rewrite an election value merely to satisfy a
validation rule.

## Authority

QUALITY documents define durable quality and review boundaries.

Implementation-specific benchmarks, dated readiness claims, migration plans,
and completed integration summaries belong in implementation history.

## Core invariants

evidence is not canonical truth;
missing data is not equivalent to zero;
a confidence score is not a verification decision;
anomaly detection is not proof of error;
quarantine preserves evidence rather than destroying it;
ML may assist review but does not independently promote canonical election
records;
discrepancies remain visible until explicitly resolved.

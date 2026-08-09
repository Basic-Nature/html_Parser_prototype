---
layout: default
title: Verification
---

# Verification

Verification determines whether extracted and normalized election data satisfy
the consistency requirements needed for downstream use.

Verification is evidence evaluation, not value fabrication.

## Verification responsibilities

Verification may evaluate:

- candidate vote-method totals;
- reported candidate totals;
- precinct contest totals;
- ballot-method totals;
- duplicate precinct identity;
- missing candidate or vote-method data;
- reporting completeness;
- parser structural expectations;
- source and artifact provenance.

## Candidate reconciliation

When vote methods form a complete decomposition:

```text
candidate total
=
sum(candidate vote-method values)
```

If the source uses different semantics, the system must retain that distinction
rather than force reconciliation.

## Precinct reconciliation

Where comparable source totals exist:

```text
sum(candidate votes)
=
reported precinct contest total
```

A mismatch becomes a discrepancy.

It does not authorize Election Pulse to manufacture a replacement value.

## Vote-method reconciliation

Where the source publishes method totals, candidate-level method values should
reconcile with those totals according to source semantics.

Additional methods must not be discarded simply because they are not part of
the default Smart Elections method family.

## Completeness

Verification must distinguish:

zero

from:

missing

A zero represents an observed or validated zero vote count.

Missing means the value could not be established.

## Duplicate detection

Duplicate precinct rows must be detected before finalization.

Potential duplicates should retain enough evidence to determine whether they
represent:

repeated source content;
page-spanning reconstruction;
naming normalization;
genuinely distinct precinct records.

## Verification outcome

Useful outcome states include:

verified
discrepant
incomplete
needs-review
quarantined

The precise runtime representation may evolve, but unresolved state must remain
explicit.

## Relationship to confidence

Verification answers whether defined consistency checks pass.

Confidence estimates uncertainty.

A high-confidence parse can still fail reconciliation.

A low-confidence observation can still be factually correct.

The two concepts must remain separate.

## Relationship to finalization

Finalization should preserve verification metadata when producing Smart
Elections output through:

```text
finalize_election_output(headers, rows, metadata)
```

## Invariants

verification never silently alters votes;
discrepancies remain inspectable;
source semantics outrank assumptions about expected totals;
missing is not rewritten as zero;
duplicate detection occurs before authoritative finalization;
verification results retain provenance to their supporting evidence.

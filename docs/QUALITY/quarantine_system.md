---
layout: default
title: Quarantine System
---

# Quarantine System

Quarantine isolates data or artifacts that should not proceed through normal
promotion or publication until their uncertainty is resolved.

Quarantine is a preservation mechanism, not a deletion mechanism.

## Reasons for quarantine

A record or artifact may be quarantined because of:

- unresolved reconciliation failure;
- malformed source structure;
- ambiguous OCR;
- duplicate identity;
- missing critical evidence;
- unsafe or unsupported file behavior;
- parser failure;
- suspicious source transition;
- review-policy requirements.

## Quarantine record

A useful quarantine record should preserve:

```text
item identity
reason code
source provenance
supporting evidence
parser/run context
timestamp
review state
```

The original evidence should remain available when retention policy permits.

## Lifecycle

```text
detected
    |
    v
quarantined
    |
    +-- additional evidence
    |
    +-- retry / alternate parser
    |
    +-- human review
    |
    +-- corrected and approved
    |
    `-- rejected / retained unresolved
```

Release from quarantine must be explicit.

## Reason codes

Reason codes should be machine-readable where practical.

Examples may include:

RECONCILIATION_FAILED
DUPLICATE_PRECINCT
MISSING_REQUIRED_EVIDENCE
OCR_AMBIGUOUS
UNSUPPORTED_STRUCTURE
SOURCE_CHANGED
REVIEW_REQUIRED

Reason codes should describe the condition without asserting an unsupported
cause.

## Quarantine and canonical data

Quarantined evidence must not silently enter canonical election output.

If a previously canonical record is later questioned, the system should retain
the audit relationship between the record, evidence, and review action.

## Quarantine and security

Security isolation and election-quality quarantine may share infrastructure but
are not identical concepts.

A suspicious file may require security isolation.

A valid file with unresolved election totals may require quality quarantine.

The reason must remain explicit.

## Recovery

Recovery may involve:

parser correction;
alternate extraction;
additional source acquisition;
manual review;
normalization correction;
evidence comparison.

Recovered data still passes normal verification and promotion policy.

## Invariants

quarantine preserves evidence;
quarantine does not imply wrongdoing;
reason codes are explicit;
release is deliberate;
quarantined data cannot silently bypass verification;
security and data-quality causes remain distinguishable.

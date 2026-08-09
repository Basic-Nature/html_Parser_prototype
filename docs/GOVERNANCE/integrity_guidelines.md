---
layout: default
title: Integrity Guidelines
---

# Integrity Guidelines

Election Pulse uses integrity signals to identify election data that may require
additional evidence, comparison, or review.

An integrity signal is not a finding of misconduct.

## Core principle

The system must distinguish:

```text
unusual
!=
incorrect
!=
fraudulent
```

Statistical variation, parser drift, source changes, reporting differences, and
legitimate election administration differences can all produce unusual data.

## Responsible interpretation

Integrity findings should be described using evidence-supported language.

Prefer:

discrepancy;
anomaly;
unresolved difference;
source change;
parser drift;
reconciliation failure;
review required.

Avoid unsupported conclusions about intent or cause.

## Evidence requirements

An integrity review should preserve the evidence necessary to understand the
signal.

That may include:

source records;
election and jurisdiction identity;
precinct identity;
vote-method values;
comparison records;
parser output;
normalization decisions;
integrity rules or thresholds;
timestamps;
review actions.

## Statistical signals

Statistical signals may help prioritize investigation.

Examples include:

unusual vote-method concentration;
abrupt ratio changes;
repeated rows;
unexpected missing values;
large differences between related contests;
deviations from nearby precinct patterns.

A statistical signal identifies an observation worth examining.

It does not establish its cause.

## Source comparison

Where multiple representations exist, Election Pulse may compare them.

Examples may include:

official results
downloaded structured data
public HTML
PDF reports
cast-vote records where lawfully available

Differences should be documented before attempting reconciliation.

## Parser and source drift

Election Pulse must consider whether an anomaly was introduced by:

parser behavior;
DOM changes;
OCR;
source formatting;
normalization;
stale cached data;
incomplete reporting.

Technical causes should be evaluated before treating a discrepancy as an
election-result anomaly.

## Escalation

A typical escalation path is:

```text
signal
    |
    v
collect evidence
    |
    v
verification
    |
    v
comparison
    |
    v
review
    |
    +-- explained
    +-- unresolved
    +-- quarantined
    `-- approved correction
```

The final state should remain auditable.

## Corrections

Corrections must preserve:

original observation;
corrected interpretation;
supporting evidence;
scope;
review status;
provenance.

A correction should not silently replace its history.

## Public presentation

Ballot Lens or other interfaces may visualize:

discrepancies;
drop-off patterns;
maps;
comparisons;
quality signals;
evidence.

Presentation must preserve appropriate uncertainty.

A visualization should not transform an unresolved signal into a definitive
claim.

## Machine learning

Machine learning may assist:

anomaly detection;
classification;
review prioritization;
pattern discovery.

Model output remains evidence or recommendation.

It does not independently establish election integrity or misconduct.

## Privacy and security

Integrity analysis should collect only the data required for legitimate
election-data review.

Sensitive information, credentials, private voter information, or unrelated
personal data should not be exposed through public analysis artifacts.

## Invariants

anomaly is not proof;
discrepancy is not evidence of intent by itself;
source and parser drift are considered;
corrections preserve history;
uncertainty remains visible;
automated systems do not make unsupported accusations;
durable findings retain provenance.

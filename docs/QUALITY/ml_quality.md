---
layout: default
title: Machine Learning Quality
---

# Machine Learning Quality

Machine-learning components in Election Pulse assist parsing, classification,
pattern discovery, anomaly detection, and review prioritization.

They do not independently establish canonical election truth.

## Appropriate ML responsibilities

Machine learning may assist with:

- parser-path recommendation;
- source-pattern classification;
- anomaly scoring;
- normalization suggestions;
- review prioritization;
- OCR or structure interpretation;
- confidence estimation;
- retrieval of related approved context.

## Authority boundary

The durable rule is:

```text
ML output
    -> evidence
    -> validation / review
    -> explicit promotion
```

Model output is evidence-bearing advice or observation.

It is not an automatic canonical write.

## Training data

Training data should distinguish:

raw parser observations;
approved corrections;
canonical reference data;
synthetic or generated examples;
rejected examples.

Training should not silently treat every historical parser result as ground
truth.

## Provenance

Where ML influences a durable decision, retain useful provenance such as:

model or strategy identifier;
input evidence;
confidence;
recommendation;
review result;
final promoted value.

Exact metadata may vary by subsystem.

## Evaluation

Model quality should be evaluated against the task being performed.

Relevant measures may include:

precision;
recall;
false-positive rate;
false-negative rate;
calibration;
review acceptance rate;
parser success improvement.

A single model metric does not establish election-data correctness.

## Drift

ML behavior may drift because:

source structures change;
training distributions change;
vocabulary evolves;
jurisdiction-specific patterns differ;
model versions change.

Drift should trigger evaluation rather than silent retraining against
unreviewed runtime data.

## Active learning

Human-reviewed corrections may provide valuable future training evidence.

Promotion into training datasets should remain explicit and preserve scope.

## Context relationship

ML-generated observations may query context.

They may propose learned context.

They must not bypass the context write policy.

## Failure behavior

When an ML component is unavailable or uncertain, the parser should prefer:

deterministic fallback;
additional evidence;
explicit review;
partial result state;

rather than fabricate confidence or completion.

## Invariants

ML output is not canonical truth;
runtime observations are not automatically training labels;
approved corrections retain provenance;
model confidence does not replace verification;
retraining does not silently promote unreviewed parser evidence;
deterministic fallbacks remain available where practical.

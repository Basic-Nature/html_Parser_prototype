---
layout: default
title: Confidence Framework
---

# Confidence Framework

Confidence represents uncertainty about an observation or interpretation.

It is not a truth score and must not be treated as one.

## Purpose

Confidence can help Election Pulse prioritize:

- parser review;
- normalization review;
- source-pattern review;
- OCR review;
- anomaly investigation;
- human attention.

A confidence score may assist workflow decisions without becoming election
authority.

## Confidence inputs

Confidence may be informed by evidence such as:

- parser agreement;
- source structure;
- selector stability;
- OCR quality;
- normalization ambiguity;
- known source patterns;
- reconciliation results;
- model output;
- prior approved observations.

The exact scoring mechanism may vary by subsystem.

## Confidence and evidence

A confidence value must remain attached to the evidence or interpretation it
describes.

For example:

```json
{
  "raw_value": "Member of Assembly",
  "normalized_value": "Member of the Assembly",
  "confidence": 0.98,
  "review_status": "approved"
}
```

Confidence alone does not supply the approval.

## Confidence and verification

Confidence and verification answer different questions.

confidence:
How uncertain is this interpretation?

verification:
Does this result satisfy defined consistency checks?

Neither substitutes for the other.

## Confidence and source trust

Source identity is another separate dimension.

trusted source
!=
high-confidence extraction
!=
verified record

An official source can still be parsed incorrectly.

## Low-confidence behavior

Low confidence may trigger:

alternate extraction;
additional evidence collection;
review;
quarantine;
deferred promotion.

It should not trigger arbitrary replacement with a guessed value.

## High-confidence behavior

High confidence may reduce review priority when other checks also pass.

It must not bypass:

required verification;
provenance requirements;
promotion policy;
jurisdiction-specific safeguards.

## Promotion

Confidence-scored evidence may become learned context only through explicit
promotion.

The promotion decision should preserve:

evidence;
scope;
confidence;
review status;
provenance.

## Invariants

confidence is not truth;
confidence does not override source evidence;
confidence does not replace reconciliation;
uncertainty remains visible;
promotion requires policy beyond a numeric threshold;
confidence is scoped to the observation it describes.

## Decision authority boundary

Confidence and verification measurements are evidence inputs. They do not
independently promote a value to verified or canonical election truth.

The durable authority boundary is documented in
[Confidence Authority](../ARCHITECTURE/confidence_authority.md).

In particular:

- `risk_gates.py` owns normalized current-state evaluation;
- `risk_gates_calculus.py` owns trajectory and convergence analysis;
- domain-specific thresholds may remain local when they perform algorithmic
  ranking, pruning, or detection rather than truth promotion;
- official-source evidence is provenance-based rather than `.gov`-only, so an
  officially delegated third-party election service can contribute verified
  provenance;
- hard security and authorization constraints remain non-compensatory.

---
layout: default
title: Evidence Model
---

# Evidence Model

Election Pulse treats parser observations as evidence until they satisfy the
requirements for promotion into durable election data or learned context.

## Evidence is not knowledge

```text
source material
  -> parser observation
  -> evidence
  -> validation / review
  -> promotion decision
  -> durable knowledge
```

A parser seeing a string does not make it canonical.

A machine-learning score does not make it verified.

An official source URL does not make every extracted field correct.

## Evidence may include

- source URL or upload identity;
- downloaded artifacts;
- raw HTML;
- DOM fragments;
- API/JSON responses;
- CSV rows;
- PDF pages;
- OCR text;
- screenshots;
- selector observations;
- normalization decisions;
- discrepancy results;
- user corrections;
- review outcomes;
- runtime logs associated with a parse.

## Provenance

Evidence should remain traceable to its source.

A durable evidence record should be able to express concepts such as:

```json
{
  "source_type": "parser_observation",
  "source_url": "...",
  "session_id": "...",
  "jurisdiction": {
    "state": "New York",
    "county": "Rockland"
  },
  "observed_at": "...",
  "artifact_hash": "...",
  "parser_component": "...",
  "raw_value": "...",
  "normalized_value": "...",
  "confidence": 0.98,
  "review_status": "pending"
}
```

## Parser evidence

Parser evidence is temporary or source-specific information used to explain a
run.

Examples:

- DOM selectors;
- table geometry;
- page orientation;
- extracted labels;
- OCR regions;
- source-specific formatting observations.

Parser evidence must not be written directly into canonical or learned stores
without an explicit promotion path.

## Corrections

Corrections are evidence-bearing decisions.

A correction should preserve:

- prior value;
- corrected value;
- proposer;
- review status;
- jurisdiction scope;
- source evidence;
- confidence or rationale;
- timestamp.

Approved corrections may become learned context when they are generalizable.

## Source trust versus verification

```text
trusted source identity
    !=
verified parsed record
```

A trusted source can still be incomplete, stale, malformed, or misparsed.

## Discrepancies

A discrepancy is evidence and remains visible.

Examples:

- candidate methods do not equal candidate total;
- precinct totals do not reconcile;
- duplicate precinct identity;
- missing vote method;
- OCR ambiguity;
- incompatible totals.

Discrepancies may be flagged, quarantined, reviewed, retried, or routed to an
alternate parser path. They should not be silently rewritten.

## Audit integrity

Audit-ready output should be able to answer:

- What source was used?
- What parser path handled it?
- What raw value was observed?
- What normalization occurred?
- What validation succeeded or failed?
- Was a correction involved?
- What was promoted?

## Retention

Evidence does not need one universal retention period.

Transient evidence may be discarded when durable provenance exists elsewhere.
Evidence needed to support canonical records, corrections, or integrity
decisions should follow the retention policy of those responsibilities.

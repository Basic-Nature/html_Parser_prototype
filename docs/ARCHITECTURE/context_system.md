---
layout: default
title: Context System
---

# Context System

The context system provides reusable election-domain knowledge without treating
runtime parser evidence as durable knowledge.

## Why it changed

The original `context_library.json` accumulated several responsibilities:

- persistent reference data;
- parser observations;
- learned corrections;
- migration state;
- runtime watches;
- cleanup targets;
- lookup data.

As the project grew, one large JSON document became difficult to query and
ambiguous about authority.

## Context domains

```text
Context System
|-- canonical/
|   |-- jurisdictions
|   |-- contest vocabularies
|   |-- ballot-method mappings
|   |-- party aliases
|   `-- parser rules
|-- learned/
|   |-- approved corrections
|   |-- confidence-scored patterns
|   `-- source-specific observations
|-- runtime/
|   |-- migration state
|   |-- empty-entry watches
|   |-- telemetry
|   `-- temporary parser evidence
`-- indexes/
    |-- search index
    |-- embeddings
    `-- generated lookup caches
```

## Canonical context

Canonical context is reviewed, stable reference knowledge.

Examples include jurisdictions, contest terminology, vote-method concepts,
party aliases, and approved parser rules.

## Learned context

Learned context is promoted knowledge derived from reviewed evidence.

Example:

```json
{
  "type": "contest_alias",
  "value": "Member of Assembly",
  "canonical_value": "Member of the Assembly",
  "source": "manual_review",
  "confidence": 0.98,
  "status": "approved",
  "jurisdiction": {
    "state": "New York"
  },
  "created_at": "...",
  "provenance": {
    "session_id": "...",
    "source_url": "..."
  }
}
```

## Runtime context

Runtime context is operational state, not project knowledge.

Examples:

- migration checkpoints;
- empty-entry watches;
- telemetry;
- temporary parser observations;
- caches.

Runtime context generally belongs outside Git.

## Indexes

Indexes accelerate lookup but do not become authority.

Examples include search indexes, embeddings, candidate indexes, and caches.

Indexes should be reproducible from authoritative source data when practical.

## Promotion boundary

```text
runtime evidence
  -> review / validation
  -> explicit promotion
  -> learned or canonical context
```

Promotion should preserve provenance, status, scope, and confidence.

## Context write policy

Context writes should pass through an explicit write policy.

The policy should answer:

- What category is being written?
- Is approval required?
- What provenance is required?
- What jurisdiction scope applies?
- Is the destination durable or regenerable?
- Can the record be safely merged?

## Parser relationship

Parser code may consult context to improve interpretation but must not make
runtime evidence durable simply because context was used.

## Invariants

1. parser evidence is not knowledge;
2. runtime state is not canonical context;
3. indexes do not outrank source data;
4. learned context requires explicit promotion;
5. corrections preserve provenance;
6. local observations must not become unsafe global rules.

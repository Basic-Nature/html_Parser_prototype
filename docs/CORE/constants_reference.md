---
layout: default
title: Constants Reference
---

# Constants Reference

This document describes the shared election vocabulary, normalization maps, and
canonical ordering contracts used across Election Pulse.

It is a human-readable reference to implemented behavior.

The primary implementation authority currently lives in:

```text
webapp/parser/Context_Integration/Context_Library/constants.py
```

Supporting vocabulary files may provide data-backed inputs to that module.

This document does not attempt to reproduce every constant value.

## Purpose

Shared constants provide reusable election semantics across parser components.

They help prevent individual handlers and utilities from inventing separate
interpretations of the same concepts.

Common responsibilities include:

- election terminology;
- candidate and contest recognition;
- party recognition and normalization;
- vote-method recognition and normalization;
- canonical ordering;
- source-label interpretation;
- parser classification hints;
- validation vocabulary.

## Authority model

The relationship between code, vocabulary data, and documentation is:

```text
vocabulary files
    |
    v
constants.py
    |
    +-- parser consumers
    +-- normalization logic
    +-- validation logic
    +-- context services
    `-- model and extraction utilities

constants_reference.md
    |
    `-- documents the implemented contract
```

Generated inventories or exported snapshots are diagnostics and review aids.

They do not become authoritative merely because they contain a complete dump of
values.

## Election terminology

Shared election terminology may include recognized words or labels associated
with:

- elections;
- contests;
- candidates;
- parties;
- districts;
- precincts;
- totals;
- reporting status;
- ballot methods.

These vocabularies support classification and extraction.

Recognition does not automatically imply canonicalization.

## Party vocabulary

Party handling uses multiple related structures rather than one universal map.

Important implemented concepts include:

```text
PARTY_KEYWORDS
_PARTY_CANON_MAP
PARTY_CODE_MAP
PARTY_CODE_DESCRIPTIONS
PARTY_NORMALIZATION_MAP
PSEUDO_PARTY_LABELS
PSEUDO_PARTY_RAW_KEYS
```

These structures serve different responsibilities.

## Party recognition

`PARTY_KEYWORDS` supports recognition of party-related text.

Recognition may be used by:

- HTML scanning;
- JSON parsing;
- PDF parsing;
- table extraction;
- context classification;
- model-training utilities.

A recognized party token is not necessarily the final canonical party value.

## Party canonicalization

`_PARTY_CANON_MAP` provides canonical mappings for recognized party aliases.

`PARTY_NORMALIZATION_MAP` combines normalization behavior used when converting
raw party labels into stable output forms.

The normalization process may consider:

```text
raw label
    |
    v
cleanup
    |
    v
party code lookup
    |
    v
canonical alias lookup
    |
    v
normalized party label
```

The exact implementation remains defined by `constants.py`.

## Party codes

`PARTY_CODE_MAP` maps recognized party codes to canonical party identities.

`PARTY_CODE_DESCRIPTIONS` provides additional reference information for known
codes.

The Federal Election Commission party-code reference may be used as an external
reference source where applicable.

External reference data does not override source evidence or local review
policy.

## Pseudo-party labels

Some source labels may resemble parties without representing normal political
party identities.

The implementation distinguishes these through structures such as:

```text
PSEUDO_PARTY_LABELS
PSEUDO_PARTY_RAW_KEYS
```

This prevents normalization logic from treating every party-like source token as
equivalent.

## Vote-method vocabulary

Vote-method handling also uses several related structures.

Important concepts include:

```text
BALLOT_TYPES
BALLOT_TYPES_SORT_ORDER
BALLOT_GROUP_CANON_ORDER
BALLOT_NAME_CANON_MAP
```

These support recognition, normalization, grouping, and stable presentation.

## Vote-method recognition

Source systems may describe the same voting method differently.

Examples may include variants of:

```text
Election Day
Early Voting
Absentee Mail
Provisional
```

Sources may also expose additional categories such as:

```text
Military
Overseas
Curbside
Election-specific methods
```

Recognition should preserve meaningful distinctions.

## Vote-method canonicalization

The general contract is:

```text
raw source label
    |
    v
recognized vote-method vocabulary
    |
    v
canonical name
    |
    v
canonical group
    |
    v
stable output order
```

Canonicalization must not silently discard a source method because it does not
appear in the default Smart Elections method family.

## Canonical ordering

Canonical ordering exists to preserve stable output and cross-precinct
comparability.

Ordering structures may define:

- ballot-method order;
- group order;
- normalized display order;
- known semantic precedence.

Ordering changes presentation.

They must not change vote identity or totals.

## Candidate and contest vocabulary

Shared vocabulary may also support recognition of:

- candidate labels;
- contest labels;
- office terminology;
- district terminology;
- election types;
- totals and footer text.

These shared concepts reduce source-specific duplication.

Handlers may extend source-specific behavior without redefining common parser
semantics unnecessarily.

## Vocabulary-backed loading

Some constants are loaded from vocabulary files rather than being embedded
directly in Python.

This allows reviewed data lists to remain separate from executable logic.

A vocabulary-backed constant should still have a defined consumer and semantic
purpose.

The existence of a text or JSON vocabulary file does not automatically make its
contents canonical election data.

## Consumers

The shared constants layer is used throughout the parser.

Current consumers include components in areas such as:

```text
Context_Integration
data_standardization
handlers/formats
services
utils
health and model-training support
```

Examples include:

- JSON handlers;
- PDF handlers;
- HTML scanners;
- dynamic table extraction;
- pivot logic;
- context services;
- party normalization utilities;
- election-data standardization.

The constants layer therefore behaves like shared parser infrastructure rather
than a handler-specific configuration file.

## Change policy

Changes to shared constants can affect many parser paths.

Before changing a shared vocabulary or normalization map, consider:

1. which parser components consume it;
2. whether the change is global or jurisdiction-specific;
3. whether aliases should be learned context instead;
4. whether source evidence supports the change;
5. whether normalization output changes;
6. whether regression tests cover the affected behavior.

A local source observation should not become a global constant without review.

## Constants versus learned context

Shared constants and learned context have different roles.

```text
shared constants
    =
reviewed reusable parser semantics

learned context
    =
promoted observations with scope and provenance
```

A source-specific correction may belong in learned context rather than in a
global constant.

See:

- [Context system](../ARCHITECTURE/context_system.md)
- [Evidence model](../ARCHITECTURE/evidence_model.md)

## Constants versus canonical election data

Parser vocabulary is not election-result data.

Constants help interpret election sources.

Canonical election records are produced through normalization, verification,
and finalization.

See:

- [Canonical election model](../ARCHITECTURE/canonical_election_model.md)
- [Parser pipeline](../ARCHITECTURE/parser_pipeline.md)
- [Verification](../QUALITY/verification.md)

## Testing expectations

Changes to shared constants should be covered by tests where they affect:

- normalization;
- parser classification;
- party mapping;
- vote-method mapping;
- candidate or contest recognition;
- ordering;
- validation.

Tests should verify behavior rather than simply asserting that a vocabulary file
contains a particular number of entries.

## Invariants

1. shared constants represent reusable parser semantics;
2. generated inventories do not outrank implementation authority;
3. recognition and canonicalization remain distinct;
4. source-specific observations do not automatically become global constants;
5. party normalization may use multiple coordinated maps;
6. vote-method normalization preserves meaningful source distinctions;
7. ordering does not alter vote identity;
8. constants do not replace evidence, verification, or canonical election data.

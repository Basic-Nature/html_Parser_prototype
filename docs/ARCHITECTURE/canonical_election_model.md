---
layout: default
title: Canonical Election Model
---

# Canonical Election Model

The canonical election model defines the normalized election-result contract
shared across formats, jurisdictions, validation, storage, and analysis.

## Primary row model

> Each row represents one precinct for one contest result set.

## Core precinct fields

```text
Precinct
% Precincts Reporting
Election Day Total
Early Voting Total
Absentee Mail Total
Provisional Total
...
Grand Total
```

Additional vote methods may be added when sources provide them.

## Candidate column group

Each candidate receives a stable group:

```text
{Candidate} - Election Day
{Candidate} - Early Voting
{Candidate} - Absentee Mail
{Candidate} - Provisional
{Candidate} - Total Votes
```

Zero-vote candidates and methods remain represented.

## Additional methods

Sources may contain methods such as Military, Curbside, Overseas, or
jurisdiction-specific categories.

Meaningfully distinct methods should be normalized and preserved rather than
discarded.

## Missing versus zero

Missing and zero are different.

`0` means zero votes were reported or validated.

A missing value means the value could not be determined.

Missing evidence must not be rewritten as zero merely to fill the schema.

## Totals

Candidate totals should reconcile with vote-method sums when the source defines
those methods as a complete decomposition.

Precinct and vote-method totals should reconcile where comparable source totals
exist.

If source semantics differ, retain the discrepancy rather than forcing
agreement.

## Grand total row

Numeric columns should support a derived aggregate row calculated from canonical
precinct rows.

## Validation metadata

Canonical output may carry metadata such as:

- discrepancy flag;
- missing method;
- duplicate precinct;
- incomplete reporting;
- break-sensitive reconstruction;
- unresolved normalization;
- review status.

Validation metadata is not a vote value.

## Precinct identity

Normalization should support stable precinct identity while preserving the
source label.

Duplicate precinct rows must be detected.

## Contest identity

Useful identity dimensions include:

- election date;
- jurisdiction;
- contest title;
- office or measure type;
- district;
- party/primary context when applicable.

## Finalization contract

Current Smart Elections integration finalizes through:

```python
finalize_election_output(headers, rows, metadata)
```

Handlers should not invent incompatible final schemas.

## Example shape

```python
{
    "Precinct": "District 5",
    "% Precincts Reporting": "100.00%",
    "Jane Doe (DEM) - Election Day": "200",
    "Jane Doe (DEM) - Early Voting": "120",
    "Jane Doe (DEM) - Absentee Mail": "80",
    "Jane Doe (DEM) - Provisional": "0",
    "Jane Doe (DEM) - Total Votes": "400"
}
```

The example demonstrates shape only. Production previews and validation must use
real extracted data.

## Invariants

1. precinct rows remain comparable;
2. zero-vote candidates and methods are preserved;
3. missing is not rewritten as zero;
4. totals are validated rather than assumed;
5. duplicate precincts are rejected or flagged;
6. canonical values retain provenance;
7. final output uses shared finalization.

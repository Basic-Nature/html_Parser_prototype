---
layout: default
title: FEC Candidate Matching Audit
---

# FEC Candidate Matching Audit

Election Pulse currently includes optional FEC name-matching support in the
fixture-index build workflow.

This capability is an audit aid.

It is not a canonical candidate-identity authority and must not independently
promote or rewrite election records.

## Current implementation

The current implementation authority is:

```text
scripts/build_election_index.py
```

The index builder exposes:

```text
--audit-against-fec
--fec-index
```

The default FEC index path is:

```text
webapp/parser/fixtures/candidate_summary_index.json
```

The matching path operates on the local FEC candidate-summary index rather than
calling the FEC API during the comparison.

## Matching function

The current helper is:

```text
fuzzy_match_candidate(name, candidates_index)
```

It normalizes the supplied name to uppercase and compares it against the
`CLYMER` value in candidate-summary records.

When RapidFuzz is available, the implementation uses:

```text
rapidfuzz.fuzz.token_sort_ratio
```

When RapidFuzz is unavailable, it falls back to:

```text
difflib.SequenceMatcher
```

The current acceptance threshold is:

```text
score >= 70
```

The helper returns the best FEC candidate identifier and score when the
threshold is met.

## Audit behavior

FEC matching is optional and is enabled only when the fixture-index builder is
run with:

```text
--audit-against-fec
```

When enabled, the builder loads the configured local candidate-summary index and
may attach an FEC identifier to generated index records.

Lower-scoring accepted matches may also be written to the fixture audit report
for review.

## Current limitation

The present `build_election_index.py` flow groups records by party and county,
and the optional FEC-matching call currently receives the value held in the
`party` variable.

That means this path should be treated as experimental/audit behavior rather
than as validated candidate identity resolution.

A future candidate-level implementation should pass an actual candidate name
and preserve enough source evidence to review the match.

## Verification boundary

A fuzzy match means only that two strings are similar under the implemented
comparison.

It does not prove:

- candidate identity;
- contest identity;
- party identity;
- election eligibility;
- source correctness.

Any durable identity decision should retain source provenance and use explicit
verification or review policy.

## Example invocation

```text
python scripts/build_election_index.py --audit-against-fec
```

A custom candidate-summary index may be supplied with `--fec-index`.

## Change policy

Before changing FEC matching behavior, verify:

1. which field is being matched;
2. which FEC index supplies the comparison record;
3. which similarity algorithm is active;
4. which threshold is used;
5. how ambiguous matches are recorded;
6. whether the result is audit-only or eligible for reviewed promotion.

## Invariants

1. FEC fuzzy matching is an audit aid, not election truth.
2. current behavior is defined by `scripts/build_election_index.py`.
3. matching uses a local candidate-summary index.
4. the current acceptance threshold is 70.
5. string similarity does not independently establish identity.
6. ambiguous or lower-confidence matches remain reviewable.
7. canonical election data is not silently rewritten by a fuzzy match.

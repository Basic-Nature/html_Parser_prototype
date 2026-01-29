# FEC Fuzzy-Match Policy

## Purpose

This document defines the policy and operational workflow for fuzzy name-matching used to enrich FEC-style candidate data when `candidate_id` is missing or ambiguous.

## Scope

- Applies to CSV/XLSX uploads and feeder scripts that ingest FEC-style candidate rows.
- Covers algorithmic thresholds, auto-enrichment rules, reviewer/manual-mapping workflow, and the persisted mappings format used by the parser.

## Definitions

- Exact match: input row contains a `candidate_id` that maps directly to a known candidate.
- Fuzzy match: a candidate lookup by normalized name (and optional state/party) using string-similarity scoring.
- Enrichment: automatically populating fields such as `candidate_id`, `candidate_name_normalized`, `party`, and `state` when confidence is high.

## Thresholds & Actions

- Auto-enrich (safe): score >= 90
  - Parser will automatically fill `candidate_id` and other canonical fields.
  - A metadata block `_fec_candidate_match` is attached recording `score`, `method`, and `candidates` list.

- Manual review (suggest): 70 <= score < 90
  - Candidate(s) will be surfaced to the reviewer triage UI / JSONL report for human confirmation.
  - The parser may attach `_fec_candidate_match` but will not auto-write `candidate_id` into authoritative outputs.

- No-match / Reject: score < 70
  - No enrichment applied. Row marked as `no_match` in reports.

## Notes

- Thresholds are configurable; default recommended values are: `MIN_FUZZY_SCORE_AUTO=90`, `MIN_FUZZY_SCORE_MANUAL=70`.
- Prefer use of `rapidfuzz` token-based scoring (token_sort_ratio / token_set_ratio) when available; fallback to Python `difflib` if not.

## Scorer Selection

- `auto`: prefer `rapidfuzz` token-sort/token-set scorer; fallback to `difflib.SequenceMatcher` if unavailable.
- `token_sort` / `token_set`: explicit token-based scorers (recommended for noisy name order and middle-initials).
- `simple`: difflib fallback.

## Reviewer Workflow

1. Run the triage report on staged inputs or the `uploads/` folder using the helper script:

```bash
python scripts/generate_fec_fuzzy_report.py --out-jsonl webapp/parser/fixtures/fuzzy_match_report.jsonl \
  --html-report webapp/parser/fixtures/fuzzy_report.html --include-context --scorer auto --top-k 3 --min-score 70
```

1. Open the generated HTML report or call the web reviewer endpoint `/fec_mappings_review` (if running the webapp) to inspect candidate suggestions.
2. Accept / Reject mappings in the reviewer UI. Acceptances are persisted to `webapp/parser/fixtures/mappings.json`.

## Mapping persistence contract

- The persisted mappings file is a simple JSON array of mapping records appended in chronological order.
- Each mapping record SHOULD contain the following keys:

```json
{
  "file": "source.csv",
  "row": 123,
  "mapped_id": "H0ZZZZZZ",
  "note": "Reviewed 2026-01-28, accepted",
  "ts": "2026-01-28T12:34:56Z"
}
```

- Parsers SHOULD consume `mappings.json` at startup or during enrichment to apply reviewer-approved mappings before fuzzy fallback.

## Logging & Metadata

- When a fuzzy lookup is performed, parsers should attach `_fec_candidate_match` to the metadata returned for that row, with the form:

```json
{
  "query": "joan doe",
  "score": 87,
  "method": "token_sort",
  "candidates": [
    {"candidate_id": "H0A1", "name": "Joan A. Doe", "score": 87, "party": "DEM", "state": "CA"},
    {"candidate_id": "H0B2", "name": "Joanne Doe", "score": 71, "party": "DEM", "state": "CA"}
  ]
}
```

## Security & Data Quality Notes

- Mapping persistence is sensitive: treat `webapp/parser/fixtures/mappings.json` as a trusted data source. Restrict write access in production or gate it behind authenticated reviewer UI.
- Reviewer actions should include a `note` and timestamp to aid audits.

## Config Keys

- `MIN_FUZZY_SCORE_AUTO` (int): default 90
- `MIN_FUZZY_SCORE_MANUAL` (int): default 70
- `FUZZY_SCORER` (str): default `auto` (choices: `auto`, `token_sort`, `token_set`, `simple`)

## Operational Checklist

- Add config keys to `webapp/parser/config.py` (or central config) and document them in `README`.
- Add unit tests for `webapp/parser/fec_lookup.py` covering scorer selection, top-k, and threshold behavior.
- Re-run `scripts/generate_fec_fuzzy_report.py` against real uploads and iterate reviewer mappings.

## Appendix: Example review / mapping flow

1. Operator runs report with `--min-score 70`.
2. Operator launches webapp, opens `/fec_mappings_review`, accepts mapping for row 42 mapping to `H0A1`.
3. Reviewer UI POSTs mapping to `/api/fec/save_mapping` and `mappings.json` is appended.
4. Future parser runs load `mappings.json` and apply the persistent mapping before performing fuzzy lookups.

## Contact

For questions about thresholds or reviewer policy, contact the repository maintainers or open an issue in the project.

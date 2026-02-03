# Verified Election Data Cache

This directory contains cached verified election data synced from the Smart Elections reference database (read-only).

## Data Source

- **Google Drive Folder**: `1uwO5BKmgf8gK4Bpu1cHaL4Fw3Bn3ETle` (read-only)
- **Sync Frequency**: Daily (configured in health router)
- **Last Synced**: See `verified_domains.json` → `last_synced` field

## Files

### `verified_domains.json`

List of verified government election websites and domain patterns.

**Structure**:

```json
{
  "domains": ["elections.maryland.gov", "sos.ca.gov", ...],
  "patterns": [".*\\.elections\\..*\\.gov$", ...],
  "last_synced": "2026-02-02T12:00:00Z",
  "sync_source": "google_drive | manual_bootstrap",
  "notes": "Human-readable sync notes"
}
```

### Future Files (Step 3+)

- `verified_schemas/`: Directory containing JSON schemas for each state/county/contest
- `verified_results/`: Sample verified result datasets for schema validation
- `trust_model_weights.json`: ML model weights for trust scoring (if trained)

## Usage

### Trust Scoring (Step 1)

```python
from webapp.parser.utils.url_trust_scorer import compute_trust_score

trust_score, factors = compute_trust_score(url, context, session_id)
# Returns 0-100 score and breakdown of trust factors
```

### Domain Verification

```python
from webapp.parser.utils.url_trust_scorer import get_domain_trust_factors

factors = get_domain_trust_factors(url, context)
# Returns dict with verified_domain, gov_domain, historical_success, etc.
```

### Mimicry Detection

```python
from webapp.parser.utils.url_trust_scorer import detect_domain_mimicry

is_mimic, target_domain = detect_domain_mimicry(url)
# Returns (True, "elections.maryland.gov") if URL mimics verified domain
```

## Sync Process (Step 3 - Not Yet Implemented)

The sync process will:

1. Download latest verified data from Google Drive folder
2. Validate JSON structure and schemas
3. Atomically replace local cache files
4. Log sync events to `log/verified_data_sync.jsonl`
5. Emit telemetry for monitoring

### Manual Sync Command (Future)

```bash
python -m webapp.parser.utils.verified_data_sync --force
```

### Automated Sync (Future)

Configured in `webapp/parser/health/health_router.py` BotPipeline:

- Daily sync at 2:00 AM UTC
- Retry on failure (3 attempts)
- Alert on persistent failures

## Security Notes

- This directory is **read-only** from the parser's perspective
- Never write user-submitted data to this directory
- Sync process validates data integrity before cache replacement
- Google Drive folder is managed separately (manual QA process)

## Testing

### Bootstrap Test Data

```bash
# Verify initial bootstrap file is valid
python -c "import json; print(json.load(open('verified_domains.json')))"
```

### Trust Scorer Integration Test

```bash
# Run parser with trust scoring enabled
python -m webapp.parser.html_election_parser
# Check log/trust_history.jsonl for scoring decisions
```

## Troubleshooting

### "Failed to load verified domains" Warning

- Check file exists: `ls verified_domains.json`
- Validate JSON: `python -c "import json; json.load(open('verified_domains.json'))"`
- Check permissions: File must be readable

### Sync Failures (Step 3)

- Check Google Drive API credentials
- Verify folder ID is correct
- Check network connectivity
- Review `log/verified_data_sync.jsonl` for error details

## Roadmap

- [x] Step 1: URL trust scorer implementation
- [ ] Step 2: DOM snapshot mode for medium-trust URLs
- [ ] Step 3: Google Drive sync automation
- [ ] Step 4: Schema validation against verified data
- [ ] Step 5: Phishing detection and mimicry alerts
- [ ] Step 6: Automated quarantine review pipeline

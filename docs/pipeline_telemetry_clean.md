# Pipeline Telemetry

This document describes the telemetry events, counters, files, and a basic runbook
used by the Smart Elections Parser pipeline.

Status: stable (2026-01-30)

## Goals

- Emit small, privacy-safe JSONL telemetry events for important pipeline lifecycle points.
- Keep lightweight aggregation counters for quick health checks.
- Provide simple commands for inspection and debugging.

## Files & Locations

- Telemetry events JSONL: `webapp/parser/log/telemetry.jsonl` (see `LOG_DIR` config).
- Aggregation counters: `webapp/parser/log/telemetry_counters.json` (atomic JSON file).
- Processed URLs cache: `PROCESSED_URLS_FILE` (`.processed_urls`).

## Environment toggles

- `ENABLE_TELEMETRY_AGG` (default: true)
- `ENABLE_VERBOSE_TELEMETRY` (default: false)
- `TELEMETRY_INCLUDE_URL` (default: false)

## Common event fields

- `event`, `ts_ms`, `ts_iso`, `session_id`, `run_id`, `url_domain`, `url_hash`, `env`

## Key events

- `navigation_start`, `navigation_complete`, `page_scrolled`, `handler_selected`, `parse_result`, `url_processed`, `processing_error`

## Aggregation counters

Example `telemetry_counters.json`:

```json
{
  "processed_total": 123,
  "processed_success": 100,
  "processed_fail": 17
}
```

## Quick checks

```bash
tail -n 50 webapp/parser/log/telemetry.jsonl | jq -c .
python -c "import json;print(json.dumps(json.load(open('webapp/parser/log/telemetry_counters.json')),indent=2))"
```

## Smoke test

Run: `python tools/collect_telemetry_sample.py`

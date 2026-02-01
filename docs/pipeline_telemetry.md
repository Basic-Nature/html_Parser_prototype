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
- Processed URLs cache: `PROCESSED_URLS_FILE` (`.processed_urls`) — unchanged layout, enriched with new keys.

Note: `LOG_DIR` is configured in `webapp/parser/config.py`. For local dev this usually maps to `webapp/parser/log`.

## Environment toggles

- `ENABLE_TELEMETRY_AGG` (default: true): enable/disable aggregation counter updates.
- `ENABLE_VERBOSE_TELEMETRY` (default: false): include larger metadata blobs in telemetry events when true.
- `TELEMETRY_INCLUDE_URL` (default: false): when true, `emit_telemetry_event` may include `url_full` (not recommended in production).

## Common event fields

All events include the following canonical fields (where applicable):

- `event` : string — event name (e.g. `navigation_start`, `parse_result`).
- `ts_ms` : integer — epoch ms timestamp.
- `ts_iso` : string — ISO8601 UTC timestamp.
- `session_id` : string | null — logical session id if present.
- `run_id` / `request_id` : string | null — optional unique id to correlate related events.
- `url_domain` : string | null — parsed hostname (no query string).
- `url_hash` : string | null — short SHA1-derived hash of the full URL.
- `env` : string — environment (dev/prod) if available.

Sensitive data: by default full URLs and PII are redacted. Use `TELEMETRY_INCLUDE_URL=true` only for local debugging.

## Key events (schema snippets)

- `navigation_start`
  - fields: `{ event, ts_*, session_id, url_hash, url_domain, run_id }`

- `navigation_complete`
  - fields: `{ event, ts_*, session_id, run_id, nav_executed:bool, nav_script_id?:str, nav_telemetry?:object }`

- `page_scrolled`
  - fields: `{ event, ts_*, session_id, run_id, scroll_metrics: { attempts:int, total_ms:int, avg_delta_px:int, max_delta_px:int } }`

- `handler_selected`
  - fields: `{ event, ts_*, session_id, run_id, handler_name, summary?:object }`

- `parse_result`
  - fields: `{ event, ts_*, session_id, run_id, handler_name, success:bool, fallback:bool, tables_seen:int, dom_table_rows:int, row_count:int, column_count:int, metadata_keys:list }`

- `url_processed` (emitted from `mark_url_processed`)
  - fields: full `entry` object written to `.processed_urls` plus `ts_*` and derived `url_domain`/`url_hash`.

- `processing_error`
  - fields: `{ event, ts_*, session_id, run_id, phase, error_type, message, short_stack }`

## Aggregation counters

`telemetry_counters.json` is a small JSON map of counters. Example:

```json
{
  "processed_total": 123,
  "processed_success": 100,
  "processed_fail": 17,
  "processed_partial": 4,
  "processed_cancelled": 2,
  "fallbacks": 9,
  "tables_seen_total": 42
}
```

Counters are updated atomically by the helper `increment_counter(name, amount=1)`.

## Quick inspection commands

- Tail last telemetry events:

```bash
tail -n 50 webapp/parser/log/telemetry.jsonl | jq -c .
```

- Show counters:

```bash
python -c "import json;print(json.dumps(json.load(open('webapp/parser/log/telemetry_counters.json')),indent=2))"
```

- Check last processed URL entry:

```bash
python -c "import orjson,sys;print(orjson.loads(open('.processed_urls','rb').read())[-1])"
```

## Runbook — triage steps

1. If `fallback_rate` (fallbacks / processed_total) > 5% over a short interval, open an incident and attach recent `telemetry.jsonl` entries and a sample failing URL.
2. If `processed_fail` rises: inspect `processing_error` events (search `telemetry.jsonl`), capture sample HTML + logs, and add to `tools/regression_samples/`.
3. For UI errors where the front-end receives non-JSON: check `/api/` endpoints; the global Flask error handler now returns JSON for API routes — search `telemetry.jsonl` for `processing_error` events matching the session.

## How to run the smoke test

1. Emit a sample event (already included at `tools/collect_telemetry_sample.py`):

```bash
python tools/collect_telemetry_sample.py
```

1. Confirm a matching line appears in `webapp/parser/log/telemetry.jsonl` and `telemetry_counters.json` updated where appropriate.

## Next improvements (suggestions)

- Add an HTTP endpoint to expose `telemetry_counters.json` for quick health checks (read-only, with auth).
- Add sampling rules and rate limit for verbose metadata to avoid log bloat.
- Integrate a Prometheus exporter or push gateway for real-time counters.

## Contact

For questions or to adjust event schemas, see `webapp/parser/html_election_parser.py` and `webapp/parser/utils/telemetry.py` — these are the canonical implementations.

## Pipeline Telemetry

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
- Processed URLs cache: `PROCESSED_URLS_FILE` (`.processed_urls`) — unchanged layout, enriched with new keys.

Note: `LOG_DIR` is configured in `webapp/parser/config.py`. For local dev this usually maps to `webapp/parser/log`.

## Environment toggles

- `ENABLE_TELEMETRY_AGG` (default: true): enable/disable aggregation counter updates.
- `ENABLE_VERBOSE_TELEMETRY` (default: false): include larger metadata blobs in telemetry events when true.
- `TELEMETRY_INCLUDE_URL` (default: false): when true, `emit_telemetry_event` may include `url_full` (not recommended in production).

## Common event fields

All events include the following canonical fields (where applicable):

- `event` : string — event name (e.g. `navigation_start`, `parse_result`).
- `ts_ms` : integer — epoch ms timestamp.
- `ts_iso` : string — ISO8601 UTC timestamp.
- `session_id` : string | null — logical session id if present.
- `run_id` / `request_id` : string | null — optional unique id to correlate related events.
- `url_domain` : string | null — parsed hostname (no query string).
- `url_hash` : string | null — short SHA1-derived hash of the full URL.
- `env` : string — environment (dev/prod) if available.

Sensitive data: by default full URLs and PII are redacted. Use `TELEMETRY_INCLUDE_URL=true` only for local debugging.

## Key events (schema snippets)

- `navigation_start`
  - fields: `{ event, ts_*, session_id, url_hash, url_domain, run_id }`

- `navigation_complete`
  - fields: `{ event, ts_*, session_id, run_id, nav_executed:bool, nav_script_id?:str, nav_telemetry?:object }`

- `page_scrolled`
  - fields: `{ event, ts_*, session_id, run_id, scroll_metrics: { attempts:int, total_ms:int, avg_delta_px:int, max_delta_px:int } }`

- `handler_selected`
  - fields: `{ event, ts_*, session_id, run_id, handler_name, summary?:object }`

- `parse_result`
  - fields: `{ event, ts_*, session_id, run_id, handler_name, success:bool, fallback:bool, tables_seen:int, dom_table_rows:int, row_count:int, column_count:int, metadata_keys:list }`

- `url_processed` (emitted from `mark_url_processed`)
  - fields: full `entry` object written to `.processed_urls` plus `ts_*` and derived `url_domain`/`url_hash`.

- `processing_error`
  - fields: `{ event, ts_*, session_id, run_id, phase, error_type, message, short_stack }`

## Aggregation counters

`telemetry_counters.json` is a small JSON map of counters. Example:

```json
{
  "processed_total": 123,
  "processed_success": 100,
  "processed_fail": 17,
  "processed_partial": 4,
  "processed_cancelled": 2,
  "fallbacks": 9,
  "tables_seen_total": 42
}
```

Counters are updated atomically by the helper `increment_counter(name, amount=1)`.

## Quick inspection commands

- Tail last telemetry events:

```bash
tail -n 50 webapp/parser/log/telemetry.jsonl | jq -c .
```

- Show counters:

```bash
python -c "import json;print(json.dumps(json.load(open('webapp/parser/log/telemetry_counters.json')),indent=2))"
```

- Check last processed URL entry:

```bash
python -c "import orjson,sys;print(orjson.loads(open('.processed_urls','rb').read())[-1])"
```

## Runbook — triage steps

1. If `fallback_rate` (fallbacks / processed_total) > 5% over a short interval, open an incident and attach recent `telemetry.jsonl` entries and a sample failing URL.
2. If `processed_fail` rises: inspect `processing_error` events (search `telemetry.jsonl`), capture sample HTML + logs, and add to `tools/regression_samples/`.
3. For UI errors where the front-end receives non-JSON: check `/api/` endpoints; the global Flask error handler now returns JSON for API routes — search `telemetry.jsonl` for `processing_error` events matching the session.

## How to run the smoke test

1. Emit a sample event (already included at `tools/collect_telemetry_sample.py`):

```bash
python tools/collect_telemetry_sample.py
```

1. Confirm a matching line appears in `webapp/parser/log/telemetry.jsonl` and `telemetry_counters.json` updated where appropriate.

## Next improvements (suggestions)

- Add an HTTP endpoint to expose `telemetry_counters.json` for quick health checks (read-only, with auth).
- Add sampling rules and rate limit for verbose metadata to avoid log bloat.
- Integrate a Prometheus exporter or push gateway for real-time counters.

## Contact

For questions or to adjust event schemas, see `webapp/parser/html_election_parser.py` and `webapp/parser/utils/telemetry.py` — these are the canonical implementations.

## Pipeline Telemetry

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
- Processed URLs cache: `PROCESSED_URLS_FILE` (`.processed_urls`) — unchanged layout, enriched with new keys.

Note: `LOG_DIR` is configured in `webapp/parser/config.py`. For local dev this usually maps to `webapp/parser/log`.

## Environment toggles

- `ENABLE_TELEMETRY_AGG` (default: true): enable/disable aggregation counter updates.
- `ENABLE_VERBOSE_TELEMETRY` (default: false): include larger metadata blobs in telemetry events when true.
- `TELEMETRY_INCLUDE_URL` (default: false): when true, `emit_telemetry_event` may include `url_full` (not recommended in production).

## Common event fields

All events include the following canonical fields (where applicable):

- `event` : string — event name (e.g. `navigation_start`, `parse_result`).
- `ts_ms` : integer — epoch ms timestamp.
- `ts_iso` : string — ISO8601 UTC timestamp.
- `session_id` : string | null — logical session id if present.
- `run_id` / `request_id` : string | null — optional unique id to correlate related events.
- `url_domain` : string | null — parsed hostname (no query string).
- `url_hash` : string | null — short SHA1-derived hash of the full URL.
- `env` : string — environment (dev/prod) if available.

Sensitive data: by default full URLs and PII are redacted. Use `TELEMETRY_INCLUDE_URL=true` only for local debugging.

## Key events (schema snippets)

- `navigation_start`
  - fields: `{ event, ts_*, session_id, url_hash, url_domain, run_id }`

- `navigation_complete`
  - fields: `{ event, ts_*, session_id, run_id, nav_executed:bool, nav_script_id?:str, nav_telemetry?:object }`

- `page_scrolled`
  - fields: `{ event, ts_*, session_id, run_id, scroll_metrics: { attempts:int, total_ms:int, avg_delta_px:int, max_delta_px:int } }`

- `handler_selected`
  - fields: `{ event, ts_*, session_id, run_id, handler_name, summary?:object }`

- `parse_result`
  - fields: `{ event, ts_*, session_id, run_id, handler_name, success:bool, fallback:bool, tables_seen:int, dom_table_rows:int, row_count:int, column_count:int, metadata_keys:list }`

- `url_processed` (emitted from `mark_url_processed`)
  - fields: full `entry` object written to `.processed_urls` plus `ts_*` and derived `url_domain`/`url_hash`.

- `processing_error`
  - fields: `{ event, ts_*, session_id, run_id, phase, error_type, message, short_stack }`

## Aggregation counters

`telemetry_counters.json` is a small JSON map of counters. Example:

```json
{
  "processed_total": 123,
  "processed_success": 100,
  "processed_fail": 17,
  "processed_partial": 4,
  "processed_cancelled": 2,
  "fallbacks": 9,
  "tables_seen_total": 42
}
```

Counters are updated atomically by the helper `increment_counter(name, amount=1)`.

## Quick inspection commands

- Tail last telemetry events:

```bash
tail -n 50 webapp/parser/log/telemetry.jsonl | jq -c .
```

- Show counters:

```bash
python -c "import json;print(json.dumps(json.load(open('webapp/parser/log/telemetry_counters.json')),indent=2))"
```

- Check last processed URL entry:

```bash
python -c "import orjson,sys;print(orjson.loads(open('.processed_urls','rb').read())[-1])"
```

## Runbook — triage steps

1. If `fallback_rate` (fallbacks / processed_total) > 5% over a short interval, open an incident and attach recent `telemetry.jsonl` entries and a sample failing URL.
1. If `processed_fail` rises: inspect `processing_error` events (search `telemetry.jsonl`), capture sample HTML + logs, and add to `tools/regression_samples/`.
1. For UI errors where the front-end receives non-JSON: check `/api/` endpoints; the global Flask error handler now returns JSON for API routes — search `telemetry.jsonl` for `processing_error` events matching the session.

## How to run the smoke test

1. Emit a sample event (already included at `tools/collect_telemetry_sample.py`):

```bash
python tools/collect_telemetry_sample.py
```

1. Confirm a matching line appears in `webapp/parser/log/telemetry.jsonl` and `telemetry_counters.json` updated where appropriate.

## Next improvements (suggestions)

- Add an HTTP endpoint to expose `telemetry_counters.json` for quick health checks (read-only, with auth).
- Add sampling rules and rate limit for verbose metadata to avoid log bloat.
- Integrate a Prometheus exporter or push gateway for real-time counters.

## Contact

For questions or to adjust event schemas, see `webapp/parser/html_election_parser.py` and `webapp/parser/utils/telemetry.py` — these are the canonical implementations.

## Pipeline Telemetry

## Goals

## Files & Locations

## Environment toggles

## Common event fields

## Key events (schema snippets)

## Aggregation counters

## Quick inspection commands

## Runbook — triage steps

## How to run the smoke test

## Next improvements (suggestions)

## Contact

`telemetry_counters.json` is a small JSON map of counters. Example:

```json
```

1. If `fallback_rate` (fallbacks / processed_total) > 5% over a short interval, open an incident and attach recent `telemetry.jsonl` entries and a sample failing URL.
1. If `processed_fail` rises: inspect `processing_error` events (search `telemetry.jsonl`), capture sample HTML + logs, and add to `tools/regression_samples/`.
1. For UI errors where the front-end receives non-JSON: check `/api/` endpoints; the global Flask error handler now returns JSON for API routes — search `telemetry.jsonl` for `processing_error` events matching the session.
1. Emit a sample event (already included at `tools/collect_telemetry_sample.py`):
1. Confirm a matching line appears in `webapp/parser/log/telemetry.jsonl` and `telemetry_counters.json` updated where appropriate.
For questions or to adjust event schemas, see `webapp/parser/html_election_parser.py` and `webapp/parser/utils/telemetry.py` — these are the canonical implementations.

## Pipeline Telemetry

This document describes the telemetry events, counters, files, and basic runbook used by the Smart Elections Parser pipeline.

Status: stable (2026-01-30)

## Goals

- Emit small, privacy-safe JSONL telemetry events for important pipeline lifecycle points.
- Keep lightweight aggregation counters for quick health checks.
- Provide simple commands for inspection and debugging.

## Files & Locations

- Telemetry events JSONL: `webapp/parser/log/telemetry.jsonl` (see `LOG_DIR` config)
- Aggregation counters: `webapp/parser/log/telemetry_counters.json` (atomic JSON file)
- Processed URLs cache: `PROCESSED_URLS_FILE` (`.processed_urls`) — unchanged layout, enriched with new keys.

Note: `LOG_DIR` is configured in `webapp/parser/config.py`. For local dev this usually maps to `webapp/parser/log`.

## Environment toggles

- `ENABLE_TELEMETRY_AGG` (default: true): enable/disable aggregation counter updates.
- `ENABLE_VERBOSE_TELEMETRY` (default: false): when true, include larger metadata blobs in telemetry events (useful for debugging).
- `TELEMETRY_INCLUDE_URL` (default: false): when true, `emit_telemetry_event` may include `url_full` (not recommended in production).

## Common event fields

All events include the following canonical fields (where applicable):

- `event` : string — event name (e.g. `navigation_start`, `parse_result`).
- `ts_ms` : integer — epoch ms timestamp.
- `ts_iso` : string — ISO8601 UTC timestamp.
- `session_id` : string | null — logical session id if present.
- `run_id` / `request_id` : string | null — optional unique id to correlate related events.
- `url_domain` : string | null — parsed hostname (no query string).
- `url_hash` : string | null — short SHA1-derived hash of the full URL.
- `env` : string — environment (dev/prod) if available.

Sensitive data: by default full URLs and PII are redacted. Use `TELEMETRY_INCLUDE_URL=true` only for local debugging.

## Key events (schema snippets)

- `navigation_start`
  - fields: `{ event, ts_*, session_id, url_hash, url_domain, run_id }`

- `navigation_complete`
  - fields: `{ event, ts_*, session_id, run_id, nav_executed:bool, nav_script_id?:str, nav_telemetry?:object }`

- `page_scrolled`
  - fields: `{ event, ts_*, session_id, run_id, scroll_metrics: { attempts:int, total_ms:int, avg_delta_px:int, max_delta_px:int } }`

- `handler_selected`
  - fields: `{ event, ts_*, session_id, run_id, handler_name, summary?:object }`

- `parse_result`
  - fields: `{ event, ts_*, session_id, run_id, handler_name, success:bool, fallback:bool, tables_seen:int, dom_table_rows:int, row_count:int, column_count:int, metadata_keys:list }`

- `url_processed` (emitted from `mark_url_processed`)
  - fields: full `entry` object written to `.processed_urls` plus `ts_*` and derived `url_domain`/`url_hash`.

- `processing_error`
  - fields: `{ event, ts_*, session_id, run_id, phase, error_type, message, short_stack }`

## Aggregation counters

`telemetry_counters.json` is a small JSON map of counters. Example:

```bash
{
  "processed_total": 123,
  "processed_success": 100,
  "processed_fail": 17,
  "processed_partial": 4,
  "processed_cancelled": 2,
  "fallbacks": 9,
  "tables_seen_total": 42
}
```

Counters are updated atomically by the helper `increment_counter(name, amount=1)`.

## Quick inspection commands

- Tail last telemetry events:

```bash
tail -n 50 webapp/parser/log/telemetry.jsonl | jq -c .
```

- Show counters:

```bash
python -c "import json;print(json.dumps(json.load(open('webapp/parser/log/telemetry_counters.json')),indent=2))"
```

- Check last processed URL entry:

```bash
python -c "import orjson,sys;print(orjson.loads(open('.processed_urls','rb').read())[-1])"
```

## Runbook — triage steps

1. If `fallback_rate` (fallbacks / processed_total) > 5% over a short interval, open an incident and attach recent `telemetry.jsonl` entries and a sample failing URL.
2. If `processed_fail` rises: inspect `processing_error` events (search `telemetry.jsonl`), capture sample HTML + logs, and add to `tools/regression_samples/`.
3. For UI errors where the front-end receives non-JSON: check `/api/` endpoints; the global Flask error handler now returns JSON for API routes — search `telemetry.jsonl` for `processing_error` events matching the session.

## How to run the smoke test

1. Emit a sample event (already included at `tools/collect_telemetry_sample.py`):

```bash
python tools/collect_telemetry_sample.py
```

1. Confirm a matching line appears in `webapp/parser/log/telemetry.jsonl` and `telemetry_counters.json` updated where appropriate.

## Next improvements (suggestions)

- Add an HTTP endpoint to expose `telemetry_counters.json` for quick health checks (read-only, with auth).
- Add sampling rules and rate limit for verbose metadata to avoid log bloat.
- Integrate a Prometheus exporter or push gateway for real-time counters.

## Contact

For questions or to adjust event schemas, see `webapp/parser/html_election_parser.py` and `webapp/parser/utils/telemetry.py` — these are the canonical implementations.

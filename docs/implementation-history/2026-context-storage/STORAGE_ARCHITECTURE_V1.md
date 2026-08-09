---
layout: default
---

# Storage Architecture – Cache, Log, Database Flow

**Version:** 1.0
**Date:** 2026-02-14
**Status:** Production

## Overview

The Smart Elections Parser employs a **three-tier storage architecture** to manage data flow from ephemeral session context through training/curation to persistent verified data:

1. **Cache Tier** – Session-scoped temporary storage
2. **Log Tier** – Append-only short-term memory (JSONL/NDJSON)
3. **Database Tier** – Long-term verified data (PostgreSQL)

This document describes the contracts, flow patterns, and automation hooks for each tier.

---

## 1. Cache Tier (Temporary)

### Purpose

- Store ephemeral context gathered during active parsing sessions
- Enable fast in-memory lookups without hitting disk/DB repeatedly
- Provide intermediate storage before curation/verification

### Key Files

| File | Location | Purpose | Lifetime |
| ------ | ---------- | --------- | ---------- |
| `context_cache.json` | `Context_Integration/Context_Library/cache/` | Aggregated page/form context, labels, discovered patterns | Single session |
| `embedding_disk_cache.pkl` | `Context_Integration/Context_Library/cache/` | Cached ML embeddings (sentence-transformers) | Multi-session (pickle) |

### Load/Save Contracts

**Loading:**

```python
from webapp.parser.utils.html_scanner import load_context_cache_from_disk

cache = load_context_cache_from_disk()  # Returns dict or {}
```

**Saving:**

```python
from webapp.parser.utils.html_scanner import save_context_cache_to_disk
from webapp.parser.Context_Integration.context_organizer import ContextOrganizer

organizer = ContextOrganizer()
organizer._context_cache.update(new_context)
save_context_cache_to_disk(organizer._context_cache)
```

**Cache Structure:**

```json
{
  "state_county_pairs": [["California", "Alameda County"], ...],
  "discovered_contests": ["State Assembly District 1", ...],
  "candidate_labels": {"John Doe": "democratic", ...},
  "panel_tags": ["div", "section", ...],
  "heading_tags": ["h1", "h2", "h3", ...],
  "custom_attr_patterns": ["^data-", "^aria-", "^role$", ...]
}
```

### Flush Mechanisms

**Manual Flush:**

```bash
export FLUSH_CACHE=true
python automate.py
```

**Programmatic Flush:**

```python
import os
from webapp.parser.Context_Integration.Context_Library import CACHE_DIR

# Delete cache files
for cache_file in ["context_cache.json", "embedding_disk_cache.pkl"]:
    path = os.path.join(CACHE_DIR, cache_file)
    if os.path.exists(path):
        os.remove(path)
```

**Automated Pipeline Flush:**

- `BotPipeline.run()` in `health_router.py` respects `FLUSH_CACHE=true`
- Passed to `manual_correction_bot.py` as `--flush-cache` flag
- Cache is cleared **before** context migration to prevent stale data pollution

---

## 2. Log Tier (Short-Term Memory)

### Purpose

- Capture append-only training signals from successful/failed parsing
- Store interim data before verification/curation
- Enable incremental learning without polluting DB with unverified data

### Key Files

| File | Location | Purpose | Format |
| ------ | ---------- | --------- | -------- |
| `spacy_ner_train_data.jsonl` | `Context_Integration/Context_Library/log/` | NER training examples (PERSON, ORG, GPE, CONTEST) | JSONL |
| `field_selection_log.jsonl` | `Context_Integration/Context_Library/log/` | User field selections for table extraction | JSONL |
| `telemetry.jsonl` | `log/` | Session telemetry (state detection, contest selection) | JSONL |
| `navigation_learning_log.jsonl` | `log/` | Navigation recipe feedback (success/fail) | JSONL |
| `selenium_ner_training.jsonl` | `log/` | NER examples captured via Selenium from CAPTCHA sites | JSONL |
| `captcha_resolution_log.jsonl` | `log/` | Manual CAPTCHA resolution events | JSONL |
| `dom_pattern_kb.jsonl` | `Context_Integration/Context_Library/log/` | Discovered DOM patterns for table extraction | JSONL |

### Write Patterns

**Append-only (thread-safe):**

```python
import orjson

example = {
    "timestamp": "2026-02-14T12:00:00Z",
    "text": "John Doe defeated Jane Smith in the State Assembly race.",
    "entities": [(0, 8, "PERSON"), (18, 28, "PERSON"), (36, 50, "CONTEST")]
}

with open("log/spacy_ner_train_data.jsonl", "a", encoding="utf-8") as f:
    f.write(orjson.dumps(example).decode() + "\n")
```

**Deduplication via `log_cache_cleaner_bot`:**

- Runs during `BotPipeline.run()` → `clean_and_migrate()`
- Loads all JSONL entries, deduplicates by content hash
- Persists deduplicated data back to log files
- Deletes duplicates to prevent training data pollution

### Migration to Database

**Flow:** Log → `context_migration.py` → PostgreSQL

```python
from webapp.parser.Context_Integration.context_migration import migrate_context_to_db

migrate_context_to_db()
# Reads all JSONL logs, validates, inserts into SQL tables
# Updates migration_state.json to track processed files
```

**Tracking State:**

- `migration_state.json` stores file modification timestamps
- Only processes files with newer timestamps than last migration
- Prevents redundant DB inserts

---

## 3. Database Tier (Long-Term Verified Data)

### Purpose

- Store verified election results, context metadata, and training signals
- Enable cross-session learning and historical pattern matching
- Provide source-of-truth for ML model training

### Schema (PostgreSQL)

**Core Tables:**

```sql
CREATE TABLE election_results (
    id SERIAL PRIMARY KEY,
    state VARCHAR(100),
    county VARCHAR(100),
    contest VARCHAR(255),
    candidate VARCHAR(255),
    party VARCHAR(100),
    votes INTEGER,
    percentage NUMERIC(5, 2),
    division VARCHAR(100),  -- precinct/district/county/state-wide
    source_url TEXT,
    ingestion_date TIMESTAMP,
    handler VARCHAR(50),
    metadata JSONB
);

CREATE TABLE context_library_entries (
    id SERIAL PRIMARY KEY,
    context_type VARCHAR(50),  -- state_county, contest, candidate, panel_tag, etc.
    key VARCHAR(255),
    value JSONB,
    confidence NUMERIC(4, 3),
    source VARCHAR(100),  -- manual, auto, ml_inferred
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE ner_training_data (
    id SERIAL PRIMARY KEY,
    text TEXT,
    entities JSONB,  -- [{"start": 0, "end": 8, "label": "PERSON"}, ...]
    source VARCHAR(100),  -- spacy_ner_train_data, selenium_ner_training
    verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP
);
```

### Access Patterns

**Read (Context Lookup):**

```python
from webapp.parser.utils.db_utils import SessionLocal

with SessionLocal() as session:
    result = session.execute(
        "SELECT value FROM context_library_entries WHERE context_type='state_county' AND key='California'"
    ).fetchone()
    if result:
        counties = result[0]  # JSONB array
```

**Write (Insert Results):**

```python
from webapp.parser.utils.db_utils import SessionLocal
from webapp.parser.utils.models import ElectionResult

with SessionLocal() as session:
    result = ElectionResult(
        state="California",
        county="Alameda County",
        contest="State Assembly District 15",
        candidate="John Doe",
        party="Democratic",
        votes=12345,
        percentage=55.6,
        division="county-wide",
        source_url="https://example.gov/results",
        handler="html_scan_handler"
    )
    session.add(result)
    session.commit()
```

### Export for Training

**Generate Training Dataset:**

```python
from webapp.parser.utils.db_utils import SessionLocal

with SessionLocal() as session:
    rows = session.execute(
        "SELECT text, entities FROM ner_training_data WHERE verified=TRUE"
    ).fetchall()

    training_data = [
        {"text": row[0], "entities": row[1]}
        for row in rows
    ]
```

---

## 4. Data Flow Pipeline

### Session Lifecycle

```branch
┌─────────────┐
│  Web Scrape │
│   (parser)  │
└──────┬──────┘
       │
       ▼
┌─────────────┐    Load/Update     ┌────────────────┐
│    Cache    │◄─────────────────►│ ContextOrganizer│
│  (Session)  │                    │  (in-memory)   │
└──────┬──────┘                    └────────────────┘
       │
       │ Save
       ▼
┌─────────────┐    Append-only     ┌────────────────┐
│  Log Files  │◄───────────────────│  Training      │
│  (JSONL)    │                    │  Signals       │
└──────┬──────┘                    └────────────────┘
       │
       │ BotPipeline.run()
       │ → log_cache_cleaner_bot (dedupe)
       │ → context_migration.py
       ▼
┌─────────────┐    Verified Data   ┌────────────────┐
│  PostgreSQL │◄───────────────────│  Manual Review │
│   (DB)      │                    │  + Auto Curate │
└─────────────┘                    └────────────────┘
```

### Automation Entry Points

**1. Web App Session:**

```branch
User uploads file → Parser extracts data → Cache updated → Logs appended
```

**2. CLI Session:**

```bash
python webapp/parser/html_election_parser.py --url https://example.gov/results
# → Cache + Logs written
```

**3. Automated Health Pipeline:**

```bash
python automate.py
# → BotPipeline.run()
# → Preclean logs/cache
# → log_cache_cleaner_bot (dedupe)
# → manual_correction_bot (verify)
# → context_migration.py (DB insert)
# → retrain_models (ML update)
```

### Automation + Embedding Cache Policy (Operational Defaults)

To keep CI/local behavior explicit and reduce noise while preserving useful training context:

1. **Automation report retention** (`automate.py`)
    - CI and explicit runs can enforce report cleanup using:
      - `--enforce-report-retention`
      - `--report-retention-days` (default `30`)
      - `--report-max-files` (default `200`)
      - `--report-max-bytes` (default `268435456`)
    - Targets `output/reports/report_*.json` while preserving `*_latest.json` and `automation_run_latest.json`.

2. **Embedding cache lifecycle gate** (`webapp/parser/utils/embedding_cache.py`)
    - Startup: load disk cache + emit precheck (`EMBEDDING_CACHE_PRECHECK=true` by default).
    - Optional startup seed: hydrate memory/disk from DB (`EMBEDDING_CACHE_SEED_ON_START=true`, `EMBEDDING_CACHE_SEED_LIMIT=250`).
    - Runtime checkpoints: periodic disk snapshots after writes/time (`EMBEDDING_CACHE_CHECKPOINT_WRITES=250`, `EMBEDDING_CACHE_CHECKPOINT_SECONDS=120`).
    - Shutdown: force final disk save via registered exit hook.
    - Size warning: alert if cache file grows beyond `EMBEDDING_CACHE_DISK_WARN_MB` (default `512`).

3. **Automation preflight visibility** (`automate.py`)
    - Every `python automate.py` run now captures an always-on `embedding_cache_preflight` stage.
    - Stage details (cache mode/state/checkpoint policy) are written to `output/reports/automation_run_latest.json` under `stage_details.embedding_cache_preflight`.

These defaults keep parser and NLP/ML pathways reactive across sessions without forcing DB writes in read-only/test modes.

---

## 5. Cache Management

### When to Flush Cache

**Required:**

- After major schema changes to `context_cache.json`
- When embedding model (`sentence-transformers`) is updated
- After bulk import of new training data to force re-extraction

**Optional (Performance):**

- Before running full `automate.py` to ensure fresh context
- After manual corrections to `context_library.json`

### Environment Variables

```bash
# Flush cache during pipeline run
export FLUSH_CACHE=true
python automate.py

# Additional cache control
export CACHE_EXPIRE_DAYS=7  # Auto-expire old cache entries
```

### Manual Cache Cleanup

```bash
# Delete all cache files
rm -rf webapp/parser/Context_Integration/Context_Library/cache/*

# Delete specific cache
rm webapp/parser/Context_Integration/Context_Library/cache/context_cache.json
rm webapp/parser/Context_Integration/Context_Library/cache/embedding_disk_cache.pkl
```

---

## 6. Log Management

### Deduplication Strategy

**Automatic (BotPipeline):**

- `log_cache_cleaner_bot` runs during `BotPipeline.clean_and_migrate()`
- Hashes each JSONL entry content
- Keeps first occurrence, deletes duplicates
- Preserves chronological order

**Manual Deduplication:**

```python
from webapp.parser.health.log_cache_cleaner_bot import deduplicate_jsonl

deduplicate_jsonl("log/spacy_ner_train_data.jsonl")
```

### Log Rotation

**Current:** No automatic rotation (JSONL append-only grows indefinitely)

**Planned (TODO):**

- Rotate logs > 100MB to `.jsonl.{timestamp}.bak`
- Keep last 30 days of logs, archive older to compressed tarball
- Add `LOG_ROTATION_SIZE_MB` and `LOG_RETENTION_DAYS` env vars

### Archival

**Best Practice:**

```bash
# Compress old logs after migration to DB
cd webapp/parser/Context_Integration/Context_Library/log
tar -czf archive_$(date +%Y%m%d).tar.gz spacy_ner_train_data.jsonl field_selection_log.jsonl
rm spacy_ner_train_data.jsonl field_selection_log.jsonl
# Recreate empty logs
touch spacy_ner_train_data.jsonl field_selection_log.jsonl
```

---

## 7. Database Maintenance

### Vacuum and Reindex

**PostgreSQL:**

```sql
-- Run weekly to reclaim space and rebuild indexes
VACUUM ANALYZE election_results;
VACUUM ANALYZE context_library_entries;
VACUUM ANALYZE ner_training_data;

-- Optional: FULL vacuum (locks table)
VACUUM FULL election_results;
```

### Export for Backup

```bash
# Dump entire database
pg_dump -U postgres smart_elections > backup_$(date +%Y%m%d).sql

# Restore
psql -U postgres smart_elections < backup_20260214.sql
```

---

## 8. Integration with NLP/ML Pipeline

### Training Data Sources

| Source | Tier | Verification | Use Case |
| -------- | ------ | -------------- | ---------- |
| `spacy_ner_train_data.jsonl` | Log | Auto-generated | Initial NER training |
| `selenium_ner_training.jsonl` | Log | Auto-generated | Entity-rich CAPTCHA site data |
| `ner_training_data` (DB) | Database | Manual verified | High-quality model training |

### Model Training Workflow

```branch
┌─────────────────────┐
│  Log Files (JSONL)  │
└──────────┬──────────┘
           │
           │ BotPipeline → context_migration
           ▼
┌─────────────────────┐    Manual Review     ┌────────────────┐
│  PostgreSQL (raw)   │───────────────────►│  Verified=TRUE │
└──────────┬──────────┘                      └────────┬───────┘
           │                                          │
           │                                          │
           └──────────────┬───────────────────────────┘
                          │
                          ▼
                 ┌────────────────────┐
                 │  Training Dataset  │
                 │  (verified only)   │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │  spaCy/HuggingFace │
                 │  Model Retraining  │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │  election_accuracy │
                 │  _model.pt (local) │
                 └────────────────────┘
```

### Local-Only Guarantees

**No External APIs:**

- All NLP/ML uses local spaCy models (`en_core_web_sm`, `en_core_web_lg`)
- Sentence embeddings via local `sentence-transformers` models
- Training/inference runs on local CPU/GPU

**Data Privacy:**

- Election data never leaves local environment/Azure webapp
- No telemetry sent to third-party services
- Database credentials stored in `.env` (never committed)

---

## 9. Troubleshooting

### Cache Issues

**Problem:** Parser uses stale context from cache
**Solution:**

```bash
export FLUSH_CACHE=true
python automate.py
```

**Problem:** `embedding_disk_cache.pkl` corrupted
**Solution:**

```bash
rm webapp/parser/Context_Integration/Context_Library/cache/embedding_disk_cache.pkl
# Cache will rebuild on next run
```

### Log Issues

**Problem:** JSONL file contains duplicate entries
**Solution:** Run deduplication manually:

```python
from webapp.parser.health.log_cache_cleaner_bot import deduplicate_jsonl
deduplicate_jsonl("log/spacy_ner_train_data.jsonl")
```

**Problem:** Log file size > 1GB (slow reads)
**Solution:** Archive old logs and rotate:

```bash
cd logs
gzip spacy_ner_train_data.jsonl
mv spacy_ner_train_data.jsonl.gz archive/
touch spacy_ner_train_data.jsonl
```

### Database Issues

**Problem:** Migration fails with schema mismatch
**Solution:** Check `migration_state.json` and reset:

```bash
rm webapp/parser/Context_Integration/Context_Library/migration_state.json
python -m webapp.parser.Context_Integration.context_migration
```

**Problem:** Slow queries on `election_results`
**Solution:** Add indexes:

```sql
CREATE INDEX idx_state_county ON election_results(state, county);
CREATE INDEX idx_contest ON election_results(contest);
CREATE INDEX idx_ingestion_date ON election_results(ingestion_date);
```

---

## 10. Future Enhancements

### Planned Features

1. **Automatic Log Rotation** (`LOG_ROTATION_SIZE_MB`, `LOG_RETENTION_DAYS`)
2. **Cache TTL** (expire cache entries older than `CACHE_EXPIRE_DAYS`)
3. **Incremental DB Backups** (daily pg_dump to S3/Azure Blob)
4. **Real-time Replication** (PostgreSQL streaming replication for HA)
5. **ML Model Versioning** (track model checkpoints with metadata)

### Research Directions

1. **Active Learning Loop:**
   - Flag low-confidence predictions for manual review
   - User corrections feed back into training dataset
   - Incremental model updates (avoid full retraining)

2. **Federated Learning:**
   - Share anonymized training signals across deployments
   - Preserve data privacy while improving collective accuracy

3. **HuggingFace Fine-tuning:**
   - Train custom BERT/RoBERTa models on election domain
   - Use verified DB data as gold standard for fine-tuning

---

## 11. Quick Reference

### Key Environment Variables

| Variable | Default | Purpose |
| ---------- | --------- | --------- |
| `FLUSH_CACHE` | `false` | Clear cache before pipeline run |
| `CACHE_EXPIRE_DAYS` | `7` | Auto-expire cache entries (days) |
| `INTEGRITY_CHECK` | `false` | Run integrity checks during pipeline |
| `EXPORT_AUDIT_LOG` | `""` | Export audit log to specified path |
| `EMBEDDING_CACHE_PRECHECK` | `true` | Emit startup cache gate summary |
| `EMBEDDING_CACHE_SEED_ON_START` | `false` | Seed disk/memory cache from DB on startup |
| `EMBEDDING_CACHE_SEED_LIMIT` | `250` | Max DB embeddings loaded during startup seed |
| `EMBEDDING_CACHE_CHECKPOINT_WRITES` | `250` | Save disk cache after N cache mutations |
| `EMBEDDING_CACHE_CHECKPOINT_SECONDS` | `120` | Save disk cache after N seconds since last save |
| `EMBEDDING_CACHE_DISK_WARN_MB` | `512` | Warn when disk cache grows beyond threshold |

### Key Commands

```bash
# Run full pipeline with cache flush
export FLUSH_CACHE=true
python automate.py

# Manual cache cleanup
rm -rf webapp/parser/Context_Integration/Context_Library/cache/*

# Manual log deduplication
python -c "from webapp.parser.health.log_cache_cleaner_bot import deduplicate_jsonl; deduplicate_jsonl('log/spacy_ner_train_data.jsonl')"

# Manual context migration
python -m webapp.parser.Context_Integration.context_migration

# Database backup
pg_dump -U postgres smart_elections > backup.sql
```

---

## Contacts & Ownership

**Maintainer:** Smart Elections Parser Team
**Last Updated:** 2026-02-14
**Related Docs:**

- [SELENIUM_NLP_INTEGRATION.md](./SELENIUM_NLP_INTEGRATION.md) – Selenium NLP training data collection
- [STATE_HANDLER_INTEGRATION.md](../STATE_HANDLER_INTEGRATION.md) – State/county detection logic
- [TECHNICAL-REFERENCE.md](../TECHNICAL-REFERENCE.md) – Full API reference

---

***End of Storage Architecture Documentation***

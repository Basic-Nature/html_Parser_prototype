# ML Training Enhancements Setup Guide

**Status**: Production-Ready  
**Last Updated**: February 2026  
**Related Docs**: [NLP_ML_TRAINING_ASSESSMENT.md](NLP_ML_TRAINING_ASSESSMENT.md), [STORAGE_ARCHITECTURE.md](STORAGE_ARCHITECTURE.md)

---

## Overview

This guide documents the recently implemented ML training enhancements for the Smart Elections Parser:

1. **BERT/RoBERTa NER Fine-Tuning** – HuggingFace Transformers integration for custom entity recognition
2. **QA Panel → NER Training Data Pipeline** – User corrections feed ML training loop
3. **Test Dataset Split** – 80/20 train/test split for precision/recall evaluation
4. **Manual Review Bot Integration** – REVIEW_WITH_MANUAL_BOT flag gates verified training data

---

## Quick Start

### 1. Enable Manual Review Bot

Add to `.env` or set environment variable:

```bash
# Gate ML training data quality with manual review
REVIEW_WITH_MANUAL_BOT=true

# Enable BERT fine-tuning (requires HuggingFace Transformers)
ENABLE_BERT_NER_FINETUNING=true

# BERT training hyperparameters (optional)
BERT_NER_EPOCHS=3
BERT_NER_BATCH_SIZE=16
BERT_NER_LEARNING_RATE=2e-5
BERT_NER_BASE_MODEL=dslim/bert-base-NER
```

### 2. Install HuggingFace Dependencies

```bash
pip install transformers datasets torch
```

Or add to `requirements.txt`:

```txt
transformers>=4.30.0
datasets>=2.14.0
torch>=2.0.0
```

### 3. Create PostgreSQL Tables (if not exist)

The following tables are required for QA panel integration:

```sql
-- Data Assurance Classifications (QA Panel)
CREATE TABLE IF NOT EXISTS data_assurance_classifications (
    dataset_id TEXT PRIMARY KEY,
    dl_status TEXT NOT NULL,  -- 'DL1', 'DL2', 'REJECTED', 'DISPUTED'
    confidence_score FLOAT NOT NULL,
    detected_issues JSONB,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    promoted_at TIMESTAMP,
    reviewer_principal TEXT
);

-- NER Training Data (ML Pipeline)
CREATE TABLE IF NOT EXISTS ner_training_data (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    entities JSONB NOT NULL,  -- [{"start": 0, "end": 8, "label": "PERSON"}, ...]
    source TEXT,  -- 'qa_panel_{dataset_id}', 'selenium', 'manual'
    verified BOOLEAN DEFAULT FALSE,  -- TRUE after manual review
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 4. Run Health Check with Retraining

```bash
python automate.py
```

Or manually:

```bash
# Create test dataset split (80/20 train/test)
python -m webapp.parser.health.create_test_dataset

# Retrain models (includes BERT if ENABLE_BERT_NER_FINETUNING=true)
python -m webapp.parser.health.retrain_table_structure_models
```

---

## Architecture

### Data Flow: QA Panel → NER Training → BERT Fine-Tuning

```branch
┌─────────────────────┐
│  User Reviews Data  │
│   (QA Panel UI)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│ POST /api/data-assurance/parse-and-classify │
│ - Auto QA checks (missing headers, etc.)    │
│ - Stores as DL1 in data_assurance_classifications
│ - Extracts text samples → ner_training_data │
│   (verified=FALSE, awaiting review)         │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│ POST /api/data-assurance/verify-and-promote │
│ - User approves DL1 → DL2                   │
│ - Updates ner_training_data: verified=TRUE  │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  Automated Retraining (automate.py)         │
│  - create_test_dataset: 80/20 split         │
│  - retrain_spacy_ner_advanced: spaCy NER    │
│  - retrain_sentence_transformer: SBERT      │
│  - fine_tune_bert_ner: BERT/RoBERTa         │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  Improved Entity Recognition in Production  │
└─────────────────────────────────────────────┘
```

### Storage Tiers

| Tier | Location | Purpose | Retention |
| ------ | ---------- | --------- | ----------- |
| **Cache** | In-memory (session) | Fast lookups during parse | Session-scoped |
| **Log** | `log/*.jsonl` | Training data backup | Persistent, append-only |
| **Database** | PostgreSQL | Source of truth for verified data | Persistent, queryable |

See [STORAGE_ARCHITECTURE.md](STORAGE_ARCHITECTURE.md) for full details.

---

## Components

### 1. BERT/RoBERTa Fine-Tuning

**File**: `webapp/parser/health/fine_tune_bert_ner.py`

**Capabilities**:

- Loads verified NER data from PostgreSQL (`verified=TRUE`)
- Falls back to JSONL logs if DB is empty
- Tokenizes with HuggingFace `AutoTokenizer`
- Fine-tunes `dslim/bert-base-NER` (or custom base model)
- Saves fine-tuned model to `MODEL_DIR/fine_tuned_bert_ner_production`

**Entity Labels**:

```python
ELECTION_ENTITY_LABELS = [
    "O",  # Outside
    "B-PERSON", "I-PERSON",
    "B-ORG", "I-ORG",
    "B-GPE", "I-GPE",
    "B-CONTEST", "I-CONTEST",
    "B-PARTY", "I-PARTY",
    "B-DISTRICT", "I-DISTRICT",
]
```

**Usage**:

```bash
# Standalone
python -m webapp.parser.health.fine_tune_bert_ner

# Integrated (via retrain_table_structure_models)
ENABLE_BERT_NER_FINETUNING=true python -m webapp.parser.health.retrain_table_structure_models
```

**Performance** (estimated, see [NLP_ML_TRAINING_ASSESSMENT.md](NLP_ML_TRAINING_ASSESSMENT.md)):

- **Training Time**: ~15-30 min for 1000 examples (3 epochs, GPU)
- **Model Size**: ~400 MB (BERT-base)
- **Inference Speed**: ~50-100 docs/sec (GPU), ~10-20 docs/sec (CPU)

---

### 2. QA Panel API Endpoints

**File**: `webapp/Smart_Elections_Parser_Webapp.py`

#### `POST /api/data-assurance/parse-and-classify`

Classify parsed election data as DL1 with auto QA checks.

**Request**:

```json
{
  "metadata": {
    "state": "GA",
    "county": "Fulton",
    "contest": "Governor"
  },
  "parsed_data": {
    "headers": ["Candidate", "Party", "Votes"],
    "rows": [
      ["John Doe", "DEM", "12345"],
      ["Jane Smith", "REP", "23456"]
    ]
  }
}
```

**Response**:

```json
{
  "dataset_id": "a1b2c3d4e5f6",
  "dl_status": "DL1",
  "confidence_score": 95.0,
  "detected_issues": [
    {
      "issue_type": "column_mismatch",
      "severity": "WARNING",
      "description": "2 rows have mismatched column counts",
      "affected_rows": 2
    }
  ],
  "created_at": "2026-02-15T10:30:00Z"
}
```

**QA Checks**:

- Missing headers → ERROR (-30 confidence)
- Empty data → CRITICAL (-40 confidence)
- Column count mismatch → WARNING (-0.5 per row, max -20)

**Side Effects**:

- Stores classification in `data_assurance_classifications`
- If `REVIEW_WITH_MANUAL_BOT=true`: Writes text samples to `ner_training_data` (verified=FALSE)

---

#### `POST /api/data-assurance/verify-and-promote`

Promote verified dataset from DL1 to DL2 after manual review.

**Request**:

```json
{
  "dataset_id": "a1b2c3d4e5f6",
  "certification_reason": "Manually verified all values"
}
```

**Response**:

```json
{
  "dataset_id": "a1b2c3d4e5f6",
  "dl_status": "DL2",
  "confidence_score": 95.0,
  "detected_issues": [],
  "created_at": "2026-02-15T10:30:00Z",
  "promoted_at": "2026-02-15T11:00:00Z",
  "reviewer_principal": "jdoe@example.com"
}
```

**Side Effects**:

- Updates `data_assurance_classifications.dl_status` to 'DL2'
- Sets `ner_training_data.verified=TRUE` for all entries with source `qa_panel_{dataset_id}`

---

#### `GET /api/data-assurance/pending-dl2-reviews`

Fetch pending DL2 reviews (DL1 datasets awaiting manual verification).

**Query Params**:

- `limit`: Max results (default: 50)

**Response**:

```json
{
  "pending_reviews": [
    {
      "dataset_id": "a1b2c3d4e5f6",
      "dl_status": "DL1",
      "confidence_score": 95.0,
      "detected_issues": [],
      "metadata": {"state": "GA", "county": "Fulton"},
      "created_at": "2026-02-15T10:30:00Z"
    }
  ]
}
```

---

### 3. Test Dataset Split

**File**: `webapp/parser/health/create_test_dataset.py`

**Capabilities**:

- Loads verified NER data from PostgreSQL or JSONL fallback
- Splits 80/20 (train/test) with stratification by entity types
- Saves to `log/test_datasets/ner_train.jsonl` and `ner_test.jsonl`
- Computes entity distribution statistics

**Environment Variables**:

- `TEST_SPLIT_RATIO=0.2` (default: 20% for testing)
- `MIN_TEST_SAMPLES=50` (minimum test samples required)

**Usage**:

```bash
python -m webapp.parser.health.create_test_dataset
```

**Output**:

```shell
[TEST_SPLIT] Loaded 1000 verified training examples from DB
[TEST_SPLIT] Split: 800 train, 200 test
[TEST_SPLIT] Saved 800 training examples to log/test_datasets/ner_train.jsonl
[TEST_SPLIT] Saved 200 test examples to log/test_datasets/ner_test.jsonl
[TEST_SPLIT] Dataset Statistics:
  Total examples: 1000
  Training: 800
  Testing: 200
  Train entity distribution: {'PERSON': 450, 'ORG': 320, 'GPE': 180}
  Test entity distribution: {'PERSON': 110, 'ORG': 78, 'GPE': 45}
```

---

## Operational Workflows

### Workflow 1: User Corrects Data via QA Panel

1. **User reviews parsed data** in QA panel UI
2. **Frontend calls** `POST /api/data-assurance/parse-and-classify`
3. **Backend auto-QA** checks for common issues
4. **Backend writes** text samples to `ner_training_data` (verified=FALSE)
5. **User approves** data as DL2
6. **Frontend calls** `POST /api/data-assurance/verify-and-promote`
7. **Backend marks** `ner_training_data.verified=TRUE`
8. **Next retraining run** includes verified data

### Workflow 2: Automated Retraining (Daily CI/CD)

```bash
# In automate.py or cron job
python -m webapp.parser.health.create_test_dataset
python -m webapp.parser.health.retrain_table_structure_models
```

**Pipeline Steps**:

1. **create_test_dataset**: Split verified data 80/20
2. **retrain_spacy_ner_advanced**: Update spaCy NER model
3. **retrain_sentence_transformer**: Update SBERT embeddings
4. **fine_tune_bert_ner** (if `ENABLE_BERT_NER_FINETUNING=true`): Fine-tune BERT
5. **cluster_container_patterns**: Group similar table structures

**Gating**:

- BERT fine-tuning skipped if `ENABLE_BERT_NER_FINETUNING=false` (default for CI speed)
- All retraining skips if `REVIEW_WITH_MANUAL_BOT=false` and no verified data exists

### Workflow 3: Model Evaluation

```python
# After retraining, evaluate on test set
from webapp.parser.health.fine_tune_bert_ner import fine_tune_bert_ner

# Load test dataset
import orjson
with open("log/test_datasets/ner_test.jsonl", "r") as f:
    test_data = [orjson.loads(line) for line in f]

# Compute precision/recall/F1
# (TODO: Implement evaluation metrics in fine_tune_bert_ner.py)
```

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
| ---------- | --------- | ------------- |
| `REVIEW_WITH_MANUAL_BOT` | `false` | Gate ML training data with manual review |
| `ENABLE_BERT_NER_FINETUNING` | `false` | Enable HuggingFace BERT fine-tuning |
| `BERT_NER_EPOCHS` | `3` | BERT training epochs |
| `BERT_NER_BATCH_SIZE` | `16` | BERT batch size |
| `BERT_NER_LEARNING_RATE` | `2e-5` | BERT learning rate |
| `BERT_NER_BASE_MODEL` | `dslim/bert-base-NER` | HuggingFace base model |
| `TEST_SPLIT_RATIO` | `0.2` | Test dataset split ratio |
| `MIN_TEST_SAMPLES` | `50` | Minimum test samples required |

### Database Schema

See [Setup Guide](#3-create-postgresql-tables-if-not-exist) for table definitions.

Key columns:

- `data_assurance_classifications.dl_status`: 'DL1', 'DL2', 'REJECTED', 'DISPUTED'
- `ner_training_data.verified`: Boolean (TRUE after manual review)
- `ner_training_data.source`: Tracks data provenance (e.g., 'qa_panel_{dataset_id}')

---

## Troubleshooting

### BERT Fine-Tuning Fails

**Symptoms**: `fine_tune_bert_ner()` raises exception during retraining

**Causes**:

1. Missing HuggingFace dependencies (`transformers`, `datasets`, `torch`)
2. Insufficient VRAM (BERT-base requires ~2GB GPU memory)
3. No verified training data in PostgreSQL

**Solutions**:

```bash
# Install dependencies
pip install transformers datasets torch

# Check for verified data
psql -d smart_elections -c "SELECT COUNT(*) FROM ner_training_data WHERE verified=TRUE;"

# Disable BERT if not needed (fallback to spaCy)
ENABLE_BERT_NER_FINETUNING=false python automate.py
```

### QA Panel Endpoints Return 500

**Symptoms**: Browser console shows `API error: 500 Internal Server Error`

**Causes**:

1. PostgreSQL tables not created
2. Database connection failure
3. Missing `orjson` import

**Solutions**:

```bash
# Create tables manually
psql -d smart_elections -f migrations/create_qa_tables.sql

# Check Flask logs for detailed error
tail -f log/session_*.log

# Test DB connection
python -c "from webapp.parser.utils.db_utils import SessionLocal; SessionLocal()"
```

### Test Dataset Split Reports Insufficient Data

**Symptoms**: `[TEST_SPLIT] Insufficient data (30 samples). Need at least 50 for reliable evaluation.`

**Causes**:

1. No verified data in PostgreSQL
2. No JSONL fallback data

**Solutions**:

```bash
# Verify data exists
psql -d smart_elections -c "SELECT COUNT(*) FROM ner_training_data WHERE verified=TRUE;"

# Check JSONL logs
ls -lh log/spacy_ner_train_data.jsonl

# Manually promote some QA panel data to DL2 to generate verified data
# (use QA panel UI or direct SQL UPDATE)
```

### REVIEW_WITH_MANUAL_BOT Has No Effect

**Symptoms**: NER training data is written even when `REVIEW_WITH_MANUAL_BOT=false`

**Causes**:

1. `.env` file not loaded (using hardcoded defaults)
2. Environment variable typo (case-sensitive)

**Solutions**:

```bash
# Check if .env is loaded
python -c "from webapp.parser.config import REVIEW_WITH_MANUAL_BOT; print(REVIEW_WITH_MANUAL_BOT)"

# Set explicitly in shell (Windows PowerShell)
$env:REVIEW_WITH_MANUAL_BOT="true"
python automate.py

# Set explicitly in shell (Linux/Mac)
REVIEW_WITH_MANUAL_BOT=true python automate.py
```

---

## Performance Benchmarks

See [NLP_ML_TRAINING_ASSESSMENT.md](NLP_ML_TRAINING_ASSESSMENT.md) for full benchmarks.

**Key Metrics** (1000 verified examples, 3 epochs):

| Model | Training Time (GPU) | Training Time (CPU) | Model Size | Inference Speed (GPU) |
| ------- | --------------------- | --------------------- | ------------ | ---------------------- |
| spaCy NER | 5-10 min | 20-40 min | 50-100 MB | 100-200 docs/sec |
| SBERT | 10-20 min | 40-80 min | 400-500 MB | 50-100 docs/sec |
| BERT-base | 15-30 min | 60-120 min | 400-500 MB | 50-100 docs/sec |

**Recommendations**:

- **CI/CD**: Keep `ENABLE_BERT_NER_FINETUNING=false` to save time (use spaCy only)
- **Production**: Enable BERT fine-tuning weekly/monthly for best accuracy
- **Development**: Use test dataset split to validate improvements before production deploy

---

## Future Enhancements

### Phase 3 (Planned)

1. **Evaluation Dashboard**
   - Real-time precision/recall/F1 tracking
   - Per-entity-type metrics
   - Confusion matrix visualization

2. **Active Learning**
   - Auto-flag low-confidence predictions for manual review
   - Prioritize high-value corrections (rare entities, borderline cases)

3. **Multi-Model Ensemble**
   - Combine spaCy + BERT predictions (voting/averaging)
   - Confidence-weighted entity extraction

4. **Integration Tests**
   - End-to-end QA panel workflow (Playwright/Selenium)
   - Headless CI checks for API endpoints

### Contributions

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for PR guidelines.

---

## References

- [NLP_ML_TRAINING_ASSESSMENT.md](NLP_ML_TRAINING_ASSESSMENT.md) – Full ML training framework assessment
- [STORAGE_ARCHITECTURE.md](STORAGE_ARCHITECTURE.md) – Cache/Log/DB data flow
- [SELENIUM_NLP_INTEGRATION.md](SELENIUM_NLP_INTEGRATION.md) – Selenium as NLP training data collector
- [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers)
- [spaCy NER Training](https://spacy.io/usage/training#ner)

---

**Last Validated**: Feb 2026  
**Contact**: See [CODEOWNERS](../../.github/CODEOWNERS)

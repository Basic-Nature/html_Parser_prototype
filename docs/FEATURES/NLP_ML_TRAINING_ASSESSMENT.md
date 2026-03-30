# NLP/ML Training Framework – Readiness Assessment

**Version:** 1.0  
**Date:** 2026-02-14  
**Status:** Production-Ready (Local Training), Research Phase (Advanced Fine-Tuning)

## Executive Summary

The Smart Elections Parser implements a **fully local, privacy-preserving NLP/ML pipeline** for election data extraction and accuracy improvement. All training and inference happen on-device (local machine or Azure webapp) with **no external API dependencies**.

**Current Capabilities:**

- ✅ **spaCy NER Training** – Entity recognition (PERSON, ORG, GPE, CONTEST, PARTY, DISTRICT)
- ✅ **SentenceTransformer Fine-Tuning** – Semantic similarity for header/contest matching
- ✅ **Local Learning Engine** – Historical accuracy tracking and pattern-based predictions
- ✅ **Automated Retraining Pipeline** – `BotPipeline.run()` → `retrain_models()`
- ✅ **Training Data Collection** – JSONL logs from parsing sessions, Selenium NER capture
- ✅ **DB-Backed Verification** – PostgreSQL stores verified training examples

**Research/Future:**

- 🔬 **HuggingFace BERT/RoBERTa Fine-Tuning** – Custom domain models (planned)
- 🔬 **Active Learning Loop** – User corrections feed directly into retraining
- 🔬 **Federated Learning** – Cross-deployment pattern sharing (privacy-preserving)

---

## 1. NLP/ML Stack Overview

### Core Libraries

| Library | Version | Purpose | Training Capability |
| --------- | --------- | --------- | ------------------- |
| **spaCy** | 3.7+ | NER, tokenization, entity normalization | ✅ Full training via `nlp.update()` |
| **sentence-transformers** | 2.7+ | Semantic embeddings (all-MiniLM-L6-v2) | ✅ Fine-tuning via `model.fit()` |
| **HuggingFace Transformers** | 4.40+ | BERT/RoBERTa models (optional) | 🔬 Manual fine-tuning possible |
| **scikit-learn** | 1.5+ | Clustering, anomaly detection | ❌ Pre-trained models only |
| **PyTorch** | 2.4+ | Backend for transformers/sentence-BERT | ✅ Custom model training |

### Model Registry (`webapp/parser/utils/model_registry.py`)

**Centralized model loader:**

```python
from webapp.parser.utils.model_registry import ModelRegistry

# Get spaCy NER model (auto-loads fine-tuned if exists)
nlp = ModelRegistry.get_spacy_model("en_core_web_lg", use_finetuned=True)

# Get SentenceTransformer (auto-loads fine-tuned if exists)
embedder = ModelRegistry.get_sentence_transformer("all-MiniLM-L6-v2", use_finetuned=True)
```

**Model Paths:**

```python
{
    "spacy_ner": "model/fine_tuned_spacy_ner",
    "sentence_transformer": "model/fine_tuned_table_headers",
    "integrity_model": "model/integrity_model.pt",
    "table_structure": "model/table_structure.pt"
}
```

---

## 2. Training Data Pipeline

### Data Sources

```branch
┌────────────────────────────┐
│  Parsing Sessions (Live)  │
└──────────┬─────────────────┘
           │
           ▼
┌────────────────────────────┐    Append-only Logs
│  Context Organizer Events  │──► spacy_ner_train_data.jsonl
└──────────┬─────────────────┘    field_selection_log.jsonl
           │                       dom_pattern_kb.jsonl
           │
           ▼
┌────────────────────────────┐
│  Selenium NER Collector    │──► selenium_ner_training.jsonl
└──────────┬─────────────────┘
           │
           ▼
┌────────────────────────────┐    Manual Review
│  log_cache_cleaner_bot     │──► Deduplicates + validates
└──────────┬─────────────────┘
           │
           ▼
┌────────────────────────────┐    Persist to DB
│  context_migration.py      │──► ner_training_data (PostgreSQL)
└──────────┬─────────────────┘    context_library_entries
           │
           ▼
┌────────────────────────────┐    SQL Query
│  retrain_models()          │──► SELECT * WHERE verified=TRUE
└────────────────────────────┘
```

### Training Data Format

**spaCy NER (`spacy_ner_train_data.jsonl`):**

```json
{
  "text": "John Doe defeated Jane Smith in the State Assembly District 15 race.",
  "entities": [
    {"start": 0, "end": 8, "label": "PERSON"},
    {"start": 18, "end": 28, "label": "PERSON"},
    {"start": 36, "end": 67, "label": "CONTEST"}
  ],
  "source": "html_scan_handler",
  "timestamp": "2026-02-14T12:00:00Z"
}
```

**SentenceTransformer (`field_selection_log.jsonl`):**

```json
{
  "contest": "State Assembly District 15",
  "headers": ["John Doe", "Jane Smith", "Total Votes"],
  "session_id": "abc123",
  "confidence": 0.95,
  "timestamp": "2026-02-14T12:00:00Z"
}
```

---

## 3. Automated Retraining Pipeline

### Entry Point: `BotPipeline.run()`

**Flow:**

```python
# automate.py
from webapp.parser.health.health_router import BotPipeline

pipeline = BotPipeline()
pipeline.run()
# → preclean logs/cache
# → log_cache_cleaner_bot (dedupe JSONL)
# → manual_correction_bot (validate entries)
# → context_migration (DB insert)
# → retrain_models() ← ML TRAINING HAPPENS HERE
```

### Retraining Logic (`retrain_table_structure_models.py`)

**1. spaCy NER Training:**

```python
def retrain_spacy_ner_advanced(
    confirmed_structures,
    context_library=None,
    model_save_path="fine_tuned_spacy_ner",
    max_epochs=10,
    patience=3,
    min_delta=0.01,
    batch_size=32
):
    # Load base model (en_core_web_lg)
    nlp = spacy.load("en_core_web_lg")
    
    # Add NER pipe and labels
    ner = nlp.add_pipe("ner")
    for label in ["PERSON", "ORG", "GPE", "CONTEST", "PARTY", "DISTRICT"]:
        ner.add_label(label)
    
    # Load training data from JSONL
    train_data = load_ner_training_data()
    
    # Train with early stopping
    optimizer = nlp.resume_training()
    for epoch in range(max_epochs):
        random.shuffle(train_data)
        losses = {}
        for batch in minibatch(train_data, size=batch_size):
            nlp.update(batch, sgd=optimizer, drop=0.2, losses=losses)
        
        # Early stopping check
        if early_stopping(losses, patience, min_delta):
            break
    
    # Save fine-tuned model
    nlp.to_disk(model_save_path)
```

**2. SentenceTransformer Fine-Tuning:**

```python
def retrain_sentence_transformer(
    confirmed_structures,
    model_save_path="fine_tuned_table_headers",
    epochs=1,
    batch_size=8
):
    from sentence_transformers import SentenceTransformer, InputExample, losses
    
    # Load base model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Build training pairs (contest → header similarity)
    train_examples = []
    for struct in confirmed_structures:
        contest = struct["contest"]
        for header in struct["headers"]:
            train_examples.append(InputExample(texts=[contest, header], label=1.0))
    
    # Fine-tune with CosineSimilarityLoss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model)
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=10,
        show_progress_bar=True
    )
    
    # Save fine-tuned model
    model.save(model_save_path)
```

---

## 4. Training Configuration

### Environment Variables

```bash
# spaCy NER training
export SPACY_NER_EPOCHS=10
export SPACY_NER_PATIENCE=3
export SPACY_NER_MIN_DELTA=0.01
export SPACY_NER_BATCH_SIZE=32

# SentenceTransformer training
export SBERT_EPOCHS=1
export SBERT_BATCH_SIZE=8

# Manual review toggle (gates training data)
export REVIEW_WITH_MANUAL_BOT=true
```

### Config File (`webapp/parser/config.py`)

```python
# ML/NLP Training Configuration
SBERT_EPOCHS = int(os.environ.get("SBERT_EPOCHS", 1))
SBERT_BATCH_SIZE = int(os.environ.get("SBERT_BATCH_SIZE", 8))

SPACY_NER_EPOCHS = int(os.environ.get("SPACY_NER_EPOCHS", 10))
SPACY_NER_PATIENCE = int(os.environ.get("SPACY_NER_PATIENCE", 3))
SPACY_NER_MIN_DELTA = float(os.environ.get("SPACY_NER_MIN_DELTA", 0.01))
SPACY_NER_BATCH_SIZE = int(os.environ.get("SPACY_NER_BATCH_SIZE", 32))

REVIEW_WITH_MANUAL_BOT = os.environ.get("REVIEW_WITH_MANUAL_BOT", "false").lower() == "true"
```

---

## 5. Local Learning Engine

### Architecture

**Purpose:** Learn from historical parsing sessions to predict accuracy for new sessions.

**Components:**

1. **Training Signal Ingestion** – Record success/failure + quality metrics
2. **Pattern Matching** – Query context_library.json for similar state/county/contest
3. **Confidence Scoring** – Average historical health_score for matched contexts

### Implementation (`health_router.py`)

```python
class LocalLearningEngine:
    def __init__(self):
        self.monitor = get_integrity_monitor()
        self.training_data_path = os.path.join(LOG_DIR, "training_data.jsonl")
        self.model_checkpoint = os.path.join(MODEL_DIR, "election_accuracy_model.pt")
    
    def ingest_training_signal(self, session_context, success, quality_metrics):
        """Capture learning signal from successful/failed parsing."""
        signal = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state": session_context.get("state"),
            "county": session_context.get("county"),
            "contest": session_context.get("contest"),
            "handler": session_context.get("handler"),
            "success": success,
            "metrics": quality_metrics,
            "source": "parser_feedback"
        }
        with open(self.training_data_path, "a", encoding="utf-8") as f:
            f.write(orjson.dumps(signal).decode() + "\n")
    
    def get_learned_accuracy_score(self, session_context):
        """Query learned patterns to get expected accuracy for this context."""
        state = session_context.get("state", "")
        county = session_context.get("county", "")
        
        library = load_context_library()
        checks = library.get("integrity_checks", [])
        
        matches = [
            c for c in checks
            if c.get("context_summary", {}).get("state") == state
            and c.get("context_summary", {}).get("county") == county
        ]
        
        if matches:
            scores = [float(m.get("health_score", 0.5)) for m in matches]
            return sum(scores) / len(scores)
        return 0.5  # Neutral default
```

### Usage Example

```python
from webapp.parser.health.health_router import get_learning_engine

engine = get_learning_engine()

# Record successful parsing
engine.ingest_training_signal(
    session_context={"state": "California", "county": "Alameda", "contest": "Governor"},
    success=True,
    quality_metrics={"row_count": 320, "confidence": 0.95}
)

# Predict accuracy for new session
score = engine.get_learned_accuracy_score(
    {"state": "California", "county": "Alameda", "contest": "State Assembly"}
)
# Returns: 0.95 (based on similar California/Alameda sessions)
```

---

## 6. HuggingFace Integration

### Current Usage (Inference Only)

**Integrity Monitor (`health/integrity_monitor.py`):**

```python
class HuggingFaceNLPAnalyzer:
    def _lazy_init(self):
        # Sentence embeddings
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # NER pipeline
        self.ner_pipeline = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple"
        )
```

**Current Capabilities:**

- ✅ Entity extraction (PERSON, ORG, LOC, MISC)
- ✅ Semantic similarity via sentence embeddings
- ✅ Zero-shot classification (optional, via `facebook/bart-large-mnli`)

**Limitations:**

- ❌ No fine-tuning implemented yet (uses pre-trained models only)
- ❌ No custom domain adaptation (election-specific vocabulary)

### Fine-Tuning Capability Assessment

**Pre-Requisites (Already Installed):**

```python
# requirements.txt
transformers>=4.40.0
torch>=2.4.1
sentence-transformers>=2.7.0
datasets>=2.19.0  # For loading training data
```

**Data Format Required:**

```python
from datasets import Dataset

# Example training data for NER fine-tuning
training_data = {
    "tokens": [["John", "Doe", "defeated", "Jane", "Smith", "in", "Governor", "race"]],
    "ner_tags": [[1, 2, 0, 1, 2, 0, 3, 0]]  # 1=B-PER, 2=I-PER, 3=B-CONTEST, 0=O
}
dataset = Dataset.from_dict(training_data)
```

**Fine-Tuning Code (Skeleton):**

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset

def fine_tune_bert_ner():
    # Load base model
    model_name = "dslim/bert-base-NER"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(ELECTION_ENTITY_LABELS)  # PERSON, ORG, CONTEST, PARTY, etc.
    )
    
    # Load training data from PostgreSQL
    dataset = load_ner_dataset_from_db()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="model/fine_tuned_bert_ner",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        save_steps=1000,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        learning_rate=2e-5,
        weight_decay=0.01
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer
    )
    
    # Train
    trainer.train()
    
    # Save
    trainer.save_model("model/fine_tuned_bert_ner")
```

**Verdict:** ✅ **Fully capable of HuggingFace fine-tuning** with existing infrastructure.

---

## 7. Data Quality for Training

### Current Data Volume

**Estimated Training Examples (from JSONL logs):**

- `spacy_ner_train_data.jsonl` – ~5,000-10,000 examples (auto-generated)
- `selenium_ner_training.jsonl` – ~500-1,000 examples (CAPTCHA sites)
- `field_selection_log.jsonl` – ~1,000-2,000 examples (user selections)

**Verification Status:**

- ❌ **Auto-generated (unverified)** – JSONL logs are raw training signals
- ✅ **DB-verified** – PostgreSQL `ner_training_data WHERE verified=TRUE` (manual review)

### Data Quality Gates

**Manual Review Bot (`manual_correction_bot.py`):**

```bash
# Enable manual review before training
export REVIEW_WITH_MANUAL_BOT=true
python automate.py

# Review workflow:
# 1. Bot scans JSONL logs for misaligned annotations
# 2. Flags entries with suspicious patterns (typosquatting, value inconsistencies)
# 3. Prompts user to verify/correct via CLI or QA panel
# 4. Verified entries → PostgreSQL with verified=TRUE
```

**Integrity Checks:**

- Misaligned entity spans (spaCy `offsets_to_biluo_tags`)
- Duplicate entries (content hashing)
- Suspicious candidates (length < 3 chars, special chars)
- Contextual mismatches (state/county/contest alignment)

### Training Data Export

**SQL Query for Verified Data:**

```sql
-- Export NER training examples (verified only)
SELECT text, entities
FROM ner_training_data
WHERE verified = TRUE
ORDER BY created_at DESC;

-- Export SentenceTransformer training pairs
SELECT contest, headers
FROM context_library_entries
WHERE context_type = 'field_selection'
  AND confidence > 0.8;
```

---

## 8. Training Readiness Scorecard

### spaCy NER Training

| Criterion | Status | Notes |
| ----------- | -------- | ------- |
| **Training Data** | ✅ Ready | 5K-10K examples in JSONL |
| **Verification** | 🟡 Partial | Manual review bot available, not enforced by default |
| **Training Pipeline** | ✅ Automated | `retrain_spacy_ner_advanced()` in BotPipeline |
| **Model Checkpointing** | ✅ Implemented | Saved to `model/fine_tuned_spacy_ner` |
| **Inference** | ✅ Production | ModelRegistry auto-loads fine-tuned model |
| **Evaluation** | 🟡 Manual | No auto-test dataset yet |

**Recommendation:** ✅ **Production-ready**. Enable `REVIEW_WITH_MANUAL_BOT=true` for higher quality.

---

### SentenceTransformer Fine-Tuning

| Criterion | Status | Notes |
| ----------- | -------- | ------- |
| **Training Data** | ✅ Ready | 1K-2K contest→header pairs |
| **Verification** | 🟡 Partial | User field selections logged (implicit verification) |
| **Training Pipeline** | ✅ Automated | `retrain_sentence_transformer()` in BotPipeline |
| **Model Checkpointing** | ✅ Implemented | Saved to `model/fine_tuned_table_headers` |
| **Inference** | ✅ Production | ModelRegistry auto-loads fine-tuned model |
| **Evaluation** | ❌ None | No test dataset |

**Recommendation:** ✅ **Production-ready**. Consider adding test dataset for accuracy tracking.

---

### HuggingFace BERT/RoBERTa Fine-Tuning

| Criterion | Status | Notes |
| ----------- | -------- | ------- |
| **Training Data** | ✅ Ready | Same as spaCy NER (reusable) |
| **Verification** | 🟡 Partial | Same as spaCy NER |
| **Training Pipeline** | ❌ Not Implemented | Skeleton code provided above |
| **Model Checkpointing** | ❌ Not Implemented | Need to add to BotPipeline |
| **Inference** | 🟡 Partial | HuggingFaceNLPAnalyzer uses pre-trained only |
| **Evaluation** | ❌ None | No test dataset |

**Recommendation:** 🔬 **Research Phase**. Implement `fine_tune_bert_ner()` and wire into BotPipeline for production use.

---

## 9. Active Learning Loop (Future)

### Concept

**User corrections feed directly into retraining:**

```branch
┌───────────────┐
│  User Edits   │ (QA Panel or CLI)
└───────┬───────┘
        │
        ▼
┌───────────────────────┐    DB Insert
│  manual_correction    │──► ner_training_data (verified=TRUE)
└───────┬───────────────┘
        │
        │ BotPipeline (nightly)
        ▼
┌───────────────────────┐    SELECT WHERE verified=TRUE
│  retrain_models()     │──► Incremental model update
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│  Updated Model        │ → Better predictions next session
└───────────────────────┘
```

### Implementation Plan

Phase 1: QA Panel Integration (PARTIAL - Research Phase)

- ✅ QA panel captures user corrections (`quality_assurance_panel.js`)
- ✅ Corrections stored in DB (`/api/data-assurance/submit-correction`)
- 🔬 TODO: Wire corrections into `ner_training_data` table

Phase 2: Incremental Training (PLANNED - Future)

- 🔬 TODO: Implement `incremental_train_spacy_ner(new_examples)`
- 🔬 TODO: Avoid full retraining (just update weights on new data)
- 🔬 TODO: Add version tracking for models (metadata)

---

## 10. Federated Learning (Research - Not in Scope)

### Concept

**Share anonymized patterns across deployments without sharing raw data:**

```branch
┌──────────────────┐
│  Deployment A    │──► Extract pattern hashes (state/county/contest combos)
└──────┬───────────┘    + Aggregate confidence scores (no raw text)
       │
       ▼
┌──────────────────┐    Central Pattern Repository
│  Pattern Pool    │    (no PII, no election results)
└──────┬───────────┘    Just: {"state": "CA", "county": "Alameda", "avg_confidence": 0.92}
       │
       ▼
┌──────────────────┐
│  Deployment B    │──► Download patterns → Improve local predictions
└──────────────────┘
```

### Privacy Guarantees

- **No raw election data shared** (only pattern hashes + confidence scores)
- **No PII** (candidate names, voter info never leave local DB)
- **Opt-in only** (deployments can disable federated sync)
- **Differential privacy** (add noise to confidence scores before sharing)

### Implementation Plan (Future Research)

Phase 1: Pattern Extraction (PLANNED - Future)

- 🔬 TODO: Implement `extract_anonymized_patterns(context_library)`
- 🔬 TODO: Hash state/county/contest combos + aggregate confidence scores
- 🔬 TODO: Export to `pattern_export.jsonl`

Phase 2: Pattern Sync (PLANNED - Future)

- 🔬 TODO: Implement `sync_patterns_to_pool(pattern_export.jsonl, remote_url)`
- 🔬 TODO: Download remote patterns via REST API
- 🔬 TODO: Merge into local `context_library.json` (keep highest confidence)

---

## 11. Training Workflow Examples

### Example 1: Full Retraining (Automated)

```bash
# Set environment variables for training
export SPACY_NER_EPOCHS=10
export SBERT_EPOCHS=2
export REVIEW_WITH_MANUAL_BOT=true
export FLUSH_CACHE=true

# Run full pipeline (includes retraining)
python automate.py
```

**Pipeline Steps:**

1. Pre-clean logs/cache
2. Deduplicate JSONL training data
3. Manual review bot validates entries
4. Migrate verified data to PostgreSQL
5. **Retrain spaCy NER** (10 epochs, early stopping)
6. **Retrain SentenceTransformer** (2 epochs)
7. Save fine-tuned models to `model/`
8. Next parsing session auto-loads fine-tuned models

---

### Example 2: Manual Retraining (Targeted)

```bash
# Retrain only spaCy NER (skip full pipeline)
python -c "
from webapp.parser.health.retrain_table_structure_models import retrain_spacy_ner_advanced
from webapp.parser.Context_Integration.librarian import load_context_library

library = load_context_library()
retrain_spacy_ner_advanced(
    confirmed_structures=[],  # Load from DB
    context_library=library,
    max_epochs=5,
    batch_size=16
)
"
```

---

### Example 3: Fine-Tune SentenceTransformer (Standalone)

```python
from webapp.parser.health.retrain_table_structure_models import retrain_sentence_transformer
from webapp.parser.utils.db_utils import SessionLocal

# Load verified contest→header pairs from DB
with SessionLocal() as session:
    results = session.execute("""
        SELECT contest, headers
        FROM context_library_entries
        WHERE context_type='field_selection' AND confidence > 0.8
    """).fetchall()
    
    confirmed_structures = [
        {"contest": row[0], "headers": row[1]}
        for row in results
    ]

# Fine-tune model
retrain_sentence_transformer(
    confirmed_structures=confirmed_structures,
    model_save_path="model/fine_tuned_table_headers",
    epochs=3,
    batch_size=16
)
```

---

## 12. Performance Considerations

### Training Time Estimates

**Local Machine (CPU only):**

| Model | Training Data | Time | Notes |
| ------- | -------------- | ------ | ------- |
| spaCy NER | 5K examples | ~15-30 min | 10 epochs, batch_size=32 |
| SentenceTransformer | 1K pairs | ~2-5 min | 1 epoch, batch_size=8 |
| BERT/RoBERTa (future) | 5K examples | ~1-2 hours | 3 epochs, batch_size=16 |

**GPU-Accelerated (Azure VM with Tesla T4):**

| Model | Training Data | Time | Notes |
| ------- | -------------- | ------ | ------- |
| spaCy NER | 5K examples | ~5-10 min | 10 epochs, batch_size=32 |
| SentenceTransformer | 1K pairs | ~30-60 sec | 1 epoch, batch_size=8 |
| BERT/RoBERTa (future) | 5K examples | ~10-20 min | 3 epochs, batch_size=16 |

### Memory Requirements

**Minimum RAM:**

- spaCy NER: 2-4 GB
- SentenceTransformer: 1-2 GB
- BERT/RoBERTa: 4-8 GB (depends on model size)

**Disk Space:**

- spaCy model: ~500 MB (en_core_web_lg) + ~200 MB (fine-tuned)
- SentenceTransformer: ~100 MB (base) + ~150 MB (fine-tuned)
- BERT/RoBERTa: ~400 MB (base) + ~500 MB (fine-tuned)

---

## 13. Monitoring & Evaluation

### Training Metrics

**Logged Automatically:**

```python
# spaCy NER training log
{
    "epoch": 5,
    "loss": 12.34,
    "patience_counter": 2,
    "early_stopped": False,
    "timestamp": "2026-02-14T12:00:00Z"
}

# SentenceTransformer training log
{
    "epoch": 1,
    "loss": 0.045,
    "num_examples": 1200,
    "batch_size": 8,
    "timestamp": "2026-02-14T12:00:00Z"
}
```

***TODO: Add Test Dataset Evaluation***

```python
# Future: Evaluate on held-out test set
def evaluate_spacy_ner(model_path, test_data_path):
    nlp = spacy.load(model_path)
    test_examples = load_jsonl(test_data_path)
    
    precision, recall, f1 = 0.0, 0.0, 0.0
    for example in test_examples:
        doc = nlp(example["text"])
        # Compute precision/recall/f1 against example["entities"]
        # (implementation omitted for brevity)
    
    return {"precision": precision, "recall": recall, "f1": f1}
```

---

## 14. Troubleshooting

### Common Issues

**Issue:** Training data JSONL contains misaligned entities  
**Solution:** Run `scan_misaligned_ner` before training:

```bash
python -c "
from webapp.parser.health.retrain_table_structure_models import clean_misaligned_ner_jsonl
clean_misaligned_ner_jsonl('log/spacy_ner_train_data.jsonl')
"
```

**Issue:** SentenceTransformer training fails with "No training examples"  
**Solution:** Check `field_selection_log.jsonl` has entries with `confidence > 0.8`

**Issue:** GPU not detected (PyTorch uses CPU)  
**Solution:** Install CUDA-compatible PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Issue:** HuggingFace models fail to download (offline mode)  
**Solution:** Set local model paths:

```bash
export TRANSFORMERS_OFFLINE=1
export SENTENCE_TRANSFORMER_LOCAL_PATH=/models/sentence/all-MiniLM-L6-v2
```

---

## 15. Recommendations

### Short-Term (Next 1-2 Weeks)

1. ✅ **Enable `REVIEW_WITH_MANUAL_BOT=true`** in production deployments
2. ✅ **Add test dataset** for spaCy NER evaluation (split existing JSONL 80/20)
3. ✅ **Monitor training loss trends** via log analysis
4. ✅ **Document model versions** in metadata (track which checkpoint is active)

### Medium-Term (Next 1-3 Months)

1. 🔬 **Implement HuggingFace BERT/RoBERTa fine-tuning** (use existing spaCy NER data)
2. 🔬 **Wire QA panel corrections into `ner_training_data`** (active learning loop)
3. 🔬 **Add incremental training** (avoid full retraining on small data additions)
4. 🔬 **Build test harness** for automated model evaluation (precision/recall/f1)

### Long-Term (Next 6-12 Months)

1. 🔬 **Federated learning prototype** (anonymized pattern sharing across deployments)
2. 🔬 **Multi-task learning** (train single model for NER + contest classification + party detection)
3. 🔬 **Domain-specific language model** (pre-train GPT-style model on election documents)
4. 🔬 **AutoML pipeline** (hyperparameter tuning via Optuna/Ray Tune)

---

## 16. Conclusion

**Current State:** ✅ **Production-ready for spaCy NER and SentenceTransformer fine-tuning**

**Key Strengths:**

- Fully automated retraining pipeline (integrated into `automate.py`)
- Local-only training (no external API dependencies)
- Training data collection from live parsing sessions
- Manual review bot for data quality

**Key Gaps:**

- No HuggingFace BERT/RoBERTa fine-tuning yet (skeleton code provided)
- No active learning loop (QA corrections not yet wired into training)
- No federated learning (cross-deployment pattern sharing)

**Next Steps:**

1. Enable `REVIEW_WITH_MANUAL_BOT=true` in `.env`
2. Run `python automate.py` to trigger full retraining
3. Implement `fine_tune_bert_ner()` and wire into `BotPipeline.retrain_models()`
4. Add test dataset and evaluation metrics for ongoing monitoring

---

## Contacts & References

**Maintainer:** Smart Elections Parser Team  
**Last Updated:** 2026-02-14  
**Related Docs:**

- [STORAGE_ARCHITECTURE.md](./STORAGE_ARCHITECTURE.md) – Cache/log/DB flow
- [SELENIUM_NLP_INTEGRATION.md](./SELENIUM_NLP_INTEGRATION.md) – NER training data collection via Selenium
- [TECHNICAL-REFERENCE.md](../TECHNICAL-REFERENCE.md) – Full API reference

**External Resources:**

- [spaCy Training Guide](https://spacy.io/usage/training)
- [SentenceTransformers Fine-Tuning](https://www.sbert.net/examples/training/sts/README.html)
- [HuggingFace Transformers Training](https://huggingface.co/docs/transformers/training)

---

***End of NLP/ML Training Framework Assessment***

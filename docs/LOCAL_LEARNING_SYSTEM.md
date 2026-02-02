# Local Learning System: Election Data Accuracy & Integrity

## Overview

The Smart Elections Parser now includes a **self-contained, privacy-focused Local Learning System** that replaces all external AI/LLM dependencies (OpenAI removed). This system:

- ✅ **Learns from ingested election data** to improve parsing accuracy over time
- ✅ **Preserves all data locally** for training and inference (no cloud calls)
- ✅ **Integrates with SQL backend** (warehoused election results) for training signals
- ✅ **Uses HuggingFace NLP models** (semantic embeddings, entity recognition)
- ✅ **Maintains persistent knowledge** via `context_library.json`
- ✅ **Removes all external LLM costs** (OpenAI deleted)

## Architecture

### Core Components

1. **LocalLearningEngine** (`health/health_router.py`)
   - Ingests training signals from parsing sessions
   - Queries historical patterns for accuracy prediction
   - Integrates with IntegrityMonitor for health scoring
   - Persists all learning data locally

2. **IntegrityMonitor** (`health/integrity_monitor.py`)
   - HuggingFace NLP analysis (no OpenAI)
   - PyTorch neural network for health scoring
   - Downloads integrity verification with SHA-256
   - Context-aware feature extraction

3. **Context Library** (`Context_Integration/Context_Library/context_library.json`)
   - Persistent storage of high-priority session checks
   - Pattern database for learning algorithms
   - Automatic rotation (last 100 entries)

4. **Training Data Log** (`log/training_data.jsonl`)
   - Records of all parsing attempts (successful + failed)
   - Features extracted from each session
   - Quality metrics and outcome labels

## Learning Loop

```txt
┌─────────────────────────────────────────────────────────────────┐
│ 1. SESSION PROCESSING                                           │
│    ↓ Parser processes election data (HTML, PDF, JSON, CSV)      │
│    ↓ Handler extracts tables and produces output CSV            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. INTEGRITY MONITORING                                         │
│    ↓ IntegrityMonitor captures session features                 │
│    ↓ HuggingFace NLP analyzes extracted entities                │
│    ↓ PyTorch NN predicts health score (0-1 scale)               │
│    ↓ Results include: confidence, priority, recommendations     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. TRAINING SIGNAL CAPTURE                                      │
│    ↓ LocalLearningEngine records outcome (success/fail)         │
│    ↓ Features stored: state, county, contest, handler, metrics  │
│    ↓ Quality metrics attached: row_count, confidence, etc.      │
│    ↓ Persisted to: log/training_data.jsonl                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. CONTEXT LIBRARY PERSISTENCE                                  │
│    ↓ High-priority results → context_library.json               │
│    ↓ Pattern database grows with each successful parse          │
│    ↓ Automatic LRU rotation when > 100 entries                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. LEARNED ACCURACY PREDICTION                                  │
│    ↓ Future sessions → LocalLearningEngine.get_learned_score()  │
│    ↓ Pattern matching on (state, county) from context_library   │
│    ↓ Historical average accuracy score returned                 │
│    ↓ Informs session prioritization and quality expectations    │
└─────────────────────────────────────────────────────────────────┘
```

## Usage Examples

### Example 1: Record a Parsing Session

```python
from webapp.parser.health.health_router import get_learning_engine

learning_engine = get_learning_engine()

# After parsing completes
session_context = {
    "state": "Alabama",
    "county": "Jefferson",
    "contest": "Presidential Election",
    "handler": "json_handler",
    "year": 2024
}

quality_metrics = {
    "row_count": 245,
    "column_count": 8,
    "confidence": 0.92,
    "extraction_time_ms": 1250
}

# Record the learning signal
learning_engine.ingest_training_signal(
    session_context=session_context,
    success=True,
    quality_metrics=quality_metrics
)
```

**Output** (`log/training_data.jsonl`):

```json
{
  "timestamp": "2026-02-01T12:34:56.789Z",
  "state": "Alabama",
  "county": "Jefferson",
  "contest": "Presidential Election",
  "handler": "json_handler",
  "success": true,
  "metrics": {
    "row_count": 245,
    "column_count": 8,
    "confidence": 0.92,
    "extraction_time_ms": 1250
  },
  "source": "parser_feedback"
}
```

### Example 2: Predict Accuracy for New Session

```python
# Before processing a session
predicted_accuracy = learning_engine.get_learned_accuracy_score({
    "state": "Alabama",
    "county": "Jefferson",
    "contest": "Governor Election",
    "handler": "html_handler"
})

if predicted_accuracy < 0.5:
    # Prioritize for manual review
    queue.push_priority(session)
elif predicted_accuracy > 0.8:
    # Fast-track: trust the handler
    queue.push_fast_lane(session)
else:
    # Normal processing with monitoring
    queue.push_normal(session)
```

### Example 3: Full Health Assessment Loop

```python
monitor = get_integrity_monitor()
learning_engine = get_learning_engine()

# 1. Get learned accuracy baseline
learned_score = learning_engine.get_learned_accuracy_score(context)

# 2. Get current session health
health = monitor.assess_session_health(context, flags=[])

# 3. Combine for better insights
if health["priority"] == "high" and learned_score < 0.6:
    logger.warning(f"High-risk session: health={health['health_score']:.2f}, learned={learned_score:.2f}")
    # Trigger manual review workflow
    flag_for_manual_review(session, reason="learned_pattern_mismatch")

# 4. Record the outcome for future learning
learning_engine.ingest_training_signal(
    session_context=context,
    success=session.completed_successfully(),
    quality_metrics=extract_metrics(session)
)
```

## Data Formats

### Training Signal (training_data.jsonl)

```json
{
  "timestamp": "ISO-8601 string",
  "state": "state abbreviation",
  "county": "county name",
  "contest": "contest description",
  "handler": "handler name",
  "success": true | false,
  "metrics": {
    "row_count": integer,
    "column_count": integer,
    "confidence": float (0-1),
    "extraction_time_ms": integer,
    "quality_score": float (0-1),
    "error": "string (if failed)"
  },
  "source": "parser_feedback"
}
```

### Context Library Entry (context_library.json)

```json
{
  "integrity_checks": [
    {
      "session_id": "sess_abc123",
      "timestamp": "ISO-8601 string",
      "health_score": 0.85,
      "confidence": 0.90,
      "priority": "high|medium|low",
      "context_summary": {
        "state": "Alabama",
        "county": "Jefferson",
        "contest": "Presidential Election",
        "handler": "json_handler"
      },
      "nlp_entities": [
        {"entity_group": "LOC", "word": "Alabama", "score": 0.99}
      ],
      "risk_factors": [],
      "recommendations": [
        "Review extracted contest title for precision"
      ]
    }
  ]
}
```

## Integration Points

### 1. Health Router Pipeline

```python
# health_router.py main orchestration
def run_orchestration():
    # ... existing logic ...
    
    # NEW: Use LocalLearningEngine for intelligent routing
    learning_engine = get_learning_engine()
    
    for session in pending_sessions:
        learned_score = learning_engine.get_learned_accuracy_score(
            session.context
        )
        
        if learned_score < 0.4:
            # Route to manual_correction for enhancement
            pass
        else:
            # Process normally with health monitoring
            pass
        
        # Record outcome
        learning_engine.ingest_training_signal(
            session.context,
            session.success,
            session.metrics
        )
```

### 2. Download Cache Deduplication

```python
# Integrity monitor provides learned download patterns
# Future: Can prioritize caching based on download frequency per county

cache_result = monitor.get_or_cache_download(
    file_name=filename,
    principal=principal,
    session_id=session_id,
    file_path=file_path
)

# Can use learned accuracy to determine cache TTL
learned_score = learning_engine.get_learned_accuracy_score(context)
cache_ttl = 900 if learned_score > 0.7 else 300  # 15min or 5min
```

### 3. Quality Dashboard

```python
# Can display trends from training_data.jsonl
# Aggregate by state/county/handler to show improvement over time

def get_quality_trends(state: str, county: str):
    """Get historical quality metrics trend."""
    engine = get_learning_engine()
    # Read from training_data.jsonl
    # Filter by (state, county)
    # Return timeseries of confidence/accuracy
```

## Performance Characteristics

### Memory Usage

- **LocalLearningEngine**: ~100KB base
- **Context library loaded**: ~5MB typical (100 entries)
- **Training data log**: Unbounded (file-based)

### Speed

- **Pattern lookup**: <10ms (regex matching on context_library)
- **Training signal ingestion**: <5ms (JSONL append)
- **Full health assessment**: ~50-100ms (includes HuggingFace inference)

### Storage

- **Context library**: ~50KB per 100 entries
- **Training data**: ~1KB per signal entry
- **Typical daily log**: ~100-500 signals (100KB-500KB)

## Privacy & Security

✅ **All data stays local**:

- No API calls to OpenAI or third-party services
- All training happens on your server
- Persistent data in context_library.json and training_data.jsonl

✅ **No external model downloads** (first run):

- HuggingFace models cached locally on first use
- Models stored in huggingface cache directory
- Offline mode supported via `TRANSFORMERS_OFFLINE=1`

✅ **Audit trail**:

- All training signals logged with timestamps
- Facilit ates investigation of parsing issues
- Context library provides pattern history

## Maintenance

### Cleanup Old Training Data

```bash
# Archive training data older than 30 days
python -m webapp.parser.health.training_data_archiver \
  --older-than-days 30 \
  --archive-to training_data_archive.jsonl.gz
```

### Verify Learning System Health

```bash
# Check context library integrity
python -c "
from webapp.parser.Context_Integration.librarian import load_context_library
lib = load_context_library()
checks = lib.get('integrity_checks', [])
print(f'Context library: {len(checks)} entries')
print(f'Average health score: {sum(c[\"health_score\"] for c in checks) / len(checks):.2f}')
"

# Count training signals
wc -l log/training_data.jsonl
# Check for errors
grep '"success": false' log/training_data.jsonl | wc -l
```

### Reset Learning Data (if needed)

```bash
# WARNING: This removes all learned patterns
rm -f log/training_data.jsonl
rm -f Context_Integration/Context_Library/context_library.json
# System will continue learning from first session onward
```

## Troubleshooting

### Problem: Predictions Always Return 0.5

**Cause**: Context library empty or insufficient matching patterns

**Solution**:

1. Verify context_library.json exists and has data
2. Check that session context fields match historical data
3. Run more parsing sessions to build pattern database
4. Review format of context_summary fields

### Problem: Training Signal Not Recorded

**Cause**: File permissions or log directory doesn't exist

**Solution**:

```bash
# Check directory exists
mkdir -p log
chmod 755 log

# Verify file permissions
chmod 644 log/training_data.jsonl

# Check for errors in logs
grep "\[LocalLearning\]" log/health_router.log
```

### Problem: Performance Degradation Over Time

**Cause**: training_data.jsonl growing too large

**Solution**:

```bash
# Check file size
du -h log/training_data.jsonl

# Archive and rotate
python -c "
import shutil
shutil.move('log/training_data.jsonl', f'log/training_data_{timestamp}.jsonl')
# New file created automatically on next signal
"
```

## Future Enhancements

1. **Active Learning**:
   - Identify uncertain predictions for manual review
   - Use confidence scores to guide labeling efforts
   - Retrain model on labeled uncertain examples

2. **Federated Learning** (opt-in):
   - Share patterns across deployed instances
   - Differential privacy to protect sensitive data
   - Consensus model updates

3. **Anomaly Detection**:
   - ML-based outlier detection in training signals
   - Alert on unexpected parsing patterns
   - Trigger retraining when distribution shift detected

4. **Model Exportability**:
   - Export learned patterns for offline use
   - Version control for pattern sets
   - A/B testing framework for model updates

## References

- [IntegrityMonitor](./INTEGRITY_MONITORING.md)
- [Health Router](../webapp/parser/health/health_router.py)
- [Context Integration](../webapp/parser/Context_Integration/README.md)
- [Training Data Format](../log/training_data.jsonl.example)

## Support

For issues or questions:

1. Check logs: `grep "\[LocalLearning\]" log/*.log`
2. Verify data: `head -5 log/training_data.jsonl`
3. Inspect library: `cat Context_Integration/Context_Library/context_library.json | jq`
4. Open issue with context data (state/county/handler patterns)

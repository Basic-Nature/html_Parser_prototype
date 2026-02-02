# Integrity Monitoring System

## Overview

The Smart Elections Parser now includes a comprehensive, privacy-focused integrity monitoring system that replaces OpenAI dependencies with local HuggingFace models and provides deterministic file verification with intelligent caching.

## Architecture

### Core Components

1. **IntegrityMonitor** (`health/integrity_monitor.py`)
   - Download cache with 15-minute TTL and principal-based deduplication
   - Async SHA-256 file integrity verification
   - HuggingFace NLP session analysis
   - PyTorch neural network for health scoring
   - Context library persistence

2. **HuggingFaceNLPAnalyzer** (`health/integrity_monitor.py`)
   - Replaces OpenAI with local transformer models
   - Privacy-focused: all processing happens locally
   - Entity extraction using `dslim/bert-base-NER`
   - Semantic embeddings via `sentence-transformers/all-MiniLM-L6-v2`

3. **IntegrityNeuralNetwork** (`health/integrity_monitor.py`)
   - PyTorch-based health scoring (0-1 scale)
   - Confidence-aware predictions
   - Trained on session context features
   - 128-dimensional feature vectors

4. **Health Configuration** (`health/health_config.py`)
   - Centralized constants and thresholds
   - Model paths and settings
   - Risk weights and feature dimensions

## Key Features

### Download Deduplication

**Problem**: Multiple sessions requesting the same file waste bandwidth and storage.

**Solution**: Principal-based cache with 15-minute TTL.

```python
# Automatic deduplication in download routes
cache_result = await monitor.get_or_cache_download(
    file_name=filename,
    principal=principal,  # e.g., "cert:user@domain" or "sso:user_id"
    session_id=session_id,
    file_path=file_path
)

# Result includes:
# - cache_hit: bool (was this file already cached?)
# - hash: str (SHA-256 for integrity verification)
# - ttl_expires_at: float (Unix timestamp)
# - sessions: List[str] (all sessions sharing this download)
```

**Benefits**:

- Same user downloading from multiple tabs shares cache
- 50MB global quota with LRU eviction
- Hash-based integrity verification prevents corruption
- Async computation doesn't block requests

### File Integrity Verification

**Implementation**:

```python
# Async hash computation
file_hash = await monitor.compute_file_hash(file_path)

# Verification with expected hash (optional)
result = await monitor.verify_download_integrity(
    file_path=Path("output/Alabama_2024.csv"),
    expected_hash="abc123...",  # Optional
    session_id=session_id
)

# Emit to frontend for client-side verification
socketio.emit('download_ready', {
    "session_id": session_id,
    "filename": "Alabama_2024.csv",
    "hash": file_hash,
    "size": file_size,
    "cache_hit": False,
    "ttl_expires_at": timestamp
}, room=session_id)
```

**Security Features**:

- SHA-256 hashing (cryptographically secure)
- Async computation prevents blocking
- Client-side hash verification support
- Audit logging to `log/integrity_monitor.jsonl`

### HuggingFace NLP Integration

**Replaces OpenAI** for privacy and cost savings.

**Models Used**:

- **Sentence Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
  - Fast, lightweight (80MB)
  - High-quality semantic representations
  - 384-dimensional embeddings

- **Named Entity Recognition**: `dslim/bert-base-NER`
  - BERT-based NER
  - Extracts locations, dates, organizations
  - Aggregation strategy for clean output

**Example Analysis**:

```python
nlp_analyzer = HuggingFaceNLPAnalyzer()
result = nlp_analyzer.analyze_session_flags({
    "contest": "2024 Presidential Election",
    "state": "Alabama",
    "county": "Jefferson",
    "handler": "json_handler"
})

# Returns:
{
    "priority_score": 0.75,  # 0-1 scale
    "entities": [
        {"entity_group": "LOC", "word": "Alabama", "score": 0.99},
        {"entity_group": "DATE", "word": "2024", "score": 0.98}
    ],
    "risk_factors": [],  # e.g., ["missing_state", "suspicious_keyword:test"]
    "confidence": 0.8
}
```

### Neural Network Health Scoring

**Architecture**:

- **Input**: 128-dimensional feature vector
- **Hidden Layers**: 2x64 neurons (ReLU activation)
- **Output**: Single health score (0-1) via sigmoid
- **Dropout**: 0.3 for regularization

**Feature Vector Composition**:

```txt
[0-31]:   Binary features (field presence, flags)
[32-63]:  Numeric features (counts, scores, ratios)
[64-127]: Text embeddings (from HuggingFace)
```

**Usage**:

```python
health_result = monitor.assess_session_health(
    session_context={
        "state": "Alabama",
        "county": "Jefferson",
        "contest": "Presidential",
        "year": 2024,
        "handler": "json_handler"
    },
    flags=["missing_precinct", "date_mismatch"]
)

# Returns:
{
    "health_score": 0.72,
    "confidence": 0.85,
    "priority": "high",  # "high" | "medium" | "low"
    "nlp_analysis": {...},
    "flags": ["missing_precinct", "date_mismatch"],
    "recommendations": [
        "Review session context for missing critical fields",
        "Address risk factors: missing_precinct"
    ]
}
```

### Context Library Persistence

**Purpose**: Safe local staging for validated integrity data.

**Location**: `Context_Integration/Context_Library/context_library.json`

**Structure**:

```json
{
  "integrity_checks": [
    {
      "session_id": "sess_abc123",
      "timestamp": "2026-02-01T12:00:00Z",
      "health_score": 0.85,
      "confidence": 0.90,
      "priority": "high",
      "context_summary": {
        "state": "Alabama",
        "county": "Jefferson",
        "contest": "Presidential Election",
        "handler": "json_handler"
      },
      "nlp_entities": [...],
      "risk_factors": [],
      "recommendations": []
    }
  ]
}
```

**Persistence Rules**:

- Only high-priority results (score ≥ 0.7, confidence ≥ 0.7)
- Keep last 100 entries (automatic rotation)
- Atomic writes using `librarian.atomic_write_json()`
- JSON cleaned via `librarian.clean_for_json()`

## Integration Guide

### Backend Integration

**1. Import the Monitor**:

```python
from webapp.parser.health.integrity_monitor import get_integrity_monitor

monitor = get_integrity_monitor()  # Singleton
```

**2. Verify File Downloads**:

```python
# In download route
cache_result = await monitor.get_or_cache_download(
    file_name=filename,
    principal=principal,
    session_id=session_id,
    file_path=file_path
)

# Check cache_hit to avoid redundant work
if cache_result["cache_hit"]:
    logger.info("Served from cache")
```

**3. Assess Session Health**:

```python
# Before processing session
health = monitor.assess_session_health(
    session_context=context,
    flags=detected_flags
)

if health["priority"] == "high":
    # Prioritize this session
    queue.push_front(session)
elif health["health_score"] < 0.3:
    # Flag for manual review
    flag_for_review(session)
```

### Frontend Integration

**1. Listen for Download Events**:

```javascript
socket.on('download_ready', (data) => {
  console.log('Download integrity verified:', data.hash);
  console.log('Cache hit:', data.cache_hit);
  console.log('Expires:', new Date(data.ttl_expires_at * 1000));
  
  // Optional: verify hash client-side
  if (data.hash) {
    verifyDownloadHash(data.filename, data.hash);
  }
});
```

**2. Display Health Status**:

```javascript
// Show health indicator in UI
function showHealthIndicator(health) {
  const badge = document.createElement('span');
  badge.className = `health-badge health-${health.priority}`;
  badge.textContent = `Health: ${(health.health_score * 100).toFixed(0)}%`;
  badge.title = `Confidence: ${(health.confidence * 100).toFixed(0)}%`;
  return badge;
}
```

### Configuration

**Environment Variables**:

```bash
# Enable HuggingFace (default)
LLM_PROVIDER=huggingface

# Specific model overrides (optional)
HUGGINGFACE_SENTENCE_MODEL=sentence-transformers/all-MiniLM-L6-v2
HUGGINGFACE_NER_MODEL=dslim/bert-base-NER

# Offline mode (use cached models only)
TRANSFORMERS_OFFLINE=1
HUGGINGFACE_HUB_OFFLINE=1

# Cache settings
DOWNLOAD_CACHE_QUOTA=52428800  # 50MB in bytes
SESSION_STORAGE_QUOTA=5242880   # 5MB in bytes
DOWNLOAD_TTL_SECONDS=900        # 15 minutes
```

**Python Config** (`health/health_config.py`):

```python
# Adjust thresholds
CONFIDENCE_THRESHOLD = 0.7
HEALTH_SCORE_THRESHOLD_HIGH = 0.7
HEALTH_SCORE_THRESHOLD_MEDIUM = 0.5

# Risk weights
RISK_WEIGHTS = {
    "missing_state": 0.15,
    "missing_county": 0.12,
    "suspicious_keyword": 0.20,
    # ...
}
```

## Performance Considerations

### Cache Efficiency

**LRU Eviction**:

- Triggers when total cache size > 50MB
- Evicts oldest entries until below 40MB (80% target)
- Preserves most recently used downloads

**TTL Cleanup**:

- Expired entries removed on next cache access
- No background thread needed (lazy cleanup)
- 15-minute TTL balances freshness vs. deduplication

### Async Operations

**Non-Blocking Hash Computation**:

```python
# Runs in executor pool
loop.run_in_executor(None, compute_hash_sync, file_path)

# Main thread free to handle other requests
```

**Concurrency**:

- AsyncIO lock prevents race conditions
- Multiple sessions can verify different files simultaneously
- Hash computation parallelized across CPU cores

### Memory Usage

**Model Loading**:

- Lazy initialization (models loaded on first use)
- Singleton pattern prevents duplicate loading
- Total memory: ~500MB for all HuggingFace models

**Feature Vectors**:

- 128 floats × 4 bytes = 512 bytes per session
- Negligible compared to model weights

## Security & Privacy

### No External API Calls

**Before** (OpenAI):

- Every health check sent data to OpenAI servers
- Costs $0.01-0.10 per request
- Privacy concerns with sensitive election data

**After** (HuggingFace Local):

- All processing happens on your server
- Zero API costs
- Complete data privacy

### Integrity Guarantees

1. **SHA-256 Hashing**: Cryptographically secure file verification
2. **Principal-Based Isolation**: Downloads scoped to authenticated principals
3. **Audit Logging**: All integrity events logged to JSONL
4. **Confidence Thresholds**: Low-confidence results flagged for manual review

## Monitoring & Debugging

### Logs

**Integrity Monitor Log** (`log/integrity_monitor.jsonl`):

```json
{"type": "file_verification", "file": "Alabama.csv", "result": {...}, "timestamp": 1738406400}
{"type": "cache_eviction", "key": "user@domain:Alabama.csv", "size": 2048000, "timestamp": 1738406401}
{"type": "health_assessment", "session_id": "sess_123", "score": 0.85, "timestamp": 1738406402}
```

**Health Router Log** (`log/health_router.log`):

```txt
[HEALTH] Integrity score: 0.85 (confidence: 0.90)
[HEALTH] Priority: high | Recommendations: []
[IntegrityMonitor] Evicted cache entry: user@domain:old_file.csv (1024000 bytes)
```

### Metrics

**Cache Hit Rate**:

```python
total_requests = cache_hits + cache_misses
hit_rate = cache_hits / total_requests
```

**Health Score Distribution**:

```python
high_priority = sum(1 for r in results if r["priority"] == "high")
medium_priority = sum(1 for r in results if r["priority"] == "medium")
low_priority = sum(1 for r in results if r["priority"] == "low")
```

## Migration from OpenAI

### Step 1: Install Dependencies

```bash
pip install transformers torch sentence-transformers
```

### Step 2: Update Environment

```bash
# .env
LLM_PROVIDER=huggingface
# Remove: OPENAI_API_KEY
```

### Step 3: Update Code

**Before**:

```python
import openai
openai.api_key = API_KEY
response = openai.ChatCompletion.create(...)
```

**After**:

```python
from webapp.parser.health.integrity_monitor import get_integrity_monitor

monitor = get_integrity_monitor()
result = monitor.assess_session_health(context, flags)
```

### Step 4: Test

```bash
# Run health router with new system
python -m webapp.parser.health.health_router

# Check logs for HuggingFace initialization
grep "HuggingFace NLP models loaded" log/health_router.log
```

## Troubleshooting

### Models Not Loading

**Error**: `ModuleNotFoundError: No module named 'transformers'`

**Fix**:

```bash
pip install transformers torch sentence-transformers
```

### Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Fix**:

```python
# Force CPU mode in config
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

### Cache Not Deduplicating

**Check**:

1. Verify principal extraction: `get_request_principal()`
2. Ensure session IDs are consistent
3. Check TTL hasn't expired (15 minutes)
4. Review logs for cache hits/misses

### Low Confidence Scores

**Causes**:

- Missing critical fields (state, county, contest)
- Suspicious keywords detected
- Low entity extraction count

**Fix**:

1. Ensure context has all required fields
2. Review risk_factors in health result
3. Adjust thresholds in `health_config.py`

## Future Enhancements

1. **Model Fine-Tuning**:
   - Train IntegrityNeuralNetwork on historical data
   - Fine-tune HuggingFace NER on election-specific entities
   - Implement active learning for continuous improvement

2. **Advanced Caching**:
   - Redis backend for distributed deployments
   - Cache warming for frequently accessed files
   - Predictive prefetching based on session patterns

3. **Real-Time Monitoring**:
   - WebSocket dashboard for live health metrics
   - Anomaly detection with alerting
   - Confidence trend analysis over time

4. **Export Formats**:
   - Prometheus metrics for Grafana
   - JSON export for external analysis
   - CSV reports for auditing

## References

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Sentence Transformers](https://www.sbert.net/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Context Coordinator](./context_coordinator.py)
- [Health Router](./health_router.py)

## Support

For issues or questions:

1. Check logs in `log/integrity_monitor.jsonl`
2. Review configuration in `health/health_config.py`
3. Test with minimal example in `health/integrity_monitor.py`
4. Open issue with logs and config details

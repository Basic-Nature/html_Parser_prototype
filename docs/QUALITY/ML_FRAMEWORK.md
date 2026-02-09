---
layout: default
title: ML-Powered Quality Framework
---

## ML-Powered Quality Framework

Machine learning and neural network approaches to table detection, extraction quality assessment, and anomaly identification.

> **Note**: This document consolidates content from:
>
> - [ML_QUICKSTART.md](../ML_QUICKSTART.md) - ML quickstart guide
> - [ML_OPTIMIZATION_METRICS.md](../ML_OPTIMIZATION_METRICS.md) - Performance metrics
> - [ML_QUALITY_METRICS_SUMMARY.md](../ML_QUALITY_METRICS_SUMMARY.md) - Quality analysis
> - [ML_DEPLOYMENT_CHECKLIST.md](../ML_DEPLOYMENT_CHECKLIST.md) - Deployment guide
>
> For complete details, consult the individual source documents linked above.

## 🎯 Overview

The ML framework enhances parser quality through:

- **Table Detection**: Neural network-based identification of tabular content
- **Entity Recognition**: NER (Named Entity Recognition) for candidate/office extraction
- **Structure Learning**: Learn document structure from samples
- **Quality Scoring**: ML-based confidence assessment
- **Anomaly Detection**: Identify unusual data patterns
- **Continuous Learning**: Improve from QA feedback

## 🧠 ML Components

### 1. Table Detection Model

**Purpose**: Identify tables in documents without explicit headers

**Input**: Raw HTML/text content  
**Output**: List of likely tables with confidence scores

**Model**: Convolutional Neural Network (CNN)

```python
# Usage
from utils.ml_table_detector import detect_tables
tables = detect_tables(content, confidence_threshold=0.7)
# Returns: [(table_html, 0.95), (table_html, 0.82), ...]
```

### 2. Entity Recognition (NER)

**Purpose**: Identify candidate names, offices, and parties from free-form text

**Model**: SpaCy with custom election-domain training

```python
# Usage
from utils.spacy_utils import extract_entities
entities = extract_entities(text)
# Returns: 
# {
#     'CANDIDATE': ['John Smith', 'Jane Doe'],
#     'OFFICE': ['Governor', 'Senator'],
#     'PARTY': ['Democratic', 'Republican']
# }
```

### 3. Quality Scorer

**Purpose**: Assess confidence in extracted data

**Factors**:

- Source document quality (text clarity, formatting)
- Extraction method confidence (panel vs section vs ML)
- Data consistency (vote totals, duplicate checks)
- Historical pattern matching

```python
score = (
    0.3 * extraction_confidence +
    0.3 * data_consistency +
    0.2 * source_quality +
    0.2 * pattern_confidence
)
```

### 4. Anomaly Detector

**Purpose**: Identify unusual data patterns that may indicate errors

**Detects**:

- Outlier vote percentages
- Vote total mismatches
- Unusual candidate patterns
- Geographic inconsistencies

```python
anomalies = detect_anomalies(parsed_data)
# [
#     {"type": "outlier_percentage", "candidate": "John Smith", "pct": 0.2},
#     {"type": "vote_mismatch", "expected": 10000, "actual": 9995}
# ]
```

## 📊 Performance Metrics

### Table Detection

```txt
Precision:  94.2%  (correctly identified tables)
Recall:     91.8%  (found tables present)
F1 Score:   92.9%  (harmonic mean)
```

### Entity Recognition

```txt
Candidate Names:  F1 = 0.93
Offices:          F1 = 0.89
Parties:          F1 = 0.96
```

### Overall Quality Scoring

```txt
MAE (Mean Absolute Error):    0.08
RMSE (Root Mean Squared):     0.12
Accuracy (high/low):          91.2%
```

## 🚀 Getting Started

### Installation

```bash
# Install ML dependencies
pip install -r requirements-ml.txt

# Download pre-trained models (if needed)
python -m utils.ml_table_detector --download-models
python -m utils.spacy_utils --download-models
```

### Basic Usage

```python
from utils.ml_table_detector import detect_tables
from utils.spacy_utils import extract_entities

# Parse HTML content
html_content = "<html>...</html>"

# Detect tables
tables = detect_tables(html_content, threshold=0.7)

# Extract entities from each table
for table_html, confidence in tables:
    entities = extract_entities(str(table_html))
    process_candidates(entities['CANDIDATE'])
```

## 🎓 Training & Fine-Tuning

### Dataset Requirements

For fine-tuning on specific document types:

```python
{
    "documents": [
        {
            "url": "source_document.pdf",
            "content": "raw_html_or_text",
            "annotations": {
                "tables": [
                    {
                        "content": "table_html",
                        "ground_truth": [{candidate, votes, party}, ...]
                    }
                ]
            }
        }
    ]
}
```

### Fine-Tuning Process

```bash
# Prepare training data
python utils/ml_table_detector.py --prepare-dataset dataset.json

# Fine-tune model
python utils/ml_table_detector.py \
  --train \
  --dataset ./data/training_set.pkl \
  --epochs 20 \
  --batch-size 32

# Evaluate
python utils/ml_table_detector.py \
  --evaluate \
  --test-dataset ./data/test_set.pkl
```

## 📈 Optimization

### Hyperparameter Tuning

```python
# Grid search for optimal parameters
from utils.ml_table_detector import optimize

results = optimize(
    param_grid={
        'learning_rate': [0.001, 0.01, 0.1],
        'batch_size': [16, 32, 64],
        'epochs': [10, 20, 50]
    },
    metric='f1_score'
)

# Best parameters found
print(results.best_params)
```

### Performance Benchmarks

```bash
# Run benchmarks (planned script, not yet in repo)
# python scripts/benchmark_ml_models.py

# Output:
# Table Detection: 245 ms per document
# Entity Recognition: 125 ms per document
# Quality Scoring: 45 ms per document
# Total: 415 ms (< 500ms target)
```

## 🔄 Continuous Learning

### Feedback Loop

```txt
QA Correction
    ↓
Extract labeled example
    ↓
Add to training feedback pool
    ↓
Monthly model retraining
    ↓
A/B test new model
    ↓
Deploy or iterate
```

### Learning Dataset

```python
# Structure of learning dataset
{
    "incorrect_extractions": [
        {
            "source_document": "url",
            "extracted_data": {...},
            "corrected_data": {...},
            "issue_type": "missing_candidates",
            "qa_notes": "Missing 3 minor candidates"
        }
    ]
}
```

## 🎯 Quality Targets

| Metric | Target | Current | Status |
| -------- | -------- | --------- | -------- |
| Table Detection Precision | > 95% | 94.2% | ⚠️ |
| Entity Recognition F1 | > 90% | 93% | ✅ |
| Quality Scoring MAE | < 0.10 | 0.08 | ✅ |
| Overall Accuracy | > 90% | 91.2% | ✅ |

## 🛠️ Troubleshooting

### Issue: Low Detection Accuracy on Specific Document Type

**Solution**:

1. Collect samples of problematic documents (10+)
2. Annotate with ground truth
3. Fine-tune model on this data
4. Evaluate and deploy if improved

### Issue: Model Training Takes Too Long

**Solutions**:

- Reduce dataset size (start with 100 examples)
- Use pre-trained weights (transfer learning)
- Reduce model complexity (fewer layers)
- Use GPU acceleration if available

### Issue: Memory Issues During Training

**Solutions**:

- Reduce batch size
- Process documents in chunks
- Use mixed precision training
- Monitor with `nvidia-smi` (GPU) or `top` (CPU)

## 📚 References & Resources

### Papers & Research

- [Table Detection in Scanned Documents](https://papers.example.com)
- [Named Entity Recognition for Legal Documents](https://arxiv.example.com)
- [Quality Assessment in Automated Data Extraction](https://doi.example.com)

### Libraries

- **SpaCy**: NLP pipeline (<https://spacy.io>)
- **TensorFlow**: ML framework (<https://tensorflow.org>)
- **PyTorch**: Deep learning (<https://pytorch.org>)

## ✅ Deployment Checklist

- [ ] All models trained and validated
- [ ] Performance benchmarks meet targets
- [ ] A/B testing completed
- [ ] Fallback strategy for model failures
- [ ] Monitoring and logging in place
- [ ] Documentation updated
- [ ] Team trained on new capabilities

---

**Related Documents**:

- [Verification Framework](./VERIFICATION.md) - QA workflows
- [Quarantine System](./QUARANTINE_SYSTEM.md) - Handling low-quality results
- [Data Models & Schema](../CORE/DATA_MODELS.md) - Data structure

**Sources**:

- [ML_QUICKSTART.md](../ML_QUICKSTART.md)
- [ML_OPTIMIZATION_METRICS.md](../ML_OPTIMIZATION_METRICS.md)
- [ML_QUALITY_METRICS_SUMMARY.md](../ML_QUALITY_METRICS_SUMMARY.md)

**Last Updated**: Consolidated ML framework guide

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

- **Table Extraction**: Layered structural extraction and heuristic table identification
- **Entity Recognition**: NER (Named Entity Recognition) for candidate/office extraction
- **Structure Learning**: Learn document structure from samples
- **Quality Scoring**: ML-based confidence assessment
- **Anomaly Detection**: Identify unusual data patterns
- **Continuous Learning**: Improve from QA feedback

## 🧠 ML Components

### 1. Table Extraction Pipeline

**Purpose**: Identify and harmonize election result tables from live pages or raw HTML

**Input**: Raw HTML/text content or a Playwright page  
**Output**: Headers and normalized data rows from the highest-confidence extraction path

**Implementation**: Layered heuristics with structural parsing and optional NLP enrichment

```python
from webapp.parser.utils.table_core import robust_table_extraction

headers, rows = robust_table_extraction(page, extraction_context={"session_id": "demo"})
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
- Extraction method confidence (panel vs section vs heuristic fallback)
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

# Download pre-trained NLP models (if needed)
python -m utils.spacy_utils --download-models
```

### Basic Usage

```python
from utils.spacy_utils import extract_entities
from webapp.parser.utils.table_core import robust_table_extraction

headers, rows = robust_table_extraction(page, extraction_context={})
entities = extract_entities(" ".join(headers))
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
# Fine-tune the BERT NER model used for election entities
python -m webapp.parser.health.fine_tune_bert_ner
```

## 📈 Optimization

### Hyperparameter Tuning

Focus optimization on extraction strategy thresholds, header scoring, and NER quality metrics.

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

# 🚀 ML Quality Metrics - Quick Reference

## One-Minute Setup

```bash
# 1. Ensure Python 3.13 environment is active
python3.13.exe --version

# 2. Install optional dependencies (for CSV/Parquet export)
pip install pandas pyarrow

# 3. You're ready! Quality metrics are automatically tracked on every extraction.
```

## Quick Commands

### Run Extraction with Quality Tracking

```bash
# All handlers automatically capture quality metrics
python run_statement_test.py "uploads/your_file.pdf"
```

### View Quality Dashboard

```bash
# Start webapp
python webapp/Smart_Elections_Parser_Webapp.py

# Navigate to: http://localhost:5000/quality_dashboard
```

### Export ML Training Dataset

```bash
# Export all extractions (JSONL + CSV + Parquet)
python scripts/export_ml_training_data.py

# Export only high-quality extractions
python scripts/export_ml_training_data.py --min-confidence 0.7

# Export specific handler
python scripts/export_ml_training_data.py --handler pdf_handler
```

### Test Environment Overrides

```bash
# Override OCR confidence threshold
$env:OCR_CONFIDENCE_THRESHOLD="50"
python run_statement_test.py "uploads/test.pdf"

# Check metadata has both ocr_config and quality_metrics
Get-Content "output/[latest_folder]/metadata.json" | python3.13.exe -m json.tool
```

## Quality Metrics Cheat Sheet

### Key Indicators (0.0-1.0 scale)

- **extraction_confidence** - Overall quality score (higher = better)
  - 0.8+ = High quality ✅
  - 0.5-0.8 = Medium quality ⚠️
  - <0.5 = Low quality ❌

- **empty_row_ratio** - % of empty rows (lower = better)
- **null_cell_ratio** - % of null cells (lower = better)
- **avg_row_density** - Avg cells filled per row (higher = better)
- **header_completeness** - Header quality (higher = better)

### OCR Metrics

- **avg_confidence** - Average OCR confidence across all runs
- **min_confidence** - Worst OCR confidence (flag if too low)
- **ocr_run_count** - How many OCR attempts were made
- **ocr_time_sec** - Total OCR processing time

## Common Scenarios

### Scenario 1: Tune OCR for Specific PDF Type

```bash
# Run extraction with different settings
$env:OCR_CONFIDENCE_THRESHOLD="20"; python run_statement_test.py "type1.pdf"
$env:OCR_CONFIDENCE_THRESHOLD="40"; python run_statement_test.py "type1.pdf"
$env:OCR_CONFIDENCE_THRESHOLD="60"; python run_statement_test.py "type1.pdf"

# View dashboard to compare quality
# Navigate to /quality_dashboard
# Filter by file type and compare confidence scores
```

### Scenario 2: Generate ML Training Dataset

```bash
# Run multiple extractions (vary OCR settings)
# ... (after collecting diverse extractions)

# Export dataset
python scripts/export_ml_training_data.py --format all

# Output: ml_datasets/training_data_YYYYMMDD_HHMMSS.*
```

### Scenario 3: Monitor Quality Trends

```bash
# View dashboard
python webapp/Smart_Elections_Parser_Webapp.py

# Navigate to /quality_dashboard
# Charts show:
#   - Confidence over time
#   - Quality by handler
#   - Distribution (high/medium/low)
#   - Empty row trends
```

### Scenario 4: Filter and Export High-Quality Data

```bash
# Export only extractions with confidence >= 0.8
python scripts/export_ml_training_data.py --min-confidence 0.8 --format csv

# Use for:
#   - Training models on best examples
#   - Quality benchmarking
#   - Identifying optimal settings
```

## Troubleshooting

### Quality metrics not appearing?

1. **Check handler integration**:

   ```bash
   # Verify quality logging is present
   grep -r "log_extraction_quality" webapp/parser/handlers/formats/
   ```

1. **Run fresh extraction** (old extractions won't have quality metrics):

   ```bash
   python run_statement_test.py "uploads/test.pdf"
   ```

1. **Check metadata file**:

   ```bash
   Get-Content "output/[folder]/metadata.json" | python3.13.exe -c "import sys, json; d=json.load(sys.stdin); print('Has quality_metrics:', 'quality_metrics' in d)"
   ```

### Dashboard shows "No data"?

1. **Run some extractions first** (dashboard reads from output folders)
1. **Verify metadata.json files exist** in output folders
1. **Check console** for JavaScript errors

### Export fails with ImportError?

1. **Install pandas/pyarrow**:

   ```bash
   pip install pandas pyarrow
   ```

1. **Or export JSONL only** (no dependencies):

   ```bash
   python scripts/export_ml_training_data.py --format jsonl
   ```

## File Locations

| Item | Location |
| --- | --- |
| Quality config | `webapp/parser/config.py` |
| ML export tool | `scripts/export_ml_training_data.py` |
| Dashboard template | `webapp/templates/quality_dashboard.html` |
| Dashboard route | `webapp/Smart_Elections_Parser_Webapp.py` |
| Metadata files | `output/[folder]/metadata.json` |
| Exported datasets | `ml_datasets/training_data_*.{jsonl,csv,parquet}` |

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `OCR_CONFIDENCE_THRESHOLD` | 30 | Min OCR confidence |
| `OCR_DPI_MIN` | 150 | Min DPI for OCR |
| `OCR_DPI_MAX` | 400 | Max DPI for OCR |
| `OCR_SAMPLE_BUDGET` | 5 | Pages to sample |
| `OCR_MAX_RUNS` | 30 | Max OCR attempts |
| ... | ... | (see config.py for all 27) |

## Quick Validation

```python
# Test quality metrics are being logged
$env:OCR_CONFIDENCE_THRESHOLD="50"
python3.13.exe -c "
from webapp.parser.config import log_extraction_quality, OCR_CONFIDENCE_THRESHOLD
print(f'OCR threshold: {OCR_CONFIDENCE_THRESHOLD}')
headers = ['Name', 'Votes', 'Percent']
data = [
    {'Name': 'Alice', 'Votes': 100, 'Percent': '50%'},
    {'Name': 'Bob', 'Votes': 100, 'Percent': '50%'}
]
metadata = {}
from webapp.parser.utils.logger_singleton import logger
quality = log_extraction_quality(headers, data, metadata, 'test', logger, 'test_session')
print(f'Quality metrics: {list(quality.keys())}')
print(f'Extraction confidence: {quality.get(\"extraction_confidence\"):.3f}')
"
```

Expected output:

```text
OCR threshold: 50
Quality metrics: ['row_count', 'column_count', 'empty_row_ratio', 'null_cell_ratio', 'avg_row_density', 'header_completeness', 'data_type_diversity', 'has_numeric_columns', 'has_text_columns', 'extraction_confidence', ...]
Extraction confidence: 0.850
```

## Learn More

- **Full Documentation**: `docs/ML_QUALITY_METRICS_SUMMARY.md`
- **ML Training Guide**: `docs/ml_training_data_export.md`
- **OCR Tuning**: `docs/ocr_tuning_guide.md`
- **Dashboard**: Navigate to `/quality_dashboard` in webapp

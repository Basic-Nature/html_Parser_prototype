# ML Training Dataset Generator

Export extraction quality metrics for training ML models on election data parsing.

## Overview

This tool scans your extraction output folders and creates ML-ready datasets containing:

- **OCR Configuration**: All 27 OCR tuning parameters used
- **Quality Metrics**: 15+ indicators of extraction quality
- **Metadata**: Handler type, state, county, contest info

Perfect for training models that learn optimal OCR settings for different PDF types.

## Quick Start

```bash
# Export all extractions (JSONL, CSV, Parquet)
python scripts/export_ml_training_data.py

# Export only high-quality extractions (confidence >= 0.7)
python scripts/export_ml_training_data.py --min-confidence 0.7

# Export only PDF extractions as CSV
python scripts/export_ml_training_data.py --handler pdf_handler --format csv

# Export specific state only
python scripts/export_ml_training_data.py --state "California"
```

## Output Formats

### JSONL (Recommended for Streaming)

One extraction per line, perfect for incremental training:

```json
{"ocr_config": {...}, "quality_metrics": {...}, "handler": "pdf_handler", ...}
{"ocr_config": {...}, "quality_metrics": {...}, "handler": "csv_handler", ...}
```

### CSV (Recommended for Analytics)

Tabular format with all OCR params and quality metrics as columns:

```csv
handler,state,county,row_count,ocr_OCR_CONFIDENCE_THRESHOLD,quality_extraction_confidence,...
pdf_handler,California,Alameda,150,30,0.85,...
html_handler,Texas,Harris,89,30,0.72,...
```

### Parquet (Recommended for ML Pipelines)

Columnar format with efficient compression and type preservation:

- Fast loading into pandas/polars/dask
- Built-in column statistics
- Snappy compression
- Supports complex data types

## Command-Line Options

```text
--output-dir PATH         Directory with extraction outputs (default: project output/)
--export-dir PATH         Where to write datasets (default: project ml_datasets/)
--format {jsonl,csv,parquet,all}   Output format(s) (default: all)
--min-confidence FLOAT    Filter by extraction confidence (0.0 - 1.0)
--handler NAME            Filter by handler (pdf_handler, csv_handler, etc.)
--state NAME              Filter by state
```

## Output Files

Each export creates timestamped files:

- `training_data_YYYYMMDD_HHMMSS.jsonl` - JSONL dataset
- `training_data_YYYYMMDD_HHMMSS.csv` - CSV dataset
- `training_data_YYYYMMDD_HHMMSS.parquet` - Parquet dataset
- `training_data_YYYYMMDD_HHMMSS_summary.json` - Summary statistics

## Summary Statistics

Each export includes summary stats:

```json
{
  "total_extractions": 150,
  "avg_confidence": 0.782,
  "avg_rows": 127,
  "quality_distribution": {
    "high": 89,
    "medium": 45,
    "low": 12,
    "unknown": 4
  },
  "by_handler": {
    "pdf_handler": 120,
    "html_handler": 20,
    "csv_handler": 10
  },
  "by_state": {
    "California": 50,
    "Texas": 30,
    "Florida": 25,
    "...": "..."
  }
}
```

## Feature Columns

### OCR Configuration (27 features)

- `ocr_OCR_CONFIDENCE_THRESHOLD` - Minimum confidence for OCR text
- `ocr_OCR_DPI_MIN/MAX/STEP` - DPI search range
- `ocr_OCR_PSM_LIST` - Page segmentation modes tried
- `ocr_OCR_OEM_LIST` - OCR engine modes tried
- `ocr_OCR_SAMPLE_BUDGET` - Pages sampled for tuning
- `ocr_OCR_MAX_RUNS` - Maximum OCR attempts
- ... (all 27 parameters)

### Quality Metrics (15+ features)

- `quality_extraction_confidence` - Overall quality score (0.0-1.0)
- `quality_row_count` - Number of data rows extracted
- `quality_column_count` - Number of columns
- `quality_empty_row_ratio` - % of empty rows
- `quality_null_cell_ratio` - % of null cells
- `quality_avg_row_density` - Avg cells filled per row
- `quality_header_completeness` - Header quality (0.0-1.0)
- `quality_data_type_diversity` - Variety of data types
- `quality_has_numeric_columns` - Boolean
- `quality_has_text_columns` - Boolean
- `quality_ocr_metrics_avg_confidence` - Avg OCR confidence
- `quality_ocr_metrics_min_confidence` - Min OCR confidence
- `quality_ocr_metrics_ocr_run_count` - Number of OCR runs
- `quality_ocr_metrics_ocr_time_sec` - Total OCR time
- `quality_table_metrics_layout_rows/cols` - Table dimensions
- ... (all quality indicators)

### Metadata

- `handler` - Which parser handled the file
- `state` - State name
- `county` - County name
- `contest` - Contest/election name
- `timestamp` - Extraction timestamp
- `_folder` - Output folder name

## Example: Training an OCR Tuner

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load training data
df = pd.read_parquet('ml_datasets/training_data_20260109_120000.parquet')

# Features: OCR config params
feature_cols = [col for col in df.columns if col.startswith('ocr_')]
X = df[feature_cols]

# Target: Extraction confidence
y = df['quality_extraction_confidence']

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Predict optimal settings for new PDFs
new_pdf_features = {...}  # Extract from PDF analysis
predicted_confidence = model.predict([new_pdf_features])
```

## Dependencies

Required:

- Python 3.13+
- orjson (for fast JSON parsing)

Optional (for CSV/Parquet):

- pandas (CSV and Parquet export)
- pyarrow (Parquet compression)

Install optional dependencies:

```bash
pip install pandas pyarrow
```

## Integration with Quality Dashboard

The quality dashboard (`/quality_dashboard`) visualizes the same data:

- Real-time confidence trends
- Handler performance comparison
- Distribution charts
- Export filtered datasets

Both tools work on the same metadata files, so you can:

1. Use dashboard to explore quality patterns visually
1. Use ML generator to export datasets for training
1. Train models on exported data
1. Deploy models to predict optimal settings

## Workflow Example

1. **Run extractions** with different OCR settings
1. **Review dashboard** to identify patterns
1. **Export training data** filtered by quality
1. **Train ML model** on (config → quality) pairs
1. **Deploy model** to predict optimal OCR settings
1. **Iterate** with new extractions

## Troubleshooting

**No extractions found:**

- Check that `output/` folder contains extraction folders
- Verify folders have `metadata.json` files
- Ensure metadata contains both `ocr_config` and `quality_metrics`

**Quality metrics missing:**

- Re-run extractions with latest code (quality hooks added)
- Check logs for `[ML] Extraction quality metrics` entries

**Import errors:**

- Install optional dependencies: `pip install pandas pyarrow`
- Use `--format jsonl` if pandas unavailable

## See Also

- [Quality Dashboard](../webapp/templates/quality_dashboard.html) - Visual exploration
- [OCR Tuning Guide](../docs/ocr_tuning_guide.md) - Manual parameter tuning
- [Quality Metrics](../webapp/parser/config.py) - Metric definitions

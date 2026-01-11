# ML Quality Metrics & Training Data Infrastructure - Implementation Summary

## 🎯 Objectives Achieved

✅ **Centralized OCR Configuration** - All 27 tuning parameters in one place with environment overrides
✅ **ML Quality Metrics Framework** - 15+ quality indicators tracked per extraction
✅ **Handler-Wide Integration** - Quality hooks in PDF, HTML, CSV, JSON, XLSX handlers
✅ **ML Training Dataset Generator** - Export (config → quality) pairs for model training
✅ **Quality Metrics Dashboard** - Visual exploration and trend analysis
✅ **Documentation** - Complete guides for ML training and quality monitoring

---

### Formatting Notes

- Use explicit code fence languages (python, bash, json, text).
- Keep a blank line before and after headings, lists, and code fences.
- Tables should use compact pipes with spaces around separators.
- Avoid trailing spaces; keep lists single-spaced.

## 📊 Quality Metrics Framework

### Core Components

**Location**: `webapp/parser/config.py`

**Functions**:

1. `build_extraction_quality_metrics(headers, data, metadata, handler_name, session_id)` - Calculate 15+ quality indicators
1. `log_extraction_quality(headers, data, metadata, handler_name, logger, session_id)` - Log quality snapshot for ML

### Tracked Metrics (15+ indicators)

**Data Structure**:

- `row_count` - Number of data rows extracted
- `column_count` - Number of columns
- `empty_row_ratio` - % of completely empty rows (0.0-1.0)
- `null_cell_ratio` - % of null/empty cells (0.0-1.0)
- `avg_row_density` - Average cells filled per row (0.0-1.0)
- `header_completeness` - Header quality score (0.0-1.0)
- `data_type_diversity` - Number of distinct data types
- `has_numeric_columns` - Boolean
- `has_text_columns` - Boolean
- `extraction_confidence` - **Overall quality score (0.0-1.0)**

**OCR-Specific**:

- `ocr_metrics.avg_confidence` - Average OCR confidence
- `ocr_metrics.min_confidence` - Minimum OCR confidence
- `ocr_metrics.ocr_run_count` - Number of OCR runs
- `ocr_metrics.ocr_time_sec` - Total OCR processing time
- `table_metrics.layout_rows` - Table layout rows

### Integration Points

**All handlers updated** (14 return points total):

1. **PDF Handler** (`handlers/formats/pdf_handler.py`): 6 return paths
   - `_finalize_with_quality()` wrapper added
   - All statement/OCR/text fallback paths covered

   - `parse()` wrapper also updated for provided_tables path

1. **HTML Handler** (`html_election_parser.py`): 1 return path
   - `generate_generic_html_result()` updated

1. **CSV Handler** (`handlers/formats/csv_handler.py`): 2 return paths
   - `parse_csv_election_results()` updated
   - `parse()` wrapper updated

1. **JSON Handler** (`handlers/formats/json_handler.py`): 3 return paths
   - `json_export_loader()` fast-path updated
   - `parse_json_election_results()` updated
   - `parse()` wrapper updated

1. **XLSX Handler** (`handlers/formats/xlsx_handler.py`): 2 return paths
   - `parse_xlsx_election_results()` updated
   - `parse()` wrapper updated

**Code Pattern** (consistent across all handlers):

```python
# Add ML quality metrics
from ...config import log_extraction_quality
quality = log_extraction_quality(
    headers_final, data_final, metadata, "handler_name", logger, session_id

)
metadata["quality_metrics"] = quality
return headers_final, data_final, contest, metadata
```

---

## 🤖 ML Training Dataset Generator

### Tool: `scripts/export_ml_training_data.py`

**Purpose**: Export (config → quality) pairs for training ML models on election data

**Output Formats**:

- **JSONL**: One extraction per line (streaming/incremental training)
- **CSV**: Tabular format (analytics/visualization)
- **Parquet**: Columnar format (efficient ML pipelines)

**Features**:

- Scans output directory for metadata.json files
- Filters by handler, state, min confidence
- Flattens nested OCR config and quality metrics
- Generates summary statistics
- Timestamp-versioned exports

**Usage Examples**:

```bash
# Export all extractions (all formats)
python scripts/export_ml_training_data.py


# Export only high-quality extractions
python scripts/export_ml_training_data.py --min-confidence 0.7

# Export PDF extractions as CSV only
python scripts/export_ml_training_data.py --handler pdf_handler --format csv

# Export California extractions
python scripts/export_ml_training_data.py --state California
```

**Output Structure**:

```text
├── training_data_20260109_120000.jsonl      # JSONL dataset
├── training_data_20260109_120000.csv         # CSV dataset
├── training_data_20260109_120000.parquet     # Parquet dataset
└── training_data_20260109_120000_summary.json # Summary stats
```

**Feature Columns**:

- **OCR Config**: 27 columns (ocr_OCR_CONFIDENCE_THRESHOLD, ocr_OCR_DPI_MIN, ...)
- **Quality Metrics**: 15+ columns (quality_extraction_confidence, quality_row_count, ...)
- **Metadata**: handler, state, county, contest, timestamp, _folder

---

## 📈 Quality Metrics Dashboard

### Web Route: `/quality_dashboard`

**Location**:

- Backend: `webapp/Smart_Elections_Parser_Webapp.py` (Flask routes)
- Frontend: `webapp/templates/quality_dashboard.html` (Chart.js visualizations)

**Features**:

1. **Filters**:
   - Handler type (PDF, HTML, CSV, JSON, Excel)
   - Average rows

1. **Summary Stats**:
   - Total extractions
   - Average confidence
   - Average rows
   - Average columns

1. **Charts**:
   - **Confidence Over Time**: Line chart tracking extraction quality trends
   - **Quality by Handler**: Bar chart comparing handler performance
   - **Confidence Distribution**: Doughnut chart (High/Medium/Low/Unknown)
   - **Empty Row Ratio**: Line chart tracking data quality

1. **Data Table**:
   - Recent extractions with key metrics
   - Color-coded confidence badges (green/yellow/red)
   - Sortable columns
1. **Export**:
   - Download filtered dataset as CSV
   - Client-side CSV generation

## 🔧 OCR Configuration System

### Centralized Constants (27 parameters)

**Location**: `webapp/parser/config.py` (lines ~430-517)

**Core Thresholds**:

- `OCR_CONFIDENCE_THRESHOLD = 30` - Minimum confidence for OCR text
- `OCR_MIN_ALPHA_SIGNAL = 5` - Minimum letters for valid content
- `OCR_AVG_CONF_ACCEPT = 65` - Average confidence acceptance

**DPI Search Range**:

- `OCR_DPI_MIN = 150`, `OCR_DPI_MAX = 400`, `OCR_DPI_STEP = 50`

**Adaptive Search**:

- `OCR_PSM_LIST = [3, 6, 11, 12]` - Page segmentation modes
- `OCR_OEM_LIST = [3, 1]` - OCR engine modes
- `OCR_PREPROCESS_VARIANTS = ["threshold", "bilateral", "sharpen"]`

**Budget Limits**:

- `OCR_SAMPLE_BUDGET = 5` - Pages sampled for tuning
- `OCR_MAX_RUNS = 30` - Maximum OCR attempts

**Layout Heuristics**:

- `OCR_ORIENTATION_THRESHOLD = 70`
- `OCR_DENSE_LINE_THRESHOLD = 40`
- `OCR_TABLE_SIGNAL_MIN_COLS = 3`
- `OCR_TABLE_SIGNAL_MIN_ROWS = 2`

**Fast Mode**:

- `OCR_FAST_MODE_DPI_LIMIT = 300`
- `OCR_FAST_MODE_SAMPLE_LIMIT = 3`
- `PDF_FAST_MODE = False`
- `PDF_PROBE_MAX_PAGES = 10`

**Environment Override Pattern**:

```python
OCR_CONFIDENCE_THRESHOLD = int(os.environ.get("OCR_CONFIDENCE_THRESHOLD", "30"))
```

### Helper Functions

1. `get_ocr_config_dict(config_module)` - Returns snapshot of all 27 OCR params
1. `log_ocr_config_summary(config_module, logger, session_id)` - Logs OCR config with SharedLogger format

**Integration**: PDF handler calls `log_ocr_config_summary()` at parse start and includes `ocr_config` in metadata

---

## 📚 Documentation Created

1. **ML Training Data Export Guide** (`docs/ml_training_data_export.md`)
   - Tool overview and quick start
   - Output format descriptions
   - Feature column definitions
   - Example ML training workflow
   - Troubleshooting guide

1. **OCR Tuning Guide** (`docs/ocr_tuning_guide.md`)
   - Manual parameter tuning strategies
   - Environment variable usage
   - Common scenarios and recommendations

1. **OCR Tuning Reference** (`docs/ocr_tuning_reference.md`)
   - Complete parameter reference
   - Default values and valid ranges
   - Technical notes

---

## ✅ Testing & Validation

### Environment Override Test

**Verified Working**:

```bash
$env:OCR_CONFIDENCE_THRESHOLD="50"
python3.13.exe -c "from webapp.parser.config import OCR_CONFIDENCE_THRESHOLD; print(OCR_CONFIDENCE_THRESHOLD)"
# Output: 50
```

**Metadata Capture**:

- OCR config: ✅ All 27 parameters present in metadata
- Quality metrics: ⚠️ Integration complete, awaiting first extraction with new code

### Quality Metrics Status

**Integration**: ✅ Complete (all 5 handlers updated, 14 return points)
**Logging**: ✅ `log_extraction_quality()` called before all returns
**Metadata**: ✅ `metadata["quality_metrics"] = quality` added before all returns

**Next Validation Step**: Run actual extraction to verify quality metrics appear in output

---

## 🚀 Next Steps

### Immediate Validation

1. **Test PDF Extraction**:

   ```bash
   python run_statement_test.py "uploads/2016 General Election Official Results.PDF"
   ```

   - Verify `metadata["quality_metrics"]` contains all expected fields
   - Check logs for `[ML] Extraction quality metrics` entries

1. **Test Other Handlers**:
   - CSV: Upload and parse CSV file
   - JSON: Parse OpenElections JSON
   - XLSX: Parse Excel file
   - HTML: Web scraping extraction

1. **Verify Dashboard**:
   - Start webapp: `python webapp/Smart_Elections_Parser_Webapp.py`
   - Navigate to `/quality_dashboard`
   - Verify charts render (after extractions run)

### ML Training Workflow

1. **Generate Training Data**:

   ```bash
   # Run multiple extractions with different OCR settings
   $env:OCR_CONFIDENCE_THRESHOLD="20"; python run_statement_test.py "file1.pdf"
   $env:OCR_CONFIDENCE_THRESHOLD="40"; python run_statement_test.py "file2.pdf"
   $env:OCR_CONFIDENCE_THRESHOLD="60"; python run_statement_test.py "file3.pdf"
   ```

1. **Export Dataset**:

   ```bash
   python scripts/export_ml_training_data.py --format all
   ```

1. **Train Model** (example):

   ```python
   import pandas as pd
   from sklearn.ensemble import RandomForestRegressor
   
   # Load training data
   df = pd.read_parquet('ml_datasets/training_data_*.parquet')
   
   # Features: OCR config
   X = df[[col for col in df.columns if col.startswith('ocr_')]]
   
   # Target: Extraction confidence
   y = df['quality_extraction_confidence']
   
   # Train
   model = RandomForestRegressor(n_estimators=100)
   model.fit(X, y)
   
   # Predict optimal settings
   prediction = model.predict(new_pdf_features)
   ```

1. **Deploy Model**:
   - Integrate model into adaptive OCR pipeline
   - Predict optimal settings based on PDF characteristics
   - Use predictions to initialize OCR search

### Optional Enhancements

1. **Add TXT Handler Quality Metrics** (if needed)
1. **Real-Time Quality Monitoring** (WebSocket streaming to dashboard)
1. **Quality Alerts** (notify when confidence drops below threshold)
1. **Automated Retraining** (scheduled model updates with new data)
1. **A/B Testing Framework** (compare OCR configurations)

---

## 📦 Deliverables Summary

### Code Changes

| File | Lines | Changes |
| ------ | ------- | --------- |
| `config.py` | 587→780 | +193 lines (OCR config, quality metrics framework) |
| `pdf_handler.py` | 6073→6089 | +16 lines (quality wrapper, 6 return paths) |
| `html_election_parser.py` | 1725→1738 | +13 lines (quality metrics) |
| `csv_handler.py` | 439→466 | +27 lines (quality metrics, 2 returns) |
| `json_handler.py` | 1458→1479 | +21 lines (quality metrics, 3 returns) |
| `xlsx_handler.py` | 470→484 | +14 lines (quality metrics, 2 returns) |
| `Smart_Elections_Parser_Webapp.py` | 2670→2745 | +75 lines (dashboard routes) |

### New Files

| File | Size | Purpose |
| ------ | ------ | --------- |
| `scripts/export_ml_training_data.py` | ~400 lines | ML dataset generator |
| `templates/quality_dashboard.html` | ~500 lines | Quality visualization dashboard |
| `docs/ml_training_data_export.md` | ~300 lines | ML training guide |

### Total Impact

- **Modified Files**: 7
- **New Files**: 3
- **Lines Added**: ~750
- **Handlers Updated**: 5 (PDF, HTML, CSV, JSON, XLSX)
- **Return Points Updated**: 14
- **Quality Metrics**: 15+ indicators
- **OCR Parameters**: 27 centralized

---

## 🎓 Key Architectural Decisions

1. **Centralized Config** - All OCR params in `config.py` (no circular imports)
1. **Handler-Agnostic** - Quality metrics work across all file types
1. **Metadata Enrichment** - Both `ocr_config` and `quality_metrics` in metadata
1. **Minimal Overhead** - Quality calculation is lightweight (no ML inference)
1. **ML-Ready** - Export formats (JSONL/CSV/Parquet) support common ML frameworks
1. **Observable** - Logs quality metrics for troubleshooting
1. **Filterable** - Dashboard and export support filtering by quality/handler/state

---

## 📖 References

- **Copilot Instructions**: `.github/copilot-instructions.md`
- **OCR Tuning**: `docs/ocr_tuning_guide.md`, `docs/ocr_tuning_reference.md`
- **ML Training**: `docs/ml_training_data_export.md`
- **Quality Dashboard**: Navigate to `/quality_dashboard` in webapp
- **Export Tool**: `python scripts/export_ml_training_data.py --help`

---

**Implementation Date**: January 9, 2026  
**Python Version**: 3.13.9  
**Status**: ✅ Complete - Ready for validation testing

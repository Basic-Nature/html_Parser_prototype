# OCR Tuning Guide — Smart Elections Parser

This document describes centralized OCR parameter tuning for complex PDF structures.

## Overview

All OCR-related thresholds live in `webapp/parser/config/ocr_tuning.py` and can be overridden via environment variables or runtime hooks for ML-based optimization.

## Key Parameters

### Quality Thresholds

| Parameter                     | Default | Description                                      | When to Tune                                                |
|-------------------------------|---------|--------------------------------------------------|-------------------------------------------------------------|
| `OCR_CONFIDENCE_THRESHOLD`    | 30      | Min per-word confidence (0-100)                  | Increase for high-quality docs; decrease for noisy scans    |
| `OCR_MIN_ALPHA_SIGNAL`        | 200     | Min alphabetic chars to avoid markup fallback    | Lower for sparse text; raise for text-heavy PDFs            |
| `OCR_AVG_CONF_ACCEPT`         | 70.0    | Stop search when avg confidence reaches this     | Raise to demand higher quality; lower for speed             |

### Search Space

| Parameter                     | Default                         | Description                                      | When to Tune                                                |
|-------------------------------|---------------------------------|--------------------------------------------------|-------------------------------------------------------------|
| `OCR_DPI_MIN`                 | 200                             | Starting DPI for page rendering                  | Raise for fine print; lower for speed                       |
| `OCR_DPI_MAX`                 | 350                             | Max DPI to try                                   | Increase for very small text; cap for large docs            |
| `OCR_DPI_STEP`                | 50                              | DPI increment                                    | Smaller steps = finer search but slower                     |
| `OCR_PSM_LIST`                | 6,4,3,11,12,1,13                | Tesseract page segmentation modes                | Reorder or remove modes based on layout                     |
| `OCR_OEM_LIST`                | 1,3,2,0                         | Tesseract OCR engine modes                       | Prefer 1 (LSTM) for modern; 0 for old typewriter            |
| `OCR_PREPROCESS_VARIANTS`     | none,gray,thresh,sharp_contrast | Image preprocessing                              | Add custom filters for specialized cases                    |

### Budget & Convergence

| Parameter                     | Default | Description                                      | When to Tune                                        |
|-------------------------------|---------|--------------------------------------------------|-----------------------------------------------------|
| `OCR_SAMPLE_BUDGET`           | 12      | Max trials during adaptive search                | Increase for exhaustive search; decrease for speed  |
| `OCR_MAX_RUNS`                | 20      | Hard cap on total OCR attempts                   | Safety valve; raise if search space is huge         |

### Layout & Table Heuristics

| Parameter                     | Default | Description                                      | When to Tune                                        |
|-------------------------------|---------|--------------------------------------------------|-----------------------------------------------------|
| `OCR_DENSE_LINE_THRESHOLD`    | 500     | Max chars per line before splitting              | Adjust for layout: lower for narrow columns         |
| `OCR_TABLE_SIGNAL_MIN_COLS`   | 2       | Min columns for table candidate                  | Raise for stricter table detection                  |
| `OCR_TABLE_SIGNAL_MIN_ROWS`   | 3       | Min rows for table candidate                     | Raise to filter out header-only fragments           |
| `OCR_ORIENTATION_THRESHOLD`   | 10.0    | Min confidence delta for rotation                | Raise to be more conservative with rotation         |

### Markup Detection

| Parameter                     | Default | Description                                      | When to Tune                                        |
|-------------------------------|---------|--------------------------------------------------|-----------------------------------------------------|
| `OCR_MARKUP_HTML_TAG_RATIO`   | 0.3     | Max tag/alpha ratio for markup detection         | Lower = stricter; higher = more tolerant            |

### Debug & Fast Mode

| Parameter                     | Default | Description                                      | When to Tune                                        |
|-------------------------------|---------|--------------------------------------------------|-----------------------------------------------------|
| `OCR_DEBUG_SAVE_IMAGES`       | 1       | Save sample raster images                        | Disable (0) in production for speed                 |
| `OCR_FAST_MODE_DPI_LIMIT`     | 250     | Cap DPI in fast mode                             | Balance speed vs quality                            |
| `OCR_FAST_MODE_SAMPLE_LIMIT`  | 6       | Cap sample trials in fast mode                   | Lower for quick probes                              |

## Usage Examples

### Manual Override (PowerShell)

```powershell
# Low-quality scan: lower confidence, higher DPI
$env:OCR_CONFIDENCE_THRESHOLD="20"
$env:OCR_DPI_MAX="400"
python scripts/run_pdf_ocr_force.py "MyBadScan.pdf"

# Complex layout: expand PSM search
$env:OCR_PSM_LIST="3,4,6,11,12,13"
python scripts/run_pdf_ocr_force.py "MultiColumn.pdf"

# Fast probe: restrict search space
$env:PDF_FAST_MODE="1"
$env:OCR_FAST_MODE_DPI_LIMIT="200"
$env:OCR_FAST_MODE_SAMPLE_LIMIT="4"
python scripts/run_pdf_ocr_force.py "QuickTest.pdf"
```

### Runtime ML Tuning (Python)

```python
from webapp.parser.config.ocr_tuning import OcrTuningConfig

# ML model predicts optimal params based on doc features
predicted_params = {
    "CONFIDENCE_THRESHOLD": 35,
    "DPI_MAX": 300,
    "PSM_LIST": [6, 4, 3],
}
OcrTuningConfig.override(predicted_params)

# Parse with tuned params
from webapp.parser.handlers.formats.pdf_handler import parse_pdf_election_results
headers, data, contest, metadata = parse_pdf_election_results("doc.pdf")

# Reset for next doc
OcrTuningConfig.reset_overrides()
```

### Logging Active Config

```python
from webapp.parser.config.ocr_tuning import OcrTuningConfig
from webapp.parser.utils.logger_singleton import logger

OcrTuningConfig.log_summary(logger)
# Logs all active params to console/session
```

## ML Training Hooks

Future ML integration can:

1. **Feature Extraction**: Extract doc length, image quality, text density, layout complexity
2. **Parameter Prediction**: Train model to predict optimal `DPI_MAX`, `PSM_LIST`, `CONFIDENCE_THRESHOLD`
3. **Feedback Loop**: Log OCR confidence, parse success, and user corrections to refine model
4. **A/B Testing**: Compare parameter sets across document types and track success rates

### Telemetry Collection

All OCR runs log:

- `metadata["ocr_config"]`: Active config snapshot
- `metadata["ocr_runs"]`: Per-run parameters and confidences
- `metadata["ocr_params"]`: Best parameter set used
- `metadata["ocr_confidence_avg"]`: Final average confidence

Use these for training data and offline analysis.

## Common Tuning Scenarios

### Scenario: Scanned Handwritten Ballots

- **Issue**: Low OCR confidence, poor text detection
- **Solution**:
  - Lower `OCR_CONFIDENCE_THRESHOLD` to 15-20
  - Increase `OCR_DPI_MAX` to 400
  - Add preprocessing: `OCR_PREPROCESS_VARIANTS="gray,thresh,sharp_contrast,denoise"`
  - Try legacy OCR engine: `OCR_OEM_LIST="0,2,1,3"`

### Scenario: Multi-Column Newspaper Layout

- **Issue**: Text runs together, columns misread
- **Solution**:
  - Prioritize column-aware PSM: `OCR_PSM_LIST="4,6,3"`
  - Lower `OCR_DENSE_LINE_THRESHOLD` to 300
  - Increase `OCR_TABLE_SIGNAL_MIN_COLS` to 3

### Scenario: High-Res Official Statement (Clean Print)

- **Issue**: Slow OCR with unnecessary trials
- **Solution**:
  - Narrow DPI range: `OCR_DPI_MIN=250`, `OCR_DPI_MAX=300`
  - Limit PSM: `OCR_PSM_LIST="6,3"`
  - Raise `OCR_AVG_CONF_ACCEPT` to 80 for early stop
  - Lower `OCR_SAMPLE_BUDGET` to 6

### Scenario: Rotated or Upside-Down Scans

- **Issue**: Text unreadable
- **Solution**:
  - Lower `OCR_ORIENTATION_THRESHOLD` to 5.0 for aggressive rotation
  - Use OSD-enabled PSM: `OCR_PSM_LIST="12,1,3"`
  - Increase `OCR_SAMPLE_BUDGET` to allow orientation trials

## Integration with Existing Flags

OCR tuning works alongside existing env flags:

- `ENABLE_OCR=1`: Enable OCR fallback
- `ENABLE_OCR_FORCE=1`: Force OCR even if text extraction succeeds
- `PDF_FAST_MODE=1`: Use fast-mode limits from `OCR_FAST_MODE_*` params
- `PDF_PROBE_MAX_PAGES=N`: Cap pages to process

All tuning parameters respect these flags and adjust search space accordingly.

## Validation & Testing

After tuning:

1. Run `python scripts/run_pdf_ocr_force.py "TestDoc.pdf"`
2. Check `output/ocr_debug/*_clean.txt` for improved text quality
3. Inspect `metadata["ocr_config"]` and `metadata["ocr_runs"]` in output JSON
4. Compare before/after CSV row counts and column richness

## Future Work

- [ ] ML model to predict params from doc preview (first 3 pages)
- [ ] Auto-tuning based on historical success rates per doc type
- [ ] Web UI for parameter override and A/B test comparison
- [ ] Regression suite with known-good param sets per document class

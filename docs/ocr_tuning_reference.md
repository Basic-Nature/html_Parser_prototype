# OCR Tuning Configuration — Quick Reference

All parameters live in `webapp/parser/config/ocr_tuning.py` and can be overridden via environment variables.

## Environment Variable Reference

```powershell
# Core Quality
$env:OCR_CONFIDENCE_THRESHOLD="30"       # Min word confidence (0-100)
$env:OCR_MIN_ALPHA_SIGNAL="200"          # Min alpha chars for non-markup
$env:OCR_AVG_CONF_ACCEPT="70.0"          # Stop search at this avg confidence

# Search Space
$env:OCR_DPI_MIN="200"                   # Starting DPI
$env:OCR_DPI_MAX="350"                   # Max DPI
$env:OCR_DPI_STEP="50"                   # DPI increment
$env:OCR_PSM_LIST="6,4,3,11,12,1,13"     # Tesseract PSM modes
$env:OCR_OEM_LIST="1,3,2,0"              # Tesseract OEM modes
$env:OCR_PREPROCESS_VARIANTS="none,gray,thresh,sharp_contrast"

# Budget
$env:OCR_SAMPLE_BUDGET="12"              # Max adaptive search trials
$env:OCR_MAX_RUNS="20"                   # Hard cap on total OCR attempts

# Layout
$env:OCR_DENSE_LINE_THRESHOLD="500"      # Max line length before split
$env:OCR_TABLE_SIGNAL_MIN_COLS="2"       # Min table columns
$env:OCR_TABLE_SIGNAL_MIN_ROWS="3"       # Min table rows
$env:OCR_ORIENTATION_THRESHOLD="10.0"    # Min rotation confidence delta

# Markup
$env:OCR_MARKUP_HTML_TAG_RATIO="0.3"     # Max tag/alpha ratio

# Debug
$env:OCR_DEBUG_SAVE_IMAGES="1"           # Save raster samples (0/1)
$env:OCR_FAST_MODE_DPI_LIMIT="250"       # Cap DPI in fast mode
$env:OCR_FAST_MODE_SAMPLE_LIMIT="6"      # Cap trials in fast mode
```

## Common Tuning Commands

### Low-Quality Scan

```powershell
$env:OCR_CONFIDENCE_THRESHOLD="20"; $env:OCR_DPI_MAX="400"
python scripts/run_pdf_ocr_force.py "BadScan.pdf"
```

### Complex Multi-Column Layout

```powershell
$env:OCR_PSM_LIST="4,6,3"; $env:OCR_DENSE_LINE_THRESHOLD="300"
python scripts/run_pdf_ocr_force.py "Newspaper.pdf"
```

### Fast Quality Probe

```powershell
$env:PDF_FAST_MODE="1"; $env:OCR_FAST_MODE_DPI_LIMIT="200"
python scripts/run_pdf_ocr_force.py "QuickTest.pdf"
```

### Rotated Document

```powershell
$env:OCR_ORIENTATION_THRESHOLD="5.0"; $env:OCR_PSM_LIST="12,1,3"
python scripts/run_pdf_ocr_force.py "UpsideDown.pdf"
```

See `docs/ocr_tuning_guide.md` for detailed scenarios and ML integration.

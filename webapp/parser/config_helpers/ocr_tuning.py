"""
OCR Tuning Parameters — Centralized Configuration
===================================================

This module centralizes all OCR-related thresholds, search spaces, and quality criteria
for PDF parsing. Values can be overridden via environment variables to support:
  - Manual tuning for specific document types
  - ML-based parameter optimization
  - A/B testing and regression benchmarks

Environment Variable Overrides:
  OCR_CONFIDENCE_THRESHOLD        — Minimum word confidence to accept (0-100)
  OCR_MIN_ALPHA_SIGNAL            — Min alphabetic chars to treat text as non-markup
  OCR_DPI_MIN, OCR_DPI_MAX        — DPI search range for adaptive OCR
  OCR_DPI_STEP                    — DPI increment (e.g., 50)
  OCR_PSM_LIST                    — Comma-separated Tesseract PSM modes to try
  OCR_OEM_LIST                    — Comma-separated Tesseract OEM modes to try
  OCR_SAMPLE_BUDGET               — Max sample trials in adaptive search
  OCR_MAX_RUNS                    — Max total OCR attempts before fallback
  OCR_AVG_CONF_ACCEPT             — Threshold to accept OCR run as "good enough"
  OCR_PREPROCESS_VARIANTS         — Comma-separated preprocessing strategies
  OCR_ORIENTATION_THRESHOLD       — Min confidence delta for rotation correction
  OCR_DENSE_LINE_THRESHOLD        — Max line length before splitting dense OCR text
  OCR_TABLE_SIGNAL_MIN_COLS       — Min candidate columns to consider as table
  OCR_TABLE_SIGNAL_MIN_ROWS       — Min candidate rows to consider as table
  OCR_MARKUP_HTML_TAG_RATIO       — Max ratio of tag chars to alpha for markup detection
  OCR_DEBUG_SAVE_IMAGES           — Save sample raster images for debugging (0/1)
  OCR_FAST_MODE_DPI_LIMIT         — Limit max DPI in fast mode
  OCR_FAST_MODE_SAMPLE_LIMIT      — Limit sample trials in fast mode

ML Tuning Hooks:
  - All parameters are exposed as class attributes
  - Can be overridden at runtime via OcrTuningConfig.override(param_dict)
  - Future: plug in ML model to predict optimal params based on doc features

Design Notes:
  - Defaults are conservative (favor recall over precision)
  - Complex PDFs (scanned, rotated, low-contrast) may need higher DPI or more PSM trials
  - ML can learn correlations: doc_length → dpi, text_density → psm, image_quality → preprocessing
"""

import os
from typing import List


class OcrTuningConfig:
    """
    Centralized OCR parameter registry with environment override support.
    """

    # --- Core Quality Thresholds ---
    CONFIDENCE_THRESHOLD: int = int(os.environ.get("OCR_CONFIDENCE_THRESHOLD", "30"))
    """Minimum per-word OCR confidence to accept (0-100). Lower = more tolerant of noisy text."""

    MIN_ALPHA_SIGNAL: int = int(os.environ.get("OCR_MIN_ALPHA_SIGNAL", "200"))
    """Min alphabetic characters to classify extracted text as non-markup (forces OCR if below)."""

    AVG_CONF_ACCEPT: float = float(os.environ.get("OCR_AVG_CONF_ACCEPT", "70.0"))
    """Average confidence threshold to accept an OCR run as "good enough" (stops search early)."""

    # --- Adaptive Search Space ---
    DPI_MIN: int = int(os.environ.get("OCR_DPI_MIN", "200"))
    DPI_MAX: int = int(os.environ.get("OCR_DPI_MAX", "350"))
    DPI_STEP: int = int(os.environ.get("OCR_DPI_STEP", "50"))
    """DPI search range and step for rendering pages. Higher DPI = better quality but slower."""

    PSM_LIST: List[int] = [
        int(x.strip()) for x in os.environ.get("OCR_PSM_LIST", "6,4,3,11,12,1,13").split(",") if x.strip()
    ]
    """Tesseract Page Segmentation Modes to try (order matters: preferred first).
    6=uniform block, 4=single column, 3=auto, 11=sparse text, 12=sparse+OSD, 1=auto+OSD, 13=raw line"""

    OEM_LIST: List[int] = [
        int(x.strip()) for x in os.environ.get("OCR_OEM_LIST", "1,3,2,0").split(",") if x.strip()
    ]
    """Tesseract OCR Engine Modes to try.
    1=LSTM, 3=Default (LSTM+legacy), 2=legacy+LSTM, 0=legacy only"""

    PREPROCESS_VARIANTS: List[str] = [
        x.strip() for x in os.environ.get("OCR_PREPROCESS_VARIANTS", "none,gray,thresh,sharp_contrast").split(",") if x.strip()
    ]
    """Image preprocessing strategies: none, gray, thresh, sharp_contrast, etc."""

    # --- Search Budget & Convergence ---
    SAMPLE_BUDGET: int = int(os.environ.get("OCR_SAMPLE_BUDGET", "12"))
    """Max trials during adaptive OCR parameter search (samples subset of pages)."""

    MAX_RUNS: int = int(os.environ.get("OCR_MAX_RUNS", "20"))
    """Max total OCR attempts before giving up (protects against runaway trials)."""

    # --- Orientation & Layout ---
    ORIENTATION_THRESHOLD: float = float(os.environ.get("OCR_ORIENTATION_THRESHOLD", "10.0"))
    """Min confidence delta (degrees) to trigger rotation correction."""

    DENSE_LINE_THRESHOLD: int = int(os.environ.get("OCR_DENSE_LINE_THRESHOLD", "500"))
    """Max characters per line before splitting (handles run-on OCR text)."""

    # --- Table Detection Heuristics ---
    TABLE_SIGNAL_MIN_COLS: int = int(os.environ.get("OCR_TABLE_SIGNAL_MIN_COLS", "2"))
    """Minimum columns to consider as table candidate."""

    TABLE_SIGNAL_MIN_ROWS: int = int(os.environ.get("OCR_TABLE_SIGNAL_MIN_ROWS", "3"))
    """Minimum rows to consider as table candidate."""

    # --- Markup Detection ---
    MARKUP_HTML_TAG_RATIO: float = float(os.environ.get("OCR_MARKUP_HTML_TAG_RATIO", "0.3"))
    """Max ratio of HTML tag chars to alphabetic chars before treating as markup-only."""

    # --- Debug & Fast Mode ---
    DEBUG_SAVE_IMAGES: bool = os.environ.get("OCR_DEBUG_SAVE_IMAGES", "1").lower() in {"1", "true", "yes"}
    """Save sample raster images during OCR for debugging."""

    FAST_MODE_DPI_LIMIT: int = int(os.environ.get("OCR_FAST_MODE_DPI_LIMIT", "250"))
    """Cap max DPI in fast mode (speeds up probe runs)."""

    FAST_MODE_SAMPLE_LIMIT: int = int(os.environ.get("OCR_FAST_MODE_SAMPLE_LIMIT", "6"))
    """Reduce sample trials in fast mode."""

    # --- ML Tuning Hooks ---
    _overrides: dict = {}

    @classmethod
    def override(cls, params: dict):
        """
        Runtime override for ML-based parameter tuning.
        Example:
            OcrTuningConfig.override({"CONFIDENCE_THRESHOLD": 40, "DPI_MAX": 400})
        """
        cls._overrides.update(params)

    @classmethod
    def get(cls, param: str, default=None):
        """Get parameter value with override support."""
        if param in cls._overrides:
            return cls._overrides[param]
        return getattr(cls, param, default)

    @classmethod
    def reset_overrides(cls):
        """Clear all runtime overrides (useful for tests)."""
        cls._overrides.clear()

    @classmethod
    def to_dict(cls) -> dict:
        """Export all current settings (for logging/telemetry)."""
        return {
            "CONFIDENCE_THRESHOLD": cls.get("CONFIDENCE_THRESHOLD"),
            "MIN_ALPHA_SIGNAL": cls.get("MIN_ALPHA_SIGNAL"),
            "AVG_CONF_ACCEPT": cls.get("AVG_CONF_ACCEPT"),
            "DPI_MIN": cls.get("DPI_MIN"),
            "DPI_MAX": cls.get("DPI_MAX"),
            "DPI_STEP": cls.get("DPI_STEP"),
            "PSM_LIST": cls.get("PSM_LIST"),
            "OEM_LIST": cls.get("OEM_LIST"),
            "PREPROCESS_VARIANTS": cls.get("PREPROCESS_VARIANTS"),
            "SAMPLE_BUDGET": cls.get("SAMPLE_BUDGET"),
            "MAX_RUNS": cls.get("MAX_RUNS"),
            "ORIENTATION_THRESHOLD": cls.get("ORIENTATION_THRESHOLD"),
            "DENSE_LINE_THRESHOLD": cls.get("DENSE_LINE_THRESHOLD"),
            "TABLE_SIGNAL_MIN_COLS": cls.get("TABLE_SIGNAL_MIN_COLS"),
            "TABLE_SIGNAL_MIN_ROWS": cls.get("TABLE_SIGNAL_MIN_ROWS"),
            "MARKUP_HTML_TAG_RATIO": cls.get("MARKUP_HTML_TAG_RATIO"),
            "DEBUG_SAVE_IMAGES": cls.get("DEBUG_SAVE_IMAGES"),
            "FAST_MODE_DPI_LIMIT": cls.get("FAST_MODE_DPI_LIMIT"),
            "FAST_MODE_SAMPLE_LIMIT": cls.get("FAST_MODE_SAMPLE_LIMIT"),
        }

    @classmethod
    def log_summary(cls, logger=None):
        """Log current config for diagnostics."""
        summary = cls.to_dict()
        msg = "[OCR Tuning] Active configuration:"
        if logger:
            logger.info({"level": "INFO", "type": "config", "message": msg, "params": summary})
        else:
            print(msg)
            for k, v in summary.items():
                print(f"  {k}: {v}")


# Convenience exports
CONFIDENCE_THRESHOLD = OcrTuningConfig.CONFIDENCE_THRESHOLD
MIN_ALPHA_SIGNAL = OcrTuningConfig.MIN_ALPHA_SIGNAL
AVG_CONF_ACCEPT = OcrTuningConfig.AVG_CONF_ACCEPT
DPI_MIN = OcrTuningConfig.DPI_MIN
DPI_MAX = OcrTuningConfig.DPI_MAX
DPI_STEP = OcrTuningConfig.DPI_STEP
PSM_LIST = OcrTuningConfig.PSM_LIST
OEM_LIST = OcrTuningConfig.OEM_LIST
PREPROCESS_VARIANTS = OcrTuningConfig.PREPROCESS_VARIANTS
SAMPLE_BUDGET = OcrTuningConfig.SAMPLE_BUDGET
MAX_RUNS = OcrTuningConfig.MAX_RUNS
ORIENTATION_THRESHOLD = OcrTuningConfig.ORIENTATION_THRESHOLD
DENSE_LINE_THRESHOLD = OcrTuningConfig.DENSE_LINE_THRESHOLD
TABLE_SIGNAL_MIN_COLS = OcrTuningConfig.TABLE_SIGNAL_MIN_COLS
TABLE_SIGNAL_MIN_ROWS = OcrTuningConfig.TABLE_SIGNAL_MIN_ROWS
MARKUP_HTML_TAG_RATIO = OcrTuningConfig.MARKUP_HTML_TAG_RATIO
DEBUG_SAVE_IMAGES = OcrTuningConfig.DEBUG_SAVE_IMAGES
FAST_MODE_DPI_LIMIT = OcrTuningConfig.FAST_MODE_DPI_LIMIT
FAST_MODE_SAMPLE_LIMIT = OcrTuningConfig.FAST_MODE_SAMPLE_LIMIT

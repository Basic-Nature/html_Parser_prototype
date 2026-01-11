"""
OCR Configuration Helper Functions
===================================
Utility functions for OCR configuration telemetry and logging.
Separated to avoid circular imports during config initialization.
"""

def get_ocr_config_dict(config_module) -> dict:
    """
    Export all active OCR tuning settings (for logging/telemetry).
    
    Args:
        config_module: The config module containing OCR parameters
        
    Returns:
        Dictionary of all OCR configuration values
    """
    return {
        "CONFIDENCE_THRESHOLD": config_module.OCR_CONFIDENCE_THRESHOLD,
        "MIN_ALPHA_SIGNAL": config_module.OCR_MIN_ALPHA_SIGNAL,
        "AVG_CONF_ACCEPT": config_module.OCR_AVG_CONF_ACCEPT,
        "DPI_MIN": config_module.OCR_DPI_MIN,
        "DPI_MAX": config_module.OCR_DPI_MAX,
        "DPI_STEP": config_module.OCR_DPI_STEP,
        "PSM_LIST": config_module.OCR_PSM_LIST,
        "OEM_LIST": config_module.OCR_OEM_LIST,
        "PREPROCESS_VARIANTS": config_module.OCR_PREPROCESS_VARIANTS,
        "SAMPLE_BUDGET": config_module.OCR_SAMPLE_BUDGET,
        "MAX_RUNS": config_module.OCR_MAX_RUNS,
        "ORIENTATION_THRESHOLD": config_module.OCR_ORIENTATION_THRESHOLD,
        "DENSE_LINE_THRESHOLD": config_module.OCR_DENSE_LINE_THRESHOLD,
        "TABLE_SIGNAL_MIN_COLS": config_module.OCR_TABLE_SIGNAL_MIN_COLS,
        "TABLE_SIGNAL_MIN_ROWS": config_module.OCR_TABLE_SIGNAL_MIN_ROWS,
        "MARKUP_HTML_TAG_RATIO": config_module.OCR_MARKUP_HTML_TAG_RATIO,
        "DEBUG_SAVE_IMAGES": config_module.OCR_DEBUG_SAVE_IMAGES,
        "FAST_MODE_DPI_LIMIT": config_module.OCR_FAST_MODE_DPI_LIMIT,
        "FAST_MODE_SAMPLE_LIMIT": config_module.OCR_FAST_MODE_SAMPLE_LIMIT,
        "PDF_FAST_MODE": config_module.PDF_FAST_MODE,
        "PDF_PROBE_MAX_PAGES": config_module.PDF_PROBE_MAX_PAGES,
    }


def log_ocr_config_summary(config_module, logger_instance=None):
    """
    Log current OCR config for diagnostics.
    
    Args:
        config_module: The config module containing OCR parameters
        logger_instance: Optional logger instance (uses print if None)
    """
    summary = get_ocr_config_dict(config_module)
    msg = "[OCR Tuning] Active configuration:"
    if logger_instance:
        logger_instance.info({"level": "INFO", "type": "config", "message": msg, "params": summary})
    else:
        print(msg)
        for k, v in summary.items():
            print(f"  {k}: {v}")

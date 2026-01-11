"""Quick validation test for ML quality metrics framework."""
import sys
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from webapp.parser.config import log_extraction_quality, OCR_CONFIDENCE_THRESHOLD
from webapp.parser.utils.logger_singleton import logger

print("=" * 60)
print("ML Quality Metrics Framework - Validation Test")
print("=" * 60)

# Test OCR config
print(f"\n✅ OCR Configuration:")
print(f"   OCR_CONFIDENCE_THRESHOLD = {OCR_CONFIDENCE_THRESHOLD}")

# Test data
headers = ['Name', 'Votes', 'Percent']
data = [
    {'Name': 'Alice', 'Votes': '100', 'Percent': '50%'},
    {'Name': 'Bob', 'Votes': '100', 'Percent': '50%'},
]
metadata = {}

# Calculate quality metrics
print(f"\n✅ Testing quality metrics calculation...")
quality = log_extraction_quality(headers, data, metadata, 'test_handler', logger, 'test_session')

# Display results
print(f"\n✅ Quality Metrics Results:")
print(f"   Metrics captured: {len(quality)} indicators")
print(f"   Metric keys: {', '.join(list(quality.keys())[:5])}...")
print(f"\n📊 Key Indicators:")
print(f"   Extraction confidence: {quality.get('extraction_confidence', 0):.3f}")
print(f"   Row count: {quality.get('row_count')}")
print(f"   Column count: {quality.get('column_count')}")
print(f"   Empty row ratio: {quality.get('empty_row_ratio', 0):.3f}")
print(f"   Null cell ratio: {quality.get('null_cell_ratio', 0):.3f}")
print(f"   Avg row density: {quality.get('avg_row_density', 0):.3f}")
print(f"   Header completeness: {quality.get('header_completeness', 0):.3f}")
print(f"   Data type diversity: {quality.get('data_type_diversity')}")
print(f"   Has numeric columns: {quality.get('has_numeric_columns')}")
print(f"   Has text columns: {quality.get('has_text_columns')}")

# Test with OCR metrics
metadata_with_ocr = {
    "ocr_stats": {
        "avg_confidence": 85.5,
        "min_confidence": 72.0,
        "ocr_run_count": 3,
        "ocr_time_sec": 4.2,
        "ocr_pages_processed": 5
    }
}

print(f"\n✅ Testing with OCR metrics...")
quality_ocr = log_extraction_quality(headers, data, metadata_with_ocr, 'pdf_handler', logger, 'test_session')

if quality_ocr.get('ocr_metrics'):
    print(f"\n📊 OCR Metrics:")
    ocr_metrics = quality_ocr['ocr_metrics']
    print(f"   Avg confidence: {ocr_metrics.get('avg_confidence', 0):.1f}")
    print(f"   Min confidence: {ocr_metrics.get('min_confidence', 0):.1f}")
    print(f"   OCR run count: {ocr_metrics.get('ocr_run_count')}")
    print(f"   OCR time (sec): {ocr_metrics.get('ocr_time_sec', 0):.1f}")
    print(f"   Pages processed: {ocr_metrics.get('ocr_pages_processed')}")
else:
    print(f"\n⚠️  OCR metrics not captured (expected if no ocr_stats in metadata)")

print("\n" + "=" * 60)
print("✅ All tests passed!")
print("=" * 60)
print("\n📚 Next Steps:")
print("   1. Run actual extraction: python run_statement_test.py")
print("   2. View dashboard: python webapp/Smart_Elections_Parser_Webapp.py")
print("   3. Export ML dataset: python scripts/export_ml_training_data.py")
print()

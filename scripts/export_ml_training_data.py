"""
Export ML training dataset from extraction results.

Generates (config → quality) pairs for training ML models that can:
- Predict optimal OCR parameters for different document types
- Tune adaptive search strategies dynamically
- Flag low-quality extractions for review
- Learn document-specific parameter profiles

Output formats:
- JSONL: One extraction per line (for streaming/incremental training)
- CSV: Tabular format (for analytics/visualization)
- Parquet: Columnar format (for efficient ML pipelines)
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False
    import json as orjson

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from webapp.parser.config import OUTPUT_DIR, LOG_DIR


def scan_output_metadata(output_dir: Path, use_cache: bool = True) -> List[Dict[str, Any]]:
    """Scan output folders for metadata.json files with extraction results.
    
    Args:
        output_dir: Directory containing extraction output folders
        use_cache: If True, cache results to speed up repeated scans
    
    Returns:
        List of metadata dictionaries with quality metrics
    """
    results = []
    
    if not output_dir.exists():
        print(f"Output directory does not exist: {output_dir}")
        return results
    
    # Cache file for faster re-scans
    cache_file = output_dir / ".ml_export_cache.json"
    folder_mtimes = {}
    
    # Load cache if available
    cached_results = {}
    if use_cache and cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                cached_results = cache_data.get("results", {})
                print(f"Loaded {len(cached_results)} cached entries")
        except Exception:
            pass
    
    for folder in output_dir.iterdir():
        if not folder.is_dir():
            continue
        
        metadata_file = folder / "metadata.json"
        if not metadata_file.exists():
            continue
        
        folder_name = folder.name
        try:
            # Check if we can use cached version
            mtime = metadata_file.stat().st_mtime
            folder_mtimes[folder_name] = mtime
            
            if (use_cache and folder_name in cached_results 
                and cached_results[folder_name].get("_cache_mtime") == mtime):
                # Use cached result
                results.append(cached_results[folder_name]["metadata"])
                continue
            
            # Load fresh metadata
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Only include if it has both config and quality metrics
            if "ocr_config" in metadata and "quality_metrics" in metadata:
                # Add folder info for tracking
                metadata["_folder"] = folder_name
                metadata["_metadata_file"] = str(metadata_file)
                results.append(metadata)
                
                # Update cache entry
                if use_cache:
                    cached_results[folder_name] = {
                        "_cache_mtime": mtime,
                        "metadata": metadata
                    }
        except Exception as e:
            print(f"Warning: Failed to load {metadata_file}: {e}")
            continue
    
    # Save updated cache
    if use_cache and cached_results:
        try:
            cache_data = {
                "generated": datetime.now().isoformat(),
                "total_folders": len(folder_mtimes),
                "cached_folders": len(cached_results),
                "results": cached_results
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
            print(f"Cached {len(cached_results)} entries for faster future scans")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
    
    return results


def flatten_for_csv(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested dictionaries for CSV export."""
    flat = {}
    
    # Extract top-level fields
    for key in ["handler", "state", "county", "contest", "row_count", "column_count", "_folder"]:
        flat[key] = metadata.get(key)
    
    # Flatten OCR config (prefix with ocr_)
    ocr_config = metadata.get("ocr_config", {})
    for key, value in ocr_config.items():
        if isinstance(value, (list, dict)):
            flat[f"ocr_{key}"] = json.dumps(value)
        else:
            flat[f"ocr_{key}"] = value
    
    # Flatten quality metrics (prefix with quality_)
    quality = metadata.get("quality_metrics", {})
    for key, value in quality.items():
        if key in ["ocr_metrics", "table_metrics"]:
            # Nested metrics - flatten one level deeper
            nested = value or {}
            for nested_key, nested_value in nested.items():
                flat[f"quality_{key}_{nested_key}"] = nested_value
        elif isinstance(value, (list, dict)):
            flat[f"quality_{key}"] = json.dumps(value)
        else:
            flat[f"quality_{key}"] = value
    
    return flat


def export_jsonl(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Export as JSONL (one extraction per line)."""
    with open(output_path, 'wb') as f:
        for result in results:
            if ORJSON_AVAILABLE:
                line = orjson.dumps(result) + b'\n'
            else:
                line = (json.dumps(result) + '\n').encode('utf-8')
            f.write(line)
    print(f"✅ Exported {len(results)} extractions to {output_path}")


def export_csv(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Export as CSV (tabular format)."""
    if not PANDAS_AVAILABLE:
        print("⚠️  pandas not available - skipping CSV export")
        print("   Install with: pip install pandas")
        return
    
    # Flatten all results
    flattened = [flatten_for_csv(r) for r in results]
    
    # Create DataFrame
    df = pd.DataFrame(flattened)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Exported {len(results)} extractions to {output_path} ({len(df.columns)} columns)")


def export_parquet(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Export as Parquet (columnar format for ML).

    Optional dependency handling:
    - Requires `pandas` and `pyarrow`. If either is missing, prints guidance and skips.
    - Resolves IDE/reportMissingImports by importing `pyarrow` inside function and guarding.
    """
    if not PANDAS_AVAILABLE:
        print("⚠️  pandas not available - skipping Parquet export")
        print("   Install with: pip install pandas pyarrow")
        return

    # Import pyarrow lazily and guard for environments without it
    try:
        import pyarrow  # type: ignore[reportMissingImports]
    except Exception:
        print("⚠️  pyarrow not available - skipping Parquet export")
        print("   Install with: pip install pyarrow")
        return

    # Flatten all results
    flattened = [flatten_for_csv(r) for r in results]

    # Create DataFrame
    df = pd.DataFrame(flattened)

    # Save to Parquet
    try:
        df.to_parquet(output_path, index=False, engine='pyarrow', compression='snappy')
        print(f"✅ Exported {len(results)} extractions to {output_path} ({len(df.columns)} columns)")
    except Exception as exc:
        print(f"❌ Failed Parquet export: {exc}")


def generate_summary_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary statistics about the dataset."""
    stats = {
        "total_extractions": len(results),
        "by_handler": defaultdict(int),
        "by_state": defaultdict(int),
        "avg_confidence": None,
        "avg_rows": None,
        "quality_distribution": {
            "high": 0,  # confidence >= 0.8
            "medium": 0,  # 0.5 <= confidence < 0.8
            "low": 0,  # confidence < 0.5
            "unknown": 0,  # no confidence score
        },
    }
    
    confidences = []
    row_counts = []
    
    for result in results:
        handler = result.get("handler", "unknown")
        stats["by_handler"][handler] += 1
        
        state = result.get("state", "unknown")
        stats["by_state"][state] += 1
        
        quality = result.get("quality_metrics", {})
        conf = quality.get("extraction_confidence")
        if conf is not None:
            confidences.append(conf)
            if conf >= 0.8:
                stats["quality_distribution"]["high"] += 1
            elif conf >= 0.5:
                stats["quality_distribution"]["medium"] += 1
            else:
                stats["quality_distribution"]["low"] += 1
        else:
            stats["quality_distribution"]["unknown"] += 1
        
        row_count = result.get("row_count")
        if row_count is not None:
            row_counts.append(row_count)
    
    if confidences:
        stats["avg_confidence"] = sum(confidences) / len(confidences)
    if row_counts:
        stats["avg_rows"] = sum(row_counts) / len(row_counts)
    
    # Convert defaultdicts to regular dicts for JSON serialization
    stats["by_handler"] = dict(stats["by_handler"])
    stats["by_state"] = dict(stats["by_state"])
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Export ML training dataset from extraction results"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory containing extraction output folders (default: project output/)",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=PROJECT_ROOT / "ml_datasets",
        help="Directory to write ML training datasets (default: project ml_datasets/)",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "csv", "parquet", "all"],
        default="all",
        help="Output format(s) to generate (default: all)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        help="Minimum extraction confidence to include (0.0 - 1.0)",
    )
    parser.add_argument(
        "--handler",
        help="Filter by specific handler (pdf_handler, csv_handler, etc.)",
    )
    parser.add_argument(
        "--state",
        help="Filter by specific state",
    )
    
    args = parser.parse_args()
    
    # Scan for metadata files
    print(f"Scanning {args.output_dir} for extraction metadata...")
    results = scan_output_metadata(args.output_dir)
    print(f"Found {len(results)} extractions with quality metrics")
    
    # Apply filters
    if args.min_confidence is not None:
        before = len(results)
        results = [
            r for r in results
            if r.get("quality_metrics", {}).get("extraction_confidence", 0) >= args.min_confidence
        ]
        print(f"Filtered to {len(results)} extractions (confidence >= {args.min_confidence})")
    
    if args.handler:
        before = len(results)
        results = [r for r in results if r.get("handler") == args.handler]
        print(f"Filtered to {len(results)} extractions (handler={args.handler})")
    
    if args.state:
        before = len(results)
        results = [r for r in results if r.get("state") == args.state]
        print(f"Filtered to {len(results)} extractions (state={args.state})")
    
    if not results:
        print("❌ No extractions match filters")
        return
    
    # Create export directory
    args.export_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for this export
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export in requested format(s)
    if args.format in ["jsonl", "all"]:
        output_path = args.export_dir / f"training_data_{timestamp}.jsonl"
        export_jsonl(results, output_path)
    
    if args.format in ["csv", "all"]:
        output_path = args.export_dir / f"training_data_{timestamp}.csv"
        export_csv(results, output_path)
    
    if args.format in ["parquet", "all"]:
        output_path = args.export_dir / f"training_data_{timestamp}.parquet"
        export_parquet(results, output_path)
    
    # Generate and save summary statistics
    stats = generate_summary_stats(results)
    stats_path = args.export_dir / f"training_data_{timestamp}_summary.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Exported summary statistics to {stats_path}")
    
    print("\n📊 Dataset Summary:")
    print(f"   Total extractions: {stats['total_extractions']}")
    print(f"   Avg confidence: {stats['avg_confidence']:.3f}" if stats['avg_confidence'] else "   Avg confidence: N/A")
    print(f"   Avg rows: {stats['avg_rows']:.1f}" if stats['avg_rows'] else "   Avg rows: N/A")
    print(f"   Quality distribution:")
    for level, count in stats['quality_distribution'].items():
        print(f"      {level}: {count}")
    print(f"   By handler: {dict(list(stats['by_handler'].items())[:5])}")
    print(f"   By state: {dict(list(stats['by_state'].items())[:5])}")


if __name__ == "__main__":
    main()

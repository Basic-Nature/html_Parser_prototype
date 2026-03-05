#!/usr/bin/env python
"""
URL Status Report Generator

Generates a comprehensive report showing:
- URLs from urls.txt
- Processing status from .processed_urls
- Production status from Google Sheets/warehouse
- Side-by-side comparison for gap analysis

Usage:
    python tools/url_status_report.py [--output-dir output] [--format md|csv|both]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add webapp to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from webapp.parser.config import PROCESSED_URLS_FILE, URL_LIST_FILE
    from webapp.parser.utils.database_comparison import check_existing_finalized_data
    from webapp.parser.utils.logger_singleton import logger
    from webapp.parser.utils.misc_utils import extract_url_and_label, load_processed_urls
except ImportError as e:
    print(f"Error importing parser modules: {e}")
    print("Make sure you're running from the repository root: python tools/url_status_report.py")
    sys.exit(1)


def load_urls_from_file(urls_file: Path) -> List[Tuple[str, Optional[str]]]:
    """
    Load URLs from urls.txt, extracting URL and label/metadata.
    
    Returns:
        List of (url, label) tuples
    """
    if not urls_file.exists():
        logger.warning(f"URLs file not found: {urls_file}")
        return []
    
    urls = []
    with open(urls_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Extract URL and label
            url, label = extract_url_and_label(line, allowlist_bypass=True)
            
            if url:
                urls.append((url, label or line))
            else:
                # Try to parse as tab-delimited schema line
                parts = line.split('\t')
                if len(parts) >= 7 and parts[6].startswith('http'):
                    url = parts[6].strip()
                    label = f"{parts[0]} {parts[1]} {parts[2]}" if parts[0] != 'TBD' else line
                    urls.append((url, label))
    
    return urls


def check_production_status(
    url: str,
    session_id: Optional[str] = None
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Check if URL is in production (Google Sheets or warehouse).
    
    Returns:
        (in_production, source, metadata)
    """
    try:
        return check_existing_finalized_data(
            url,
            session_id=session_id
        )
    except Exception as e:
        logger.error(f"Error checking production status for {url}: {e}")
        return False, None, None


def generate_markdown_report(
    report_data: List[Dict[str, Any]],
    output_path: Path,
    stats: Dict[str, int]
) -> None:
    """Generate markdown report with status breakdown."""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# URL Status Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary statistics
        f.write("## Summary\n\n")
        f.write(f"- **Total URLs**: {stats['total']}\n")
        f.write(f"- **Parsed (Success)**: {stats['parsed_success']} ✅\n")
        f.write(f"- **Failed/Error**: {stats['failed']} ❌\n")
        f.write(f"- **In Production**: {stats['in_production']} 📦\n")
        f.write(f"- **Skipped (Data Exists)**: {stats['skipped_exists']} ⏭️\n")
        f.write(f"- **Pending**: {stats['pending']} ⏳\n")
        f.write(f"- **Other Status**: {stats['other']} ⚠️\n\n")
        
        # Production sources breakdown
        if stats['production_sources']:
            f.write("### Production Sources\n\n")
            for source, count in sorted(stats['production_sources'].items()):
                f.write(f"- {source}: {count}\n")
            f.write("\n")
        
        # Gap analysis
        gap_count = stats['pending'] + stats['failed']
        if gap_count > 0:
            f.write("### Gap Analysis\n\n")
            f.write(f"**{gap_count} URLs** need attention (pending or failed)\n\n")
        
        # Detailed table
        f.write("## Detailed Status\n\n")
        f.write("| # | URL | Label | Parser Status | Production | Last Processed | Retry Count |\n")
        f.write("|---|-----|-------|---------------|------------|----------------|-------------|\n")
        
        for idx, entry in enumerate(report_data, 1):
            url_display = entry['url'][:60] + '...' if len(entry['url']) > 60 else entry['url']
            label_display = entry['label'][:50] + '...' if len(entry['label']) > 50 else entry['label']
            
            # Status badges
            parser_status = entry['parser_status']
            if parser_status == 'success':
                status_badge = '✅ Success'
            elif parser_status in ('fail', 'error'):
                status_badge = f'❌ {parser_status.title()}'
            elif parser_status == 'skipped_data_exists':
                status_badge = '⏭️ Skipped'
            elif parser_status == 'pending':
                status_badge = '⏳ Pending'
            elif parser_status in ('partial', 'cancelled'):
                status_badge = f'⚠️ {parser_status.title()}'
            else:
                status_badge = parser_status or '-'
            
            # Production badge
            if entry['in_production']:
                prod_badge = f"📦 {entry['production_source'] or 'Yes'}"
            else:
                prod_badge = '○ No'
            
            # Last processed
            last_proc = entry.get('last_processed', '-')
            if last_proc != '-':
                try:
                    dt = datetime.strptime(last_proc, '%Y-%m-%d %H:%M:%S')
                    last_proc = dt.strftime('%Y-%m-%d')
                except Exception:
                    pass
            
            retry_count = entry.get('retry_count', 0)
            
            f.write(f"| {idx} | {url_display} | {label_display} | {status_badge} | {prod_badge} | {last_proc} | {retry_count} |\n")
        
        f.write("\n---\n\n")
        f.write("**Legend**:\n")
        f.write("- ✅ Success: Parsed successfully\n")
        f.write("- ❌ Fail/Error: Parsing failed\n")
        f.write("- ⏭️ Skipped: Already in production\n")
        f.write("- ⏳ Pending: Not yet processed\n")
        f.write("- ⚠️ Other: Partial, cancelled, etc.\n")
        f.write("- 📦 Production: In Google Sheets or warehouse\n")
        f.write("- ○ Not in production\n")


def generate_csv_report(
    report_data: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """Generate CSV report for spreadsheet analysis."""
    
    import csv
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'URL',
            'Label',
            'Parser Status',
            'In Production',
            'Production Source',
            'Last Processed',
            'Retry Count',
            'State',
            'County',
            'Contest'
        ])
        
        # Data rows
        for entry in report_data:
            writer.writerow([
                entry['url'],
                entry['label'],
                entry['parser_status'] or 'pending',
                'Yes' if entry['in_production'] else 'No',
                entry['production_source'] or '-',
                entry.get('last_processed', '-'),
                entry.get('retry_count', 0),
                entry.get('state', '-'),
                entry.get('county', '-'),
                entry.get('contest', '-')
            ])


def main():
    parser = argparse.ArgumentParser(
        description='Generate URL status report comparing parsed URLs vs production data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for reports (default: output)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['md', 'csv', 'both'],
        default='both',
        help='Output format: md (markdown), csv, or both (default: both)'
    )
    parser.add_argument(
        '--session-id',
        type=str,
        default=None,
        help='Session ID for logging'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    md_path = output_dir / f'url_status_report_{timestamp}.md'
    csv_path = output_dir / f'url_status_report_{timestamp}.csv'
    
    print(f"\n{'='*70}")
    print("URL Status Report Generator")
    print(f"{'='*70}\n")
    
    # Load URLs from urls.txt
    print(f"[1/4] Loading URLs from {URL_LIST_FILE}...")
    urls_list = load_urls_from_file(URL_LIST_FILE)
    print(f"      Found {len(urls_list)} URLs\n")
    
    # Load processed URLs
    print(f"[2/4] Loading processing history from {PROCESSED_URLS_FILE}...")
    processed_map = load_processed_urls()
    print(f"      Found {len(processed_map)} processed entries\n")
    
    # Check production status for each URL
    print("[3/4] Checking production status (Google Sheets + warehouse)...")
    report_data = []
    stats = {
        'total': len(urls_list),
        'parsed_success': 0,
        'failed': 0,
        'in_production': 0,
        'skipped_exists': 0,
        'pending': 0,
        'other': 0,
        'production_sources': defaultdict(int)
    }
    
    for idx, (url, label) in enumerate(urls_list, 1):
        # Progress indicator
        if idx % 10 == 0 or idx == len(urls_list):
            print(f"      Progress: {idx}/{len(urls_list)}", end='\r')
        
        # Get processing status
        processed_entry = processed_map.get(url)
        parser_status = None
        last_processed = None
        retry_count = 0
        state = None
        county = None
        contest = None
        
        if processed_entry:
            parser_status = processed_entry.get('status')
            last_processed = processed_entry.get('timestamp')
            retry_count = processed_entry.get('retry_count', 0)
            state = processed_entry.get('state')
            county = processed_entry.get('county')
            contest = processed_entry.get('contest')
        else:
            parser_status = 'pending'
        
        # Check production status
        in_production, prod_source, prod_metadata = check_production_status(
            url,
            session_id=args.session_id
        )
        
        # Update metadata from production if available
        if prod_metadata:
            state = state or prod_metadata.get('state')
            county = county or prod_metadata.get('county')
            contest = contest or prod_metadata.get('contest')
        
        # Build entry
        entry = {
            'url': url,
            'label': label,
            'parser_status': parser_status,
            'in_production': in_production,
            'production_source': prod_source,
            'last_processed': last_processed,
            'retry_count': retry_count,
            'state': state,
            'county': county,
            'contest': contest
        }
        
        report_data.append(entry)
        
        # Update stats
        if parser_status == 'success':
            stats['parsed_success'] += 1
        elif parser_status in ('fail', 'error'):
            stats['failed'] += 1
        elif parser_status == 'skipped_data_exists':
            stats['skipped_exists'] += 1
        elif parser_status == 'pending':
            stats['pending'] += 1
        else:
            stats['other'] += 1
        
        if in_production:
            stats['in_production'] += 1
            if prod_source:
                stats['production_sources'][prod_source] += 1
    
    print("\n      Complete!\n")
    
    # Generate reports
    print("[4/4] Generating reports...")
    
    if args.format in ('md', 'both'):
        generate_markdown_report(report_data, md_path, stats)
        print(f"      ✓ Markdown report: {md_path}")
    
    if args.format in ('csv', 'both'):
        generate_csv_report(report_data, csv_path)
        print(f"      ✓ CSV report: {csv_path}")
    
    print(f"\n{'='*70}")
    print("Summary Statistics")
    print(f"{'='*70}\n")
    print(f"  Total URLs:           {stats['total']}")
    print(f"  Parsed (Success):     {stats['parsed_success']} ({stats['parsed_success']/stats['total']*100:.1f}%)")
    print(f"  Failed/Error:         {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
    print(f"  In Production:        {stats['in_production']} ({stats['in_production']/stats['total']*100:.1f}%)")
    print(f"  Skipped (Exists):     {stats['skipped_exists']} ({stats['skipped_exists']/stats['total']*100:.1f}%)")
    print(f"  Pending:              {stats['pending']} ({stats['pending']/stats['total']*100:.1f}%)")
    print(f"  Other:                {stats['other']} ({stats['other']/stats['total']*100:.1f}%)")
    
    if stats['production_sources']:
        print("\n  Production Sources:")
        for source, count in sorted(stats['production_sources'].items()):
            print(f"    - {source}: {count}")
    
    print(f"\n{'='*70}\n")
    
    # Gap analysis
    gap_count = stats['pending'] + stats['failed']
    if gap_count > 0:
        print(f"⚠️  Gap Analysis: {gap_count} URLs need attention")
        print(f"   - {stats['pending']} pending (not yet processed)")
        print(f"   - {stats['failed']} failed (need retry/investigation)")
        print("\n💡 Tip: Review failed URLs in the report for common error patterns\n")
    else:
        print("✅ All URLs processed successfully!\n")


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
Phase 2: Pre-migration header audit.
Scans all output folder CSVs and validates header confidence.
Flags low-confidence headers to flagged_headers.jsonl for manual review.

Usage:
  python scripts/audit_headers_before_promotion.py [--output-dir OUTPUT] [--threshold SCORE] [--limit ROWS]
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webapp.parser.config import LOG_DIR, OUTPUT_DIR
from webapp.parser.utils.header_confidence import validate_row_headers
from webapp.parser.utils.logger_singleton import logger

# Ensure log directory exists
os.makedirs(str(LOG_DIR), exist_ok=True)

FLAGGED_HEADERS_FILE = os.path.join(str(LOG_DIR), 'flagged_headers.jsonl')
AUDIT_REPORT_FILE = os.path.join(str(LOG_DIR), 'header_audit_report.json')


def audit_csv_headers(csv_path: str, threshold: float = 0.85, limit: int = 500) -> dict:
    """
    Audit a single CSV file for header confidence.
    
    Returns:
        {
            'file': str,
            'headers': list[str],
            'confidence_scores': dict[str, float],
            'pass': bool,
            'flagged_headers': list[str],
            'row_count': int,
            'timestamp': str
        }
    """
    result = {
        'file': os.path.basename(csv_path),
        'headers': [],
        'confidence_scores': {},
        'pass': False,
        'flagged_headers': [],
        'row_count': 0,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'errors': None
    }
    
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                result['errors'] = 'No headers found'
                return result
            
            headers = list(reader.fieldnames)
            result['headers'] = headers
            
            # Critical columns to validate
            critical = ['candidate', 'party', 'votes']
            
            # Score each header
            passed, scores, flagged = validate_row_headers(headers, critical, threshold)
            
            result['confidence_scores'] = scores
            result['flagged_headers'] = flagged
            result['pass'] = passed
            
            # Count rows
            for i, row in enumerate(reader):
                if i >= limit:
                    break
                result['row_count'] = i + 1
    
    except Exception as e:
        result['errors'] = str(e)
    
    return result


def audit_output_directory(output_dir: str, threshold: float = 0.85, limit: int = 500) -> dict:
    """
    Scan all CSV files in output directory and audit headers.
    
    Returns audit summary with pass/fail counts and flagged files.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        logger.warning({
            'level': 'WARNING',
            'type': 'audit',
            'message': f'Output directory not found: {output_dir}',
            'session_id': None
        })
        return {'error': f'Directory not found: {output_dir}', 'audited': []}
    
    audited = []
    passed_count = 0
    failed_count = 0
    flagged_files = []
    
    # Find all CSV files in subdirectories
    csv_files = sorted(output_path.rglob('*.csv'), reverse=True)[:50]  # Limit to recent 50
    
    logger.info({
        'level': 'INFO',
        'type': 'audit',
        'message': f'Starting header audit on {len(csv_files)} CSV files',
        'session_id': None,
        'output_dir': str(output_path),
        'threshold': threshold
    })
    
    for csv_file in csv_files:
        result = audit_csv_headers(str(csv_file), threshold=threshold, limit=limit)
        audited.append(result)
        
        if result.get('pass'):
            passed_count += 1
            logger.info({
                'level': 'INFO',
                'type': 'audit',
                'message': f'✓ PASS: {result["file"]}',
                'session_id': None,
                'confidence_scores': result['confidence_scores']
            })
        else:
            failed_count += 1
            flagged_files.append(result)
            logger.warning({
                'level': 'WARNING',
                'type': 'audit',
                'message': f'✗ FAIL: {result["file"]}',
                'session_id': None,
                'confidence_scores': result['confidence_scores'],
                'flagged_headers': result['flagged_headers']
            })
            
            # Append to flagged headers JSONL
            try:
                with open(FLAGGED_HEADERS_FILE, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({
                        'timestamp': result['timestamp'],
                        'file': result['file'],
                        'headers': result['headers'],
                        'confidence_scores': result['confidence_scores'],
                        'flagged_headers': result['flagged_headers'],
                        'row_count': result['row_count'],
                        'full_path': str(csv_file)
                    }) + '\n')
            except Exception as e:
                logger.error({
                    'level': 'ERROR',
                    'type': 'audit',
                    'message': f'Failed to write flagged header: {e}',
                    'session_id': None
                })
    
    # Write audit report
    report = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'threshold': threshold,
        'total_audited': len(audited),
        'passed': passed_count,
        'failed': failed_count,
        'pass_rate': f'{100.0 * passed_count / len(audited):.1f}%' if audited else 'N/A',
        'flagged_files': [
            {
                'file': f['file'],
                'confidence_scores': f['confidence_scores'],
                'flagged_headers': f['flagged_headers']
            }
            for f in flagged_files
        ]
    }
    
    try:
        with open(AUDIT_REPORT_FILE, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
    except Exception as e:
        logger.error({
            'level': 'ERROR',
            'type': 'audit',
            'message': f'Failed to write audit report: {e}',
            'session_id': None
        })
    
    logger.info({
        'level': 'INFO',
        'type': 'audit',
        'message': f'Header audit complete: {passed_count} passed, {failed_count} failed',
        'session_id': None,
        'pass_rate': report['pass_rate'],
        'flagged_files_count': len(flagged_files),
        'audit_report': AUDIT_REPORT_FILE,
        'flagged_headers_log': FLAGGED_HEADERS_FILE
    })
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description='Audit header confidence in output CSVs before migration'
    )
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR),
                       help='Output directory to scan (default: OUTPUT_DIR)')
    parser.add_argument('--threshold', type=float, default=0.85,
                       help='Confidence threshold (0.0-1.0, default: 0.85)')
    parser.add_argument('--limit', type=int, default=500,
                       help='Max rows per CSV to scan (default: 500)')
    
    args = parser.parse_args()
    
    report = audit_output_directory(
        args.output_dir,
        threshold=args.threshold,
        limit=args.limit
    )
    
    # Print summary
    print('\n' + '='*60)
    print('HEADER CONFIDENCE AUDIT SUMMARY')
    print('='*60)
    print(f"Threshold: {args.threshold}")
    print(f"Total Audited: {report.get('total_audited', 0)}")
    print(f"Passed: {report.get('passed', 0)}")
    print(f"Failed: {report.get('failed', 0)}")
    print(f"Pass Rate: {report.get('pass_rate', 'N/A')}")
    
    if report.get('flagged_files'):
        print(f"\nFlagged Files ({len(report['flagged_files'])}):")
        for flagged in report['flagged_files']:
            print(f"\n  • {flagged['file']}")
            print(f"    Confidence: {flagged['confidence_scores']}")
            print(f"    Issues: {', '.join(flagged['flagged_headers'])}")
    
    print(f"\nAudit Report: {AUDIT_REPORT_FILE}")
    print(f"Flagged Headers Log: {FLAGGED_HEADERS_FILE}")
    print('='*60 + '\n')
    
    # Exit with appropriate code
    sys.exit(0 if report.get('failed', 0) == 0 else 1)


if __name__ == '__main__':
    main()

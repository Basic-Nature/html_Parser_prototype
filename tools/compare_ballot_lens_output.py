#!/usr/bin/env python3
"""
Ballot Lens Output → Database Comparison Utility
================================================

Compares CSV election data from ballot lens parsing with existing
database records to validate data correctness and identify discrepancies.

Usage:
    python tools/compare_ballot_lens_output.py \
      --csv output/election_results.csv \
      --state CA \
      --county "Alameda" \
      --election-date 2024-11-05 \
      --election-type general \
      [--database-connection postgresql://...]
      
    python tools/compare_ballot_lens_output.py \
      --csv output/results.csv \
      --query-file query_election_from_db.sql \
      --report-format json \
      --output-report comparison_report.json
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class CandidateRecord:
    """Single candidate/party/votes record."""
    office: str
    candidate: str
    party: str
    votes: int
    
    @classmethod
    def from_csv_row(cls, row: Dict[str, str]) -> 'CandidateRecord':
        """Create from CSV row dict."""
        try:
            votes = int(row.get('Votes', 0).replace(',', ''))
        except (ValueError, AttributeError):
            votes = 0
        
        return cls(
            office=str(row.get('Office', '')).strip(),
            candidate=str(row.get('Candidate', '')).strip(),
            party=str(row.get('Party', '')).strip(),
            votes=votes
        )
    
    def normalize(self) -> 'CandidateRecord':
        """Return normalized version for comparison."""
        return CandidateRecord(
            office=normalize_text(self.office),
            candidate=normalize_text(self.candidate),
            party=normalize_party_code(self.party),
            votes=self.votes
        )


@dataclass
class ElectionDataset:
    """Complete election results dataset."""
    source: str  # 'csv' or 'database'
    state: str
    county: str
    election_date: str
    election_type: str
    records: List[CandidateRecord] = field(default_factory=list)
    
    @property
    def row_count(self) -> int:
        return len(self.records)
    
    @property
    def total_votes(self) -> int:
        return sum(r.votes for r in self.records)
    
    @property
    def unique_candidates(self) -> Set[str]:
        return {r.candidate for r in self.records}
    
    @property
    def unique_parties(self) -> Set[str]:
        return {r.party for r in self.records}
    
    @property
    def unique_offices(self) -> Set[str]:
        return {r.office for r in self.records}
    
    def content_hash(self) -> str:
        """Create hash of dataset content for comparison."""
        content = json.dumps(
            sorted([asdict(r) for r in self.records]),
            sort_keys=True
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class Discrepancy:
    """Single point of difference."""
    type: str  # 'missing', 'extra', 'mismatch', 'count'
    field: str
    csv_value: Optional[str] = None
    db_value: Optional[str] = None
    context: Optional[Dict] = None
    
    @property
    def severity(self) -> str:
        """Assess severity of discrepancy."""
        if self.type == 'missing' or self.type == 'extra':
            return 'high'  # Missing/extra candidates is significant
        elif self.type == 'mismatch' and self.field == 'Votes':
            return 'high'  # Vote count mismatch is critical
        elif self.type == 'mismatch' and self.field in ['Candidate', 'Party']:
            return 'medium'  # Name/party mismatch needs review
        else:
            return 'low'


@dataclass
class ComparisonReport:
    """Complete comparison report."""
    timestamp: str
    election_key: Dict[str, str]
    csv_dataset: ElectionDataset
    db_dataset: ElectionDataset
    discrepancies: List[Discrepancy] = field(default_factory=list)
    
    @property
    def is_consistent(self) -> bool:
        """True if no significant discrepancies found."""
        high_severity = [d for d in self.discrepancies if d.severity == 'high']
        return len(high_severity) == 0
    
    @property
    def summary(self) -> Dict:
        """Generate summary statistics."""
        return {
            'timestamp': self.timestamp,
            'election': self.election_key,
            'csv_rows': self.csv_dataset.row_count,
            'db_rows': self.db_dataset.row_count,
            'row_count_match': self.csv_dataset.row_count == self.db_dataset.row_count,
            'csv_total_votes': self.csv_dataset.total_votes,
            'db_total_votes': self.db_dataset.total_votes,
            'votes_match': self.csv_dataset.total_votes == self.db_dataset.total_votes,
            'discrepancies_count': len(self.discrepancies),
            'high_severity': len([d for d in self.discrepancies if d.severity == 'high']),
            'is_consistent': self.is_consistent
        }


# ============================================================================
# Utility Functions
# ============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.strip()).lower()


def normalize_party_code(party: str) -> str:
    """Normalize party code/name for comparison."""
    if not party:
        return ""
    
    party = normalize_text(party)
    
    # Map common variants
    mappings = {
        'democratic party': 'dem',
        'democrat': 'dem',
        'dem': 'dem',
        'republican party': 'rep',
        'republican': 'rep',
        'rep': 'rep',
        'independent': 'ind',
        'ind': 'ind',
        'american independent party': 'ind',
        'green party': 'grn',
        'green': 'grn',
        'grn': 'grn',
        'libertarian party': 'lib',
        'libertarian': 'lib',
        'lib': 'lib',
    }
    
    return mappings.get(party, party)


def load_csv_dataset(
    csv_path: Path,
    state: str,
    county: str,
    election_date: str,
    election_type: str
) -> ElectionDataset:
    """Load election data from CSV file."""
    records = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    record = CandidateRecord.from_csv_row(row)
                    records.append(record)
                except Exception as e:
                    print(f"Warning: Skipping row due to parse error: {e}", file=sys.stderr)
                    continue
    
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")
    
    return ElectionDataset(
        source='csv',
        state=state,
        county=county,
        election_date=election_date,
        election_type=election_type,
        records=records
    )


def mock_database_dataset(
    state: str,
    county: str,
    election_date: str,
    election_type: str
) -> Optional[ElectionDataset]:
    """
    Mock database query for demonstration.
    
    In real usage, this would query actual database.
    """
    # For demo, create a sample dataset
    # In production, this would be:
    # conn = psycopg2.connect(database_url)
    # cursor = conn.cursor()
    # cursor.execute(query, (state, county, election_date, election_type))
    # rows = cursor.fetchall()
    
    records = [
        CandidateRecord('President', 'Alice Johnson', 'Democratic Party', 45230),
        CandidateRecord('President', 'Bob Smith', 'Republican Party', 38920),
        CandidateRecord('Governor', 'Carol White', 'Democratic Party', 42100),
        CandidateRecord('Governor', 'David Brown', 'Republican Party', 39800),
    ]
    
    return ElectionDataset(
        source='database',
        state=state,
        county=county,
        election_date=election_date,
        election_type=election_type,
        records=records
    )


def compare_datasets(
    csv_dataset: ElectionDataset,
    db_dataset: ElectionDataset
) -> List[Discrepancy]:
    """Compare CSV and database datasets, return list of discrepancies."""
    discrepancies = []
    
    # Normalize for comparison
    csv_normalized = [r.normalize() for r in csv_dataset.records]
    db_normalized = [r.normalize() for r in db_dataset.records]
    
    # Create lookup dicts
    csv_lookup = {(r.office, r.candidate): r for r in csv_normalized}
    db_lookup = {(r.office, r.candidate): r for r in db_normalized}
    
    # Check row count
    if len(csv_dataset.records) != len(db_dataset.records):
        discrepancies.append(Discrepancy(
            type='count',
            field='row_count',
            csv_value=str(len(csv_dataset.records)),
            db_value=str(len(db_dataset.records))
        ))
    
    # Check for missing records in CSV
    for key, db_record in db_lookup.items():
        if key not in csv_lookup:
            discrepancies.append(Discrepancy(
                type='missing',
                field='candidate',
                csv_value=None,
                db_value=f"{key[1]} ({key[0]})",
                context={'office': key[0], 'candidate': key[1]}
            ))
    
    # Check for extra records in CSV
    for key, csv_record in csv_lookup.items():
        if key not in db_lookup:
            discrepancies.append(Discrepancy(
                type='extra',
                field='candidate',
                csv_value=f"{key[1]} ({key[0]})",
                db_value=None,
                context={'office': key[0], 'candidate': key[1]}
            ))
    
    # Check for mismatches in matching records
    for key, csv_record in csv_lookup.items():
        if key in db_lookup:
            db_record = db_lookup[key]
            
            # Check votes
            if csv_record.votes != db_record.votes:
                discrepancies.append(Discrepancy(
                    type='mismatch',
                    field='Votes',
                    csv_value=str(csv_record.votes),
                    db_value=str(db_record.votes),
                    context={'office': key[0], 'candidate': key[1]}
                ))
            
            # Check party
            if csv_record.party != db_record.party:
                discrepancies.append(Discrepancy(
                    type='mismatch',
                    field='Party',
                    csv_value=csv_record.party,
                    db_value=db_record.party,
                    context={'office': key[0], 'candidate': key[1]}
                ))
    
    return discrepancies


def generate_html_report(report: ComparisonReport) -> str:
    """Generate HTML report of comparison."""
    summary = report.summary
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Ballot Lens Comparison Report</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 4px; }}
        .consistent {{ color: green; }}
        .inconsistent {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        .high {{ background: #ffe6e6; }}
        .medium {{ background: #fff3cd; }}
        .low {{ background: #e6f3ff; }}
    </style>
</head>
<body>
    <h1>Ballot Lens → Database Comparison Report</h1>
    <p>Generated: {summary['timestamp']}</p>
    
    <h2>Election</h2>
    <div class="summary">
        State: {summary['election']['state']} | 
        County: {summary['election']['county']} | 
        Date: {summary['election']['election_date']}
    </div>
    
    <h2>Summary</h2>
    <div class="summary">
        <p>Status: <span class="{'consistent' if summary['is_consistent'] else 'inconsistent'}">
            {'CONSISTENT' if summary['is_consistent'] else 'INCONSISTENT'}</span>
        </p>
        <p>Rows - CSV: {summary['csv_rows']}, DB: {summary['db_rows']} 
            ({('MATCH' if summary['row_count_match'] else 'MISMATCH')})</p>
        <p>Votes - CSV: {summary['csv_total_votes']:,}, DB: {summary['db_total_votes']:,} 
            ({('MATCH' if summary['votes_match'] else 'MISMATCH')})</p>
        <p>Discrepancies: {summary['discrepancies_count']} 
            (High: {summary['high_severity']})</p>
    </div>
"""
    
    if report.discrepancies:
        html += """
    <h2>Discrepancies</h2>
    <table>
        <tr>
            <th>Type</th>
            <th>Field</th>
            <th>CSV Value</th>
            <th>DB Value</th>
            <th>Severity</th>
        </tr>
"""
        for disc in report.discrepancies:
            html += f"""        <tr class="{disc.severity}">
            <td>{disc.type}</td>
            <td>{disc.field}</td>
            <td>{disc.csv_value or '—'}</td>
            <td>{disc.db_value or '—'}</td>
            <td>{disc.severity}</td>
        </tr>
"""
        html += """    </table>"""
    
    html += """
</body>
</html>
"""
    return html


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare ballot lens CSV output with database election data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--csv',
        type=Path,
        required=True,
        help='Path to CSV output from ballot lens'
    )
    parser.add_argument(
        '--state',
        required=True,
        help='State code (e.g., CA, GA, NV)'
    )
    parser.add_argument(
        '--county',
        required=True,
        help='County name'
    )
    parser.add_argument(
        '--election-date',
        required=True,
        help='Election date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--election-type',
        default='general',
        help='Election type (general, primary, special)'
    )
    parser.add_argument(
        '--database-connection',
        help='Database connection string (optional, uses mock if not provided)'
    )
    parser.add_argument(
        '--output-report',
        type=Path,
        help='Save HTML report to this file'
    )
    parser.add_argument(
        '--output-json',
        type=Path,
        help='Save JSON results to this file'
    )
    
    args = parser.parse_args()
    
    # Load CSV
    print(f"Loading CSV from {args.csv}...")
    try:
        csv_dataset = load_csv_dataset(
            args.csv,
            args.state,
            args.county,
            args.election_date,
            args.election_type
        )
        print(f"  - Loaded {csv_dataset.row_count} records")
        print(f"  - Total votes: {csv_dataset.total_votes:,}")
        print(f"  - Candidates: {len(csv_dataset.unique_candidates)}")
    except Exception as e:
        print(f"ERROR: Failed to load CSV: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Load from database (or mock)
    print(f"\nQuerying database for {args.state}/{args.county} on {args.election_date}...")
    db_dataset = mock_database_dataset(
        args.state,
        args.county,
        args.election_date,
        args.election_type
    )
    
    if db_dataset:
        print(f"  - Found {db_dataset.row_count} records")
        print(f"  - Total votes: {db_dataset.total_votes:,}")
    else:
        print("  - No database records found")
        sys.exit(0)
    
    # Compare
    print("\nComparing datasets...")
    discrepancies = compare_datasets(csv_dataset, db_dataset)
    
    report = ComparisonReport(
        timestamp=datetime.now().isoformat(),
        election_key={
            'state': args.state,
            'county': args.county,
            'date': args.election_date,
            'type': args.election_type
        },
        csv_dataset=csv_dataset,
        db_dataset=db_dataset,
        discrepancies=discrepancies
    )
    
    # Print summary
    summary = report.summary
    print("\n" + "="*60)
    print(f"Status: {'CONSISTENT' if report.is_consistent else 'INCONSISTENT'}")
    print(f"CSV Rows:      {summary['csv_rows']}")
    print(f"Database Rows: {summary['db_rows']}")
    print(f"Match:         {summary['row_count_match']}")
    print(f"CSV Votes:     {summary['csv_total_votes']:,}")
    print(f"DB Votes:      {summary['db_total_votes']:,}")
    print(f"Match:         {summary['votes_match']}")
    print(f"Discrepancies: {summary['discrepancies_count']}")
    print(f"  - High:      {summary['high_severity']}")
    print("="*60)
    
    # Output reports
    if args.output_json:
        print(f"\nSaving JSON report to {args.output_json}...")
        with open(args.output_json, 'w') as f:
            json.dump({
                'summary': summary,
                'discrepancies': [asdict(d) for d in report.discrepancies]
            }, f, indent=2)
    
    if args.output_report:
        print(f"Saving HTML report to {args.output_report}...")
        with open(args.output_report, 'w') as f:
            f.write(generate_html_report(report))
    
    # Exit with appropriate code
    sys.exit(0 if report.is_consistent else 1)


if __name__ == '__main__':
    main()

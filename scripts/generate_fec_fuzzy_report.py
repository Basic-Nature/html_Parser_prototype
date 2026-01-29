from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import sys
import argparse
import csv as _csv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from webapp.parser.fec_lookup import load_fec_candidates, get_candidate_by_id, find_candidate_by_name
from webapp.parser.config import MIN_FUZZY_SCORE_MANUAL, FUZZY_SCORER
from html import escape as _html_escape


FIXTURES_DIR = Path(__file__).resolve().parents[1] / 'webapp' / 'parser' / 'fixtures'
OUT_PATH = FIXTURES_DIR / 'fuzzy_match_report.jsonl'


def _normalize_local(name: str) -> str:
    if not name:
        return ""
    s = str(name).strip()
    s = s.replace('"', '')
    if "," in s:
        parts = [p.strip() for p in s.split(',') if p.strip()]
        if len(parts) >= 2:
            s = f"{parts[1]} {parts[0]}"
    s = " ".join(s.split()).lower()
    return s


def scan_fixture(path: Path, scorer: str = "auto", top_k: int = 1, cutoff: int | None = None) -> list[dict[str, Any]]:
    results = []
    with path.open('r', encoding='utf-8', errors='replace') as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader, start=1):
            # prefer common header names used in fixtures
            cand_id = (row.get('Cand_Id') or row.get('Cand_ID') or row.get('CandId') or row.get('candidate_id') or '').strip()
            cand_name = (row.get('Cand_Name') or row.get('CandName') or row.get('candidate_name') or '').strip()
            state = (row.get('Cand_Office_St') or row.get('Cand_State') or row.get('state') or '').strip()
            exact = None
            if cand_id:
                exact = get_candidate_by_id(cand_id)
            if exact:
                results.append({'file': path.name, 'row': i, 'cand_id': cand_id, 'cand_name': cand_name, 'state': state, 'normalized_name': _normalize_local(cand_name), 'match_type': 'exact', 'score': 100, 'matched_id': cand_id, 'matched_name': exact.get('Cand_Name'), 'method': 'id', 'candidates': [{'cand_id': cand_id, 'score': 100}], 'orig': dict(row)})
                continue
            # try fuzzy name match
            match = None
            if cand_name:
                match = find_candidate_by_name(cand_name, state=state, cutoff=cutoff, scorer=scorer, top_k=top_k)
            if match:
                # assemble candidates list
                cands = match.get('candidates') if isinstance(match.get('candidates'), list) else ([{"cand_id": match.get('cand_id'), "score": match.get('score')}] if match.get('cand_id') else [])
                results.append({'file': path.name, 'row': i, 'cand_id': cand_id or None, 'cand_name': cand_name, 'state': state, 'normalized_name': _normalize_local(cand_name), 'match_type': 'fuzzy', 'score': match.get('score'), 'matched_id': match.get('cand_id'), 'matched_name': (match.get('record') or {}).get('Cand_Name'), 'method': match.get('method'), 'candidates': cands, 'orig': dict(row)})
            else:
                results.append({'file': path.name, 'row': i, 'cand_id': cand_id or None, 'cand_name': cand_name, 'state': state, 'normalized_name': _normalize_local(cand_name), 'match_type': 'none', 'score': 0, 'matched_id': None, 'matched_name': None, 'method': None, 'candidates': [], 'orig': dict(row)})
    return results


def main():
    load_fec_candidates()  # warm cache
    files = sorted([p for p in FIXTURES_DIR.glob('candidate_summary_*.csv') if p.is_file()])
    parser = argparse.ArgumentParser(description='Generate FEC fuzzy-match report from fixtures')
    parser.add_argument('--only-fuzzy', action='store_true', help='Print only fuzzy matches (JSON)')
    parser.add_argument('--out-csv', help='Write concise CSV of problem rows (non-exact or low-confidence)')
    parser.add_argument('--out-jsonl', help='Write full JSONL of results to this path (overrides default)')
    parser.add_argument('--html-report', help='Write a simple HTML report for quick triage')
    parser.add_argument('--min-score', type=int, default=MIN_FUZZY_SCORE_MANUAL, help=f'Minimum score considered high-confidence (default: {MIN_FUZZY_SCORE_MANUAL})')
    parser.add_argument('--include-context', action='store_true', help='Include original row context columns in concise CSV output')
    parser.add_argument('--scorer', choices=['auto', 'rapidfuzz', 'difflib'], default=FUZZY_SCORER, help=f'Choose fuzzy backend (auto, rapidfuzz, difflib) (default: {FUZZY_SCORER})')
    parser.add_argument('--top-k', type=int, default=3, help='Return top-K nearest candidates for context (default: 3)')
    args = parser.parse_args()
    if not files:
        print('No candidate_summary_*.csv fixtures found in', FIXTURES_DIR)
        return
    # remove previous report
    try:
        if OUT_PATH.exists():
            OUT_PATH.unlink()
    except Exception:
        pass

    aggregated = []
    for f in files:
        print('Scanning', f.name)
        rows = scan_fixture(f, scorer=args.scorer, top_k=args.top_k, cutoff=args.min_score)
        aggregated.extend(rows)

    # write JSONL (default or overridden)
    jsonl_path = Path(args.out_jsonl) if args.out_jsonl else OUT_PATH
    with jsonl_path.open('w', encoding='utf-8') as outfh:
        for rec in aggregated:
            outfh.write(json.dumps(rec) + '\n')

    # Summarize
    total = len(aggregated)
    exact = sum(1 for r in aggregated if r['match_type'] == 'exact')
    fuzzy_high = sum(1 for r in aggregated if r['match_type'] == 'fuzzy' and (r.get('score') or 0) >= args.min_score)
    fuzzy_low = sum(1 for r in aggregated if r['match_type'] == 'fuzzy' and 0 < (r.get('score') or 0) < args.min_score)
    none = sum(1 for r in aggregated if r['match_type'] == 'none')

    print('\nFuzzy match report written to', jsonl_path)
    print(f'Total rows: {total}  exact: {exact}  fuzzy>={args.min_score}: {fuzzy_high}  fuzzy<{args.min_score}: {fuzzy_low}  none: {none}')

    # print a few low-confidence samples
    print('\nLow-confidence fuzzy matches (score < {min_score}) — sample 10:'.format(min_score=args.min_score))
    low_samples = [r for r in aggregated if r['match_type'] == 'fuzzy' and (r.get('score') or 0) < args.min_score]
    for r in low_samples[:10]:
        print(json.dumps(r, ensure_ascii=False))
    # If user asked for only fuzzy, stream fuzzy results
    if args.only_fuzzy:
        for r in aggregated:
            if r['match_type'] == 'fuzzy':
                print(json.dumps(r, ensure_ascii=False))
        return

    # If user requested a concise CSV, write problem rows
    if args.out_csv:
        out_csv_path = Path(args.out_csv)
        with out_csv_path.open('w', encoding='utf-8', newline='') as cf:
            # Determine context columns union if requested
            context_keys = []
            if args.include_context:
                keyset = set()
                for r in aggregated:
                    orig = r.get('orig') or {}
                    keyset.update(orig.keys())
                context_keys = sorted(keyset)

            headers = ['file', 'row', 'cand_id', 'cand_name', 'normalized', 'state', 'match_type', 'score', 'matched_id', 'matched_name', 'nearest']
            if context_keys:
                headers += [f'ctx::{k}' for k in context_keys]
            writer = _csv.writer(cf)
            writer.writerow(headers)
            for r in aggregated:
                is_problem = (r['match_type'] != 'exact') or (r['match_type'] == 'fuzzy' and (r.get('score') or 0) < args.min_score)
                if is_problem:
                    # nearest candidates as semicolon-separated id|score|name
                    neigh = ''
                    try:
                        neigh_list = r.get('candidates') or []
                        neigh = ';'.join([f"{c.get('cand_id') or ''}|{c.get('score') or 0}|{(c.get('record') or {}).get('Cand_Name') or ''}" for c in neigh_list])
                    except Exception:
                        neigh = ''
                    base = [r['file'], r['row'], r.get('cand_id'), r.get('cand_name'), r.get('normalized_name'), r.get('state'), r['match_type'], r.get('score'), r.get('matched_id'), r.get('matched_name'), neigh]
                    if context_keys:
                        orig = r.get('orig') or {}
                        ctx_vals = [orig.get(k) for k in context_keys]
                        writer.writerow(base + ctx_vals)
                    else:
                        writer.writerow(base)
        print('Wrote concise CSV of problem rows to', out_csv_path)
        return

    # print a few low-confidence samples
    print(f'\nLow-confidence fuzzy matches (score < {args.min_score}) — sample 10:')
    low_samples = [r for r in aggregated if r['match_type'] == 'fuzzy' and (r.get('score') or 0) < args.min_score]
    for r in low_samples[:10]:
        print(json.dumps(r, ensure_ascii=False))

    # Optionally write a simple HTML report for triage
    if args.html_report:
        html_path = Path(args.html_report)
        print('Writing HTML report to', html_path)
        try:
            with html_path.open('w', encoding='utf-8') as hf:
                hf.write('<html><head><meta charset="utf-8"><title>FEC Fuzzy Report</title>')
                hf.write('<style>body{font-family:Inter,Arial,Helvetica;margin:12px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:6px;font-size:12px}thead th{position:sticky;top:0;background:#f9fafb}</style>')
                hf.write('</head><body>')
                hf.write(f'<h2>FEC Fuzzy Report ({len(aggregated)} rows)</h2>')
                hf.write('<table><thead><tr>')
                cols = ['file','row','cand_id','cand_name','normalized','state','match_type','score','matched_id','matched_name','nearest']
                for c in cols:
                    hf.write(f'<th>{_html_escape(c)}</th>')
                hf.write('</tr></thead><tbody>')
                for r in aggregated:
                    neigh_list = r.get('candidates') or []
                    neigh_html = '<br>'.join([_html_escape(f"{c.get('cand_id') or ''} ({c.get('score')}) - {(c.get('record') or {}).get('Cand_Name') or ''}") for c in neigh_list])
                    hf.write('<tr>')
                    hf.write(f"<td>{_html_escape(str(r.get('file','')))}</td>")
                    hf.write(f"<td>{r.get('row','')}</td>")
                    hf.write(f"<td>{_html_escape(str(r.get('cand_id') or ''))}</td>")
                    hf.write(f"<td>{_html_escape(str(r.get('cand_name') or ''))}</td>")
                    hf.write(f"<td>{_html_escape(str(r.get('normalized_name') or ''))}</td>")
                    hf.write(f"<td>{_html_escape(str(r.get('state') or ''))}</td>")
                    hf.write(f"<td>{_html_escape(str(r.get('match_type') or ''))}</td>")
                    hf.write(f"<td>{_html_escape(str(r.get('score') or ''))}</td>")
                    hf.write(f"<td>{_html_escape(str(r.get('matched_id') or ''))}</td>")
                    hf.write(f"<td>{_html_escape(str(r.get('matched_name') or ''))}</td>")
                    hf.write(f"<td>{neigh_html}</td>")
                    hf.write('</tr>')
                hf.write('</tbody></table></body></html>')
            print('HTML report written to', html_path)
        except Exception as e:
            print('Failed to write HTML report:', e)


if __name__ == '__main__':
    main()

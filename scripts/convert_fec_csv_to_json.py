#!/usr/bin/env python3
"""Convert FEC candidate CSV(s) in fixtures to a JSON index keyed by candidate id.

Usage:
  python scripts/convert_fec_csv_to_json.py --src webapp/parser/fixtures --out webapp/parser/fixtures/candidate_summary_index.json

By default it will look for .csv/.xlsx/.xls files in the source dir and produce a single JSON mapping.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict

try:
    import pandas as pd
except Exception:
    pd = None


def read_csv(path: str):
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            yield {k: (v if v is not None else '') for k, v in r.items()}


def read_excel(path: str):
    if pd is None:
        raise RuntimeError('pandas required for Excel support')
    df = pd.read_excel(path, sheet_name=0, dtype=str)
    df = df.fillna('')
    for r in df.to_dict(orient='records'):
        yield {k: (v if v is not None else '') for k, v in r.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    src = os.path.abspath(args.src)
    out = os.path.abspath(args.out)
    files = [os.path.join(src, f) for f in os.listdir(src) if os.path.splitext(f)[1].lower() in ('.csv', '.xlsx', '.xls')]
    index: Dict[str, Dict[str, Any]] = {}
    for fpath in files:
        ext = os.path.splitext(fpath)[1].lower()
        if ext in ('.xlsx', '.xls'):
            if pd is None:
                print('Skipping Excel (pandas missing):', fpath)
                continue
            reader = read_excel(fpath)
        else:
            reader = read_csv(fpath)
        for row in reader:
            # prefer common FEC id column names
            cand_id = row.get('Cand_Id') or row.get('CAND_ID') or row.get('cand_id') or row.get('candidate_id') or ''
            cand_id = str(cand_id).strip()
            if not cand_id:
                # try to synthesize id from name+office if missing
                name = (row.get('Cand_Name') or row.get('candidate_name') or '')
                office = (row.get('Cand_Office') or row.get('Cand_Office_St') or '')
                if name:
                    cand_id = f"anon::{name.strip()}::{office.strip()}"
            if not cand_id:
                continue
            index[cand_id] = row

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(index, fh, indent=2, ensure_ascii=False)
    print('Wrote', out, 'with', len(index), 'entries')


if __name__ == '__main__':
    main()

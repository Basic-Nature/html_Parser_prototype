# election_pipeline_monolith.py
# SINGLE-FILE EDITION: Self-contained, secure, and portable

import os
import re
import csv
import sys
import json
import logging
import argparse
from io import StringIO
from datetime import datetime
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ================================
# SECTION 0 — GLOBAL SETUP
# ================================
OUTPUT_DIR = r"C:\ElectionDataOutput"
SCHEMA_FILE = os.path.join(OUTPUT_DIR, "schema.json")
COUNTIES_FILE = os.path.join(OUTPUT_DIR, "known_counties.txt")
STRESS_DIR = os.path.join(OUTPUT_DIR, "stress_inputs")

if os.path.abspath(OUTPUT_DIR) == os.path.expanduser("~"):
    raise ValueError("SECURITY ERROR: Output directory must not be your home directory itself. Please set OUTPUT_DIR to a subfolder or another location.")
if os.path.abspath(OUTPUT_DIR).startswith(os.path.expanduser("~")):
    print("WARNING: Output is inside your home directory. Proceeding anyway.")
  
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STRESS_DIR, exist_ok=True)
# Set up logging
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "log.txt"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# ================================
# SECTION 1 — UTILITIES
# ================================
def safe_convert(df, col):
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    if df[col].isnull().any():
        raise ValueError(f"Non-numeric or missing values in: {col}")
    return df

def normalize_county(df):
    df['County'] = df['County'].astype(str).str.strip().str.title()
    return df

def print_summary(df):
    print("\n✅ Summary:")
    print(f"Counties: {len(df)}")
    for col in df.columns:
        if 'Votes' in col:
            print(f"{col}: {df[col].sum():,}")

# ================================
# SECTION 2 — FORMAT CLEANERS
# ================================
def process_format_1(df):
    df = normalize_county(df)
    for col in ['REP', 'DEM']:
        df = safe_convert(df, col)
    return df[['County', 'REP', 'DEM']].rename(columns={'REP': 'REP Total Votes', 'DEM': 'DEM Total Votes'})

def process_format_2(df):
    df = df.rename(columns={df.columns[0]: 'County'})
    df = normalize_county(df)
    df = pd.DataFrame({
        'County': df['County'],
        'DEM Total Votes': df[('Biden', 'Total')],
        'REP Total Votes': df[('Trump', 'Total')]
    })
    for col in ['REP Total Votes', 'DEM Total Votes']:
        df = safe_convert(df, col)
    return df

def process_format_3(df):
    df = normalize_county(df)
    for col in ['Trump', 'Harris', 'Other']:
        if col in df.columns:
            df = safe_convert(df, col)
    return df[['County', 'Trump', 'Harris', 'Other']].rename(columns={
        'Trump': 'REP Total Votes',
        'Harris': 'DEM Total Votes',
        'Other': 'Other Votes'
    })
def process_format_precinct_totals(df):
    # Find the "Precinct" column
    if 'Precinct' not in df.columns:
        raise ValueError("No 'Precinct' column found")
    # All other columns are candidate totals
    candidate_cols = [col for col in df.columns if col != 'Precinct' and col.endswith('- Total')]
    # Melt the DataFrame to long format
    melted = df.melt(id_vars=['Precinct'], value_vars=candidate_cols,
                     var_name='Candidate', value_name='Votes')
    # Clean up
    melted['Votes'] = pd.to_numeric(melted['Votes'].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)
    return melted, 'precinct_totals'
# ================================
# SECTION 3 — FORMAT DETECTION
# ================================
def detect_and_process(raw, delimiter='\t'):
    print("DEBUG: Raw input preview:\n", raw[:500])
    try:
        df = pd.read_csv(StringIO(raw), sep=delimiter, header=[0, 1])
        print("DEBUG: Multi-index columns:", df.columns.tolist())
        if ('Biden', 'Total') in df.columns:
            return process_format_2(df), 'format_2'
    except Exception as e:
        print("DEBUG: Format 2 failed:", e)

    try:
        df = pd.read_csv(StringIO(raw), sep=delimiter)
        print("DEBUG: Columns:", df.columns.tolist())
        if {'County', 'REP', 'DEM'}.issubset(df.columns):
            return process_format_1(df), 'format_1'
    except Exception as e:
        print("DEBUG: Format 1 failed:", e)

    try:
        df = pd.read_csv(StringIO(raw), sep=delimiter)
        print("DEBUG: Columns:", df.columns.tolist())
        if {'County', 'Trump', 'Harris'}.issubset(df.columns):
            return process_format_3(df), 'format_3'
    except Exception as e:
        print("DEBUG: Format 3 failed:", e)
    try:
        df = pd.read_csv(StringIO(raw), sep=delimiter)
        print("DEBUG: Columns:", df.columns.tolist())
        if 'Precinct' in df.columns:
            return process_format_precinct_totals(df)
    except Exception as e:
        print("DEBUG: Precinct totals format failed:", e)
    raise ValueError("Unrecognized format")

# ================================
# SECTION 4 — INGESTION / FETCH
# ================================
def ingest_url(url):
    logging.info(f"Fetching: {url}")
    r = requests.get(url)
    content = r.text
    if '<table' in content:
        soup = BeautifulSoup(content, 'html.parser')
        table = soup.find('table')
        rows = ["\t".join([cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])])
                for row in table.find_all('tr')]
        return "\n".join(rows)
    return content

# ================================
# SECTION 5 — VALIDATOR
# ================================
def validate_output(df):
    if df.isnull().sum().sum() > 0:
        raise ValueError("Null values detected")
    for col in df.columns:
        if 'Votes' in col and not pd.api.types.is_integer_dtype(df[col]):
            raise ValueError(f"Non-integer detected in: {col}")
    return df

# ================================
# SECTION 6 — STRESS FILES
# ================================
def write_stress_inputs():
    if not os.listdir(STRESS_DIR):
        samples = {
            'missing_column.csv': "County,REP\nAlpha,12345",
            'bad_delimiter.txt': "County|REP|DEM\nAlpha|12345|54321",
            'extra_whitespace.tsv': "\n\nCounty\tREP\tDEM\nAlpha\t12345\t54321\n\nBeta\t67890\t9876\n",
            'empty_file.csv': "",
            'malformed_header.csv': "Couny,Republicans,Democracts\nAlpha,12345,54321"
        }
        for name, content in samples.items():
            with open(os.path.join(STRESS_DIR, name), 'w') as f:
                f.write(content)
        logging.info("Stress inputs written")

# ================================
# SECTION 7 — CONTROLLER
# ================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Path to file')
    parser.add_argument('--url', help='URL to fetch')
    args = parser.parse_args()

    # Add this check here
    if args.input and not os.path.isfile(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        return

    with open(SCHEMA_FILE, 'w') as f:
        json.dump({"County": "str", "REP Total Votes": "int", "DEM Total Votes": "int"}, f)
    with open(COUNTIES_FILE, 'w') as f:
        f.write("Los Angeles\nSan Diego\nOrange")

    write_stress_inputs()

    if not args.input and not args.url:
        print("Provide --input or --url")
        return

    raw = ingest_url(args.url) if args.url else open(args.input).read()
    df, fmt = detect_and_process(raw)
    df = validate_output(df)
    print_summary(df)

    out_file = os.path.join(OUTPUT_DIR, f"cleaned_{fmt}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(out_file, index=False)
    print(f"\n✅ Output: {out_file}")
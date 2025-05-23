#Merry ChristmuhKwanzikah:

import pandas as pd
import csv
import re
from io import StringIO
from difflib import SequenceMatcher

COMMON_DELIMITERS = [',', '\t', ';', '|']
LIKELY_HEADERS = ['county', 'rep', 'dem', 'trump', 'biden', 'harris', 'total', 'votes', 'other']


def guess_delimiter(text):
    scores = {}
    for delim in COMMON_DELIMITERS:
        lines = text.strip().splitlines()[:10]
        fields = [len(line.split(delim)) for line in lines]
        scores[delim] = sum(fields)
    return max(scores, key=scores.get)


def clean_column_name(name):
    return re.sub(r'[^a-z0-9]', '', name.strip().lower())


def match_header_column(col_name):
    norm_col = clean_column_name(col_name)
    best = None
    best_score = 0
    for candidate in LIKELY_HEADERS:
        score = SequenceMatcher(None, norm_col, candidate).ratio()
        if score > best_score:
            best_score = score
            best = candidate
    return best if best_score > 0.6 else None


def interpret_structure(text):
    delimiter = guess_delimiter(text)
    try:
        df = pd.read_csv(StringIO(text), sep=delimiter, engine='python')
    except Exception as e:
        return None, f"Failed to parse with delimiter '{delimiter}': {e}"

    # Normalize and match headers
    column_mapping = {}
    for col in df.columns:
        mapped = match_header_column(col)
        if mapped:
            column_mapping[col] = mapped

    df.rename(columns=column_mapping, inplace=True)

    if 'county' not in df.columns or ('rep' not in df.columns and 'trump' not in df.columns):
        return None, "Unrecognized structure. Could not match essential headers."

    return df, None


if __name__ == "__main__":
    with open("test_input.txt", "r", encoding="utf-8") as f:
        content = f.read()
    df, err = interpret_structure(content)
    if err:
        print(f"❌ Error: {err}")
    else:
        print("✅ Parsed dataframe:")
        print(df.head())
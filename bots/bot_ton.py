# bot_ton.py
# ===============================================================================
# Election Data Cleaner
# This script cleans and validates messy election data from various formats.
# Here's my bot so far: by Tonnocus 
# ================================================================================
import os
import json
import logging
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from io import StringIO
from datetime import datetime

# === CONFIGURATION ===
def load_config(config_file='config.json'):
    with open(config_file, 'r') as f:
        return json.load(f)

# === LOGGING ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def log_error_to_file(message):
    error_log_path = os.path.expanduser("~/Election Data Output/errors.log")
    with open(error_log_path, 'a') as log:
        log.write(f"{datetime.now()}: {message}\n")

# === CLEANING HELPERS ===
def normalize_county(df):
    df['County'] = df['County'].astype(str).str.strip().str.title()
    return df

def safe_convert(df, col):
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    if df[col].isnull().any():
        raise ValueError(f"Non-numeric or missing values in column: {col}")
    return df

# === FORMAT HANDLERS ===
def process_format_1(df):
    logger.info(f"Processing {len(df)} counties in Format 1...")
    df = df[df['County'].notnull()]
    df = normalize_county(df)
    for col in ['REP', 'DEM']:
        df = safe_convert(df, col)
    return df[['County', 'REP', 'DEM']].rename(columns={'REP': 'REP Total Votes', 'DEM': 'DEM Total Votes'})

def process_format_2(df):
    logger.info(f"Processing {len(df)} counties in Format 2...")
    df = df.rename(columns={df.columns[0]: 'County'})
    df = normalize_county(df)
    output = pd.DataFrame({
        'County': df['County'],
        'DEM Total Votes': df[('Biden', 'Total')],
        'REP Total Votes': df[('Trump', 'Total')]
    })
    for col in ['DEM Total Votes', 'REP Total Votes']:
        output = safe_convert(output, col)
    return output

def process_format_3(df):
    logger.info(f"Processing {len(df)} counties in Format 3...")
    df = normalize_county(df)
    for col in ['Trump', 'Harris', 'Other']:
        if col in df.columns:
            df = safe_convert(df, col)
    return df[['County', 'Trump', 'Harris', 'Other']].rename(columns={
        'Trump': 'REP Total Votes',
        'Harris': 'DEM Total Votes',
        'Other': 'Other Votes'
    })

# === FORMAT DETECTOR ===
def detect_and_process(messy_data, delimiter):
    if not messy_data.strip():
        raise ValueError("Empty input data provided")

    try:
        df = pd.read_csv(StringIO(messy_data), sep=delimiter, header=[0, 1], engine='python')
        if ('Biden', 'Total') in df.columns and ('Trump', 'Total') in df.columns:
            return process_format_2(df), 'format_2'
    except Exception as e:
        logger.warning(f"Format 2 detection failed: {e}")

    try:
        df = pd.read_csv(StringIO(messy_data), sep=delimiter, engine='python')
        df.columns = df.columns.str.strip()
        if {'County', 'REP', 'DEM'}.issubset(df.columns):
            return process_format_1(df), 'format_1'
    except Exception as e:
        logger.warning(f"Format 1 detection failed: {e}")

    try:
        df = pd.read_csv(StringIO(messy_data), sep=delimiter, engine='python')
        df.columns = df.columns.str.strip()
        if {'County', 'Trump', 'Harris'}.issubset(df.columns):
            return process_format_3(df), 'format_3'
    except Exception as e:
        logger.warning(f"Format 3 detection failed: {e}")

    raise ValueError("Unknown data format. Please verify structure and delimiters.")

# === VALIDATOR ===
def validate_output(df):
    if df.isnull().sum().sum() > 0:
        raise ValueError("Validation failed: Null values detected")
    for col in df.columns:
        if 'Votes' in col and not pd.api.types.is_integer_dtype(df[col]):
            raise ValueError(f"Validation failed: {col} contains non-integer values")
    logger.info("\u2705 Data validation passed")
    return df

# === GUI HANDLER ===
def handle_drop():
    file_path = filedialog.askopenfilename(filetypes=[
        ("All supported", "*.tsv *.csv *.txt"),
        ("TSV files", "*.tsv"),
        ("CSV files", "*.csv"),
        ("Text files", "*.txt")
    ])
    if not file_path:
        return

    try:
        config = load_config()
        delimiter = config.get("delimiter", "\t")
        output_folder = os.path.expanduser(config.get("output_folder", "~/Election Data Output"))
        os.makedirs(output_folder, exist_ok=True)

        with open(file_path, 'r', encoding='utf-8') as f:
            messy_data = f.read()

        cleaned_df, fmt = detect_and_process(messy_data, delimiter)
        validate_output(cleaned_df)

        output_file = os.path.join(output_folder, f"cleaned_{fmt}_election_data.csv")
        cleaned_df.to_csv(output_file, index=False)

        messagebox.showinfo("Success", f"\u2705 Cleaned data saved to:\n{output_file}")
    except Exception as e:
        log_error_to_file(str(e))
        messagebox.showerror("Error", f"\u274C Cleaning failed:\n{str(e)}")

# === GUI LAUNCHER ===
def launch_gui():
    root = tk.Tk()
    root.title("Election Data Cleaner (Drop a File)")

    label = tk.Label(root, text="\U0001F4C2 Click to select a data file\n(Format will be auto-detected and tested first)",
                     font=("Arial", 14), padx=20, pady=20, relief=tk.RIDGE, borderwidth=2)
    label.pack(padx=30, pady=30)

    label.bind("<Button-1>", lambda e: handle_drop())
    root.mainloop()

# === MAIN CLI HANDLER ===
def main():
    parser = argparse.ArgumentParser(description="Clean and validate messy election data.")
    parser.add_argument('--input', type=str, help="Path to data file to process")
    parser.add_argument('--delimiter', type=str, help="Delimiter override (e.g., ',', '\\t')")
    parser.add_argument('--gui', action='store_true', help="Launch drag-and-drop interface")

    args = parser.parse_args()

    if args.gui:
        launch_gui()
        return

    if not args.input:
        logger.error("\u274C You must provide an input file with --input <path>")
        return

    try:
        config = load_config()
        output_folder = os.path.expanduser(config.get("output_folder", "~/Election Data Output"))
        delimiter = args.delimiter if args.delimiter else config.get("delimiter", "\t")
        os.makedirs(output_folder, exist_ok=True)

        with open(args.input, 'r', encoding='utf-8') as f:
            messy_data = f.read()

        cleaned_df, fmt = detect_and_process(messy_data, delimiter)
        validate_output(cleaned_df)

        output_path = os.path.join(output_folder, f"cleaned_{fmt}_election_data.csv")
        cleaned_df.to_csv(output_path, index=False)

        logger.info(f"\n\u2705 Exported to: {output_path}")
        logger.info(f"\U0001F50D Detected format: {fmt}\n")
        logger.info(f"\n{cleaned_df.head()}")

    except Exception as e:
        logger.error(f"\u274C Processing failed: {e}")
        log_error_to_file(str(e))

if __name__ == "__main__":
    main()
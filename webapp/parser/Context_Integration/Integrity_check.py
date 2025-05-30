import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
import matplotlib
# Use Agg backend for non-GUI environments # (e.g., servers, CI/CD pipelines)
matplotlib.use('Agg')
#/ comment out to see plots
import matplotlib.pyplot as plt
import threading
import json
import time
import sqlite3
from pathlib import Path
from ..utils.shared_logger import rprint
from typing import List, Dict, Any, Tuple, Optional
from ..utils.spacy_utils import extract_dates
from ..config import CONTEXT_DB_PATH, CONTEXT_LIBRARY_PATH
from ..utils import db_utils
# --- Rich imports for CLI output ---
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def _ensure_alerts_table(db_path=None):
    path = db_utils._safe_db_path(db_path or CONTEXT_DB_PATH)
    conn = sqlite3.connect(path)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        level TEXT,
        msg TEXT,
        context TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()
_ensure_alerts_table()

# --- Data Processing Functions (unchanged) ---

def find_date_anomalies(contests, expected_year=None):
    anomalies = []
    for c in contests:
        dates = extract_dates(c.get("title", ""))
        if expected_year and not any(str(expected_year) in d for d in dates):
            anomalies.append(c)
    return anomalies

def detect_anomalies_with_ml(
    contexts: List[Dict[str, Any]],
    contamination: float = 0.05,
    n_estimators: int = 100,
    random_state: int = 42
) -> Tuple[List[int], np.ndarray]:
    if not contexts:
        return [], np.array([])
    features = []
    le_state = LabelEncoder()
    le_county = LabelEncoder()
    states = [c.get("state", "unknown") for c in contexts]
    counties = [c.get("county", "unknown") for c in contexts]
    le_state.fit(states)
    le_county.fit(counties)
    for c in contexts:
        features.append([
            le_state.transform([c.get("state", "unknown")])[0],
            le_county.transform([c.get("county", "unknown")])[0],
            int(c.get("year", 0)) if str(c.get("year", "0")).isdigit() else 0,
            len(str(c.get("title", ""))),
        ])
    X = np.array(features)
    clf = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state
    )
    preds = clf.fit_predict(X)
    anomalies = [i for i, p in enumerate(preds) if p == -1]
    clustering = DBSCAN(eps=3, min_samples=2).fit(X)
    clusters = clustering.labels_
    return anomalies, clusters

feature_names = ["state", "county", "year", "title_length"]

def election_integrity_checks(contests: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    seen = set()
    issues = []
    for c in contests:
        key = (str(c.get("title")).strip().lower(), c.get("year"), c.get("state"), c.get("county"))
        if key in seen:
            issues.append(("duplicate", c))
        else:
            seen.add(key)
        if not c.get("county") or not c.get("state"):
            issues.append(("missing_location", c))
        if not c.get("year") or not str(c.get("year")).isdigit():
            issues.append(("missing_year", c))
    return issues

def advanced_cross_field_validation(contests: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    issues = []
    for c in contests:
        if c.get("type") == "Presidential" and c.get("state") not in ("us", "USA", "United States"):
            issues.append(("presidential_state_mismatch", c))
        if "votes" in c and isinstance(c["votes"], (int, float)) and c["votes"] < 0:
            issues.append(("negative_votes", c))
    return issues

def summarize_context_entities(contests):
    from collections import Counter
    from ..utils.spacy_utils import extract_entities
    entity_counter = Counter()
    for c in contests:
        title = c.get("title", "")
        entities = extract_entities(title)
        for _, label in entities:
            entity_counter[label] += 1
    return dict(entity_counter)

def analyze_contest_titles(contests, expected_year=None, context_library_path=None):
    from ..utils.spacy_utils import flag_suspicious_contests
    integrity_issues = election_integrity_checks(contests)
    date_anomalies = find_date_anomalies(contests, expected_year=expected_year)
    anomalies, clusters = detect_anomalies_with_ml(contests)
    if context_library_path is None:
        context_library_path = CONTEXT_LIBRARY_PATH
    flagged = flag_suspicious_contests(contests, context_library_path=context_library_path)
    return {
        "integrity_issues": integrity_issues,
        "date_anomalies": date_anomalies,
        "ml_anomalies": anomalies,
        "clusters": clusters.tolist() if hasattr(clusters, "tolist") else clusters,
        "flagged_suspicious": flagged,
    }

def auto_tune_contamination(
    X: np.ndarray,
    initial_contamination: float = 0.2,
    min_contamination: float = 0.01,
    max_contamination: float = 0.2,
    plot: bool = False
) -> float:
    clf = IsolationForest(contamination=initial_contamination, random_state=42)
    clf.fit(X)
    scores = -clf.decision_function(X)
    cutoff = np.percentile(scores, 90)
    n_anomalies = np.sum(scores >= cutoff)
    contamination = n_anomalies / len(scores)
    contamination = max(min_contamination, min(max_contamination, contamination))
    return contamination

# --- Rich Output Functions for CLI ---

def print_issues_table(issues, title="Issues"):
    if not issues:
        console.print(f"[bold green]No {title.lower()} found.[/bold green]")
        return
    table = Table(title=title, show_lines=True)
    table.add_column("Issue Type", style="red")
    table.add_column("Title", style="cyan")
    table.add_column("Year", style="green")
    table.add_column("State", style="yellow")
    table.add_column("County", style="blue")
    for issue_type, contest in issues:
        table.add_row(
            issue_type,
            contest.get("title", ""),
            str(contest.get("year", "")),
            contest.get("state", ""),
            contest.get("county", "")
        )
    console.print(table)

def print_entity_summary(entity_summary):
    table = Table(title="Entity Label Summary")
    table.add_column("Entity Label", style="cyan")
    table.add_column("Count", style="magenta")
    for label, count in entity_summary.items():
        table.add_row(label, str(count))
    console.print(table)

def print_ml_anomalies(anomaly_indices, contests):
    if not anomaly_indices:
        console.print("[bold green]No ML anomalies detected.[/bold green]")
        return
    table = Table(title="ML Detected Anomalies", show_lines=True)
    table.add_column("Index", style="magenta")
    table.add_column("Title", style="cyan")
    table.add_column("Year", style="green")
    table.add_column("State", style="yellow")
    table.add_column("County", style="blue")
    for idx in anomaly_indices:
        c = contests[idx]
        table.add_row(
            str(idx),
            c.get("title", ""),
            str(c.get("year", "")),
            c.get("state", ""),
            c.get("county", "")
        )
    console.print(table)

def print_date_anomalies(date_anomalies):
    if not date_anomalies:
        console.print("[bold green]No date anomalies found.[/bold green]")
        return
    table = Table(title="Date Anomalies", show_lines=True)
    table.add_column("Title", style="cyan")
    table.add_column("Year", style="green")
    table.add_column("State", style="yellow")
    table.add_column("County", style="blue")
    for contest in date_anomalies:
        table.add_row(
            contest.get("title", ""),
            str(contest.get("year", "")),
            contest.get("state", ""),
            contest.get("county", "")
        )
    console.print(table)

def print_auto_tune_result(contamination):
    console.print(Panel(f"Auto-tuned contamination: [bold green]{contamination:.4f}[/bold green]", title="IsolationForest Auto-Tune"))

def print_analyze_contest_titles(results):
    print_issues_table(results["integrity_issues"], title="Integrity Issues")
    print_date_anomalies(results["date_anomalies"])
    print_ml_anomalies(results["ml_anomalies"], results.get("contests", []))
    if results.get("flagged_suspicious"):
        console.print(Panel(f"[yellow]{len(results['flagged_suspicious'])} suspicious contests flagged[/yellow]: {results['flagged_suspicious']}", title="Suspicious Contests"))
    else:
        console.print("[bold green]No suspicious contests flagged.[/bold green]")

# --- Real-Time Monitoring (unchanged) ---

def monitor_db_for_alerts(db_path: str = None, poll_interval: int = 10):
    path = db_utils._safe_db_path(db_path or CONTEXT_DB_PATH)
    def monitor():
        last_alert_id = 0
        while True:
            try:
                conn = sqlite3.connect(path)
                cursor = conn.cursor()
                cursor.execute("SELECT id, level, msg, context, created_at FROM alerts WHERE id > ? ORDER BY id ASC", (last_alert_id,))
                rows = cursor.fetchall()
                for row in rows:
                    last_alert_id = row[0]
                    rprint(f"[REAL-TIME ALERT][{row[1]}] {row[2]} | Context: {row[3]} | ALERT_TYPE: {row[1]}")
                conn.close()
            except Exception as e:
                print(f"[MONITOR] Error in real-time alert monitor: {e}")
            time.sleep(poll_interval)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

# --- Utility: Audit Logging (unchanged) ---

def log_integrity_issues(issues: List[Tuple[str, Dict[str, Any]]], log_path: str = None):
    if log_path:
        log_path = db_utils._safe_db_path(log_path)
    else:
        log_path = str((Path(CONTEXT_DB_PATH).parent / "integrity_issues.log").resolve())
    with open(log_path, "a", encoding="utf-8") as f:
        for issue_type, contest in issues:
            f.write(json.dumps({"issue": issue_type, "contest": contest}) + "\n")

def detect_statistical_outliers(
    values: List[float],
    threshold: float = 3.0
) -> List[int]:
    if not values:
        return []
    arr = np.array(values)
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return []
    z_scores = np.abs((arr - mean) / std)
    return [i for i, z in enumerate(z_scores) if z > threshold]

# --- Example Usage ---
# After calling any processing function, call the corresponding print_* function for rich output.
# For example:
# results = analyze_contest_titles(contests)
# print_analyze_contest_titles(results)
# entity_summary = summarize_context_entities(contests)
# print_entity_summary(entity_summary)
# issues = advanced_cross_field_validation(contests)
# print_issues_table(issues, title="Advanced Cross-Field Validation Issues")
# contamination = auto_tune_contamination(X)
# print_auto_tune_result(contamination)
# anomalies, clusters = detect_anomalies_with_ml(contests)
# print_ml_anomalies(anomalies, contests)
"""
    from .Integrity_check import print_integrity_summary

    print_integrity_summary(contests, expected_year=2024)
    # or, if you have X:
    # print_integrity_summary(contests, expected_year=2024, X=X)
"""


def print_integrity_summary(contests, expected_year=None, X=None):
    """
    Print a full integrity summary using rich tables and panels.
    - contests: list of contest dicts
    - expected_year: optional, for date anomaly checks
    - X: optional, feature matrix for auto_tune_contamination
    """
    # Analyze contest titles (integrity, date, ML, suspicious)
    results = analyze_contest_titles(contests, expected_year=expected_year)
    # Add contests to results for ML anomaly printing
    results["contests"] = contests

    console.rule("[bold blue]Election Data Integrity Summary[/bold blue]")

    # Print integrity issues, date anomalies, ML anomalies, suspicious contests
    print_analyze_contest_titles(results)

    # Print entity summary
    entity_summary = summarize_context_entities(contests)
    print_entity_summary(entity_summary)

    # Print advanced cross-field validation issues
    issues = advanced_cross_field_validation(contests)
    print_issues_table(issues, title="Advanced Cross-Field Validation Issues")

    # Print auto-tuned contamination if X is provided
    if X is not None:
        contamination = auto_tune_contamination(X)
        print_auto_tune_result(contamination)

    console.rule("[bold blue]End of Integrity Summary[/bold blue]")
from __future__ import annotations

import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import numpy as np
import orjson
from rich.panel import Panel
from rich.table import Table
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import select

from ..config import CONTEXT_DB_PATH, CONTEXT_LIBRARY_PATH, LOG_DIR
from ..Context_Integration.librarian import clean_for_json
from ..utils import misc_utils
from ..utils.db_utils import get_session
from ..utils.logger_singleton import console
from ..utils.models import Alert
from ..utils.privilege_tiers import PrivilegeTier
from ..utils.shared_logic import (
    safe_all,
    safe_encode,
    safe_execute,
    safe_get,
    safe_items,
    safe_tolist,
)
from ..utils.spacy_utils import extract_dates, extract_entities, flag_suspicious_contests

# Use Agg backend for non-GUI environments (e.g., servers, CI/CD pipelines)
matplotlib.use("Agg")

# Lightweight, bounded monitor log (non-DB)
INTEGRITY_MONITOR_LOG = Path(LOG_DIR) / "integrity_monitor.jsonl"
_LOG_ENTRY_LIMITS = {
    "max_depth": 6,
    "max_list": 200,
    "max_dict": 120,
    "max_string": 2000,
}
try:
    INTEGRITY_MONITOR_LOG.touch(exist_ok=True)
except Exception:
    pass

def _trim_monitor_log(path: Path, max_bytes: int, keep_tail: int) -> None:
    if not path.exists():
        return
    try:
        size = path.stat().st_size
        if size <= max_bytes:
            return
        with path.open("rb") as handle:
            if keep_tail >= size:
                return
            handle.seek(-keep_tail, 2)
            tail = handle.read()
        if not tail:
            return
        # Align to next newline to avoid partial JSON lines
        first_nl = tail.find(b"\n")
        if first_nl != -1:
            tail = tail[first_nl + 1 :]
        with path.open("wb") as handle:
            handle.write(tail)
    except Exception:
        return

def _cap_log_value(value: Any, *, depth: int = 0) -> Any:
    if depth > _LOG_ENTRY_LIMITS["max_depth"]:
        return None
    if isinstance(value, str):
        if len(value) > _LOG_ENTRY_LIMITS["max_string"]:
            return value[: _LOG_ENTRY_LIMITS["max_string"]] + "...(truncated)"
        return value
    if isinstance(value, (bytes, bytearray)):
        size = len(value)
        return f"<bytes:{size}>"
    if isinstance(value, dict):
        capped = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= _LOG_ENTRY_LIMITS["max_dict"]:
                break
            capped[key] = _cap_log_value(item, depth=depth + 1)
        return capped
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        return [_cap_log_value(item, depth=depth + 1) for item in items[: _LOG_ENTRY_LIMITS["max_list"]]]
    return value

def log_integrity_monitor(event: Dict[str, Any], max_bytes: int = 2_000_000, keep_tail: int = 1_500_000) -> None:
    payload = dict(event or {})
    payload.setdefault("timestamp", time.time())
    payload = _cap_log_value(clean_for_json(payload))
    try:
        with INTEGRITY_MONITOR_LOG.open("ab") as handle:
            handle.write(orjson.dumps(payload) + b"\n")
        _trim_monitor_log(INTEGRITY_MONITOR_LOG, max_bytes=max_bytes, keep_tail=keep_tail)
    except Exception:
        return

def _ensure_alerts_table():
    # Table is managed by SQLAlchemy migrations; nothing to do here
    pass
_ensure_alerts_table()

# --- Data Processing Functions (unchanged) ---

def find_date_anomalies(contests, expected_year=None):
    anomalies = []
    for c in contests:
        dates = extract_dates(safe_get(c, "title", ""))
        if expected_year and not any(str(expected_year) in d for d in dates):
            anomalies.append(c)
    return anomalies

def detect_anomalies_with_ml(
    contexts: List[Dict[str, Any]],
    contamination: float = 0.05,
    n_estimators: int = 100,
    random_state: int = 42,
    embedding_model=None,
    trust_factors: Dict[str, Any] | None = None,
    privilege_tier: PrivilegeTier | None = None
) -> Tuple[List[int], np.ndarray]:
    if not contexts:
        return [], np.array([])
    features = []
    le_state = LabelEncoder()
    le_county = LabelEncoder()
    le_type = LabelEncoder()
    states = [safe_get(c, "state", "unknown") for c in contexts]
    counties = [safe_get(c, "county", "unknown") for c in contexts]
    types = [safe_get(c, "type_", "unknown") for c in contexts]
    le_state.fit(states)
    le_county.fit(counties)
    le_type.fit(types)
    for c in contexts:
        # Base features (7 dimensions)
        base_features = [
            le_state.transform([safe_get(c, "state", "unknown")])[0],
            le_county.transform([safe_get(c, "county", "unknown")])[0],
            le_type.transform([safe_get(c, "type_", "unknown")])[0],
            int(safe_get(c, "year", 0)) if str(safe_get(c, "year", "0")).isdigit() else 0,
            len(str(safe_get(c, "title", ""))),
            len(str(safe_get(c, "candidate", ""))) if safe_get(c, "candidate") else 0,
            len(str(safe_get(c, "party", ""))) if safe_get(c, "party") else 0,
        ]
        
        # Trust factors (9 dimensions) - normalized to [0, 1]
        trust_features = []
        if trust_factors:
            trust_features = [
                float(trust_factors.get("verified_domain", False)),        # 0 or 1
                float(trust_factors.get("gov_domain", False)),             # 0 or 1
                float(trust_factors.get("ssl_valid", False)) if trust_factors.get("ssl_valid") is not None else 0.5,  # 0-1
                float(trust_factors.get("suspicious_tld", False)),         # 0 or 1 (penality)
                float(len(trust_factors.get("phishing_indicators", []))) / 10.0,  # 0-1 (normalized count)
                float(trust_factors.get("historical_success", 0.0)),       # 0.0-1.0
                float(trust_factors.get("admin_boost_applied", False)),    # 0 or 1
                float(trust_factors.get("domain_mimicry", {}).get("detected", False)),  # 0 or 1
                float(trust_factors.get("allowlist_match", False)),        # 0 or 1
            ]
        else:
            # Default to neutral values (0.5) if no trust factors provided
            trust_features = [0.5] * 9
        
        # Embedding features (N dimensions, optional)
        emb = []
        title = safe_get(c, "title", "")
        if embedding_model and title:
            emb_encoded = safe_encode(title) if hasattr(embedding_model, "encode") else embedding_model.encode([title])
            emb = safe_tolist(emb_encoded[0] if isinstance(emb_encoded, (list, tuple)) and emb_encoded else emb_encoded)
        
        # Combine all features (7 base + 9 trust + N embedding)
        features.append(base_features + trust_features + emb)
    
    X = np.array(features)
    
    # Admin bypass: For FULL_TRUST + verified domain, use stricter anomaly detection
    if privilege_tier and privilege_tier >= PrivilegeTier.ADMIN_FULL_TRUST and trust_factors and trust_factors.get("verified_domain"):
        # Only flag SEVERE anomalies (raise contamination threshold)
        contamination = min(contamination, 0.01)  # Much stricter
    
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

feature_names = [
    "state", "county", "type", "year", "title_length", "candidate_length", "party_length",
    # Trust factor dimensions (9)
    "verified_domain", "gov_domain", "ssl_valid", "suspicious_tld", "phishing_count",
    "historical_success", "admin_boost_applied", "domain_mimicry", "allowlist_match",
    # Embedding dimensions follow (N)
]

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
        # Advanced: suspicious candidate reuse
        if isinstance(c.get("candidate"), str) and c.get("candidate").lower() in ["unknown", "n/a"]:
            issues.append(("suspicious_candidate", c))
        # Advanced: negative or zero votes
        if "votes" in c and isinstance(c["votes"], (int, float)) and c["votes"] <= 0:
            issues.append(("nonpositive_votes", c))
    return issues

def advanced_cross_field_validation(contests: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    issues = []
    for c in contests:
        if c.get("type_") == "Presidential" and c.get("state") not in ("us", "USA", "United States"):
            issues.append(("presidential_state_mismatch", c))
        if "votes" in c and isinstance(c["votes"], (int, float)) and c["votes"] < 0:
            issues.append(("negative_votes", c))
    return issues

def summarize_context_entities(contests) -> Dict[str, int]:
    entity_counter = Counter()
    for c in contests:
        title = safe_get(c, "title", "")
        entities = extract_entities(title)
        for _, label in entities:
            entity_counter[label] += 1
    return dict(entity_counter)

def analyze_contests(contests, expected_year=None, context_library_path=None, trust_factors: Dict[str, Any] | None = None, privilege_tier: PrivilegeTier | None = None) -> Dict[str, Any]:
    integrity_issues = election_integrity_checks(contests)
    date_anomalies = find_date_anomalies(contests, expected_year=expected_year)
    anomalies, clusters = detect_anomalies_with_ml(contests, trust_factors=trust_factors, privilege_tier=privilege_tier)
    if context_library_path is None:
        context_library_path = CONTEXT_LIBRARY_PATH
    flagged = flag_suspicious_contests(contests, context_library_path=context_library_path)
    
    # Build tier-specific summary
    tier_summary = {
        "privilege_tier": privilege_tier.value if privilege_tier else None,
        "tier_name": privilege_tier.name if privilege_tier else "UNKNOWN",
        "trust_factors_present": bool(trust_factors),
        "verified_domain": trust_factors.get("verified_domain") if trust_factors else None,
        "admin_boost_applied": trust_factors.get("admin_boost_applied") if trust_factors else False,
    }
    
    # Tier-specific anomaly thresholds (logging)
    if privilege_tier:
        if privilege_tier >= PrivilegeTier.ADMIN_FULL_TRUST and trust_factors and trust_factors.get("verified_domain"):
            tier_summary["anomaly_strategy"] = "strict_verified (only severe anomalies flagged)"
        elif privilege_tier == PrivilegeTier.ROOT_ADMIN:
            tier_summary["anomaly_strategy"] = "all_anomalies_reviewed (root admin bypass)"
        else:
            tier_summary["anomaly_strategy"] = "standard (default thresholds)"
    
    log_integrity_monitor({
        "type": "integrity_summary",
        "contests": len(contests or []),
        "integrity_issues": len(integrity_issues),
        "date_anomalies": len(date_anomalies),
        "ml_anomalies": len(anomalies or []),
        "flagged_suspicious": len(flagged or []),
        "privilege_tier": tier_summary["privilege_tier"],
        "trust_factors_present": tier_summary["trust_factors_present"],
    })
    
    return {
        "integrity_issues": integrity_issues,
        "date_anomalies": date_anomalies,
        "ml_anomalies": anomalies,
        "clusters": clusters.tolist() if hasattr(clusters, "tolist") else clusters,
        "flagged_suspicious": flagged,
        "tier_summary": tier_summary,
    }

def auto_tune_contamination(
    X: np.ndarray,
    initial_contamination: float = 0.2,
    min_contamination: float = 0.01,
    max_contamination: float = 0.2,
    plot: bool = False
) -> float:
    clf = IsolationForest(contamination=initial_contamination, random_state=42)
    if X is None or len(X) == 0:
        console.print("No contest features to check for anomalies.")
        return
    clf.fit(X)
    scores = -clf.decision_function(X)
    cutoff = np.percentile(scores, 90)
    n_anomalies = np.sum(scores >= cutoff)
    contamination = n_anomalies / len(scores)
    contamination = max(min_contamination, min(max_contamination, contamination))
    return contamination

# --- Rich Output Functions for CLI ---

def print_issues_table(issues, title="Issues") -> None:
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
            safe_get(contest, "title", ""),
            str(safe_get(contest, "year", "")),
            safe_get(contest, "state", ""),
            safe_get(contest, "county", "")
        )
    console.print(table)

def print_entity_summary(entity_summary) -> None:
    table = Table(title="Entity Label Summary")
    table.add_column("Entity Label", style="cyan")
    table.add_column("Count", style="magenta")
    for label, count in safe_items(entity_summary):
        table.add_row(label, str(count))
    console.print(table)

def print_ml_anomalies(anomaly_indices, contests, X=None, feature_names=None) -> None:
    if not anomaly_indices:
        console.print("[bold green]No ML anomalies detected.[/bold green]")
        return
    table = Table(title="ML Detected Anomalies", show_lines=True)
    table.add_column("Index", style="magenta")
    table.add_column("Title", style="cyan")
    table.add_column("Year", style="green")
    table.add_column("State", style="yellow")
    table.add_column("County", style="blue")
    if X is not None and feature_names is not None:
        for fname in feature_names:
            table.add_column(f"Δ {fname}", style="red")
    for idx in anomaly_indices:
        c = contests[idx]
        row = [
            str(idx),
            safe_get(c, "title", ""),
            str(safe_get(c, "year", "")),
            safe_get(c, "state", ""),
            safe_get(c, "county", "")
        ]
        if X is not None and feature_names is not None:
            # Show deviation from median for each feature
            medians = np.median(X, axis=0)
            deviations = [f"{X[idx, i] - medians[i]:.2f}" for i in range(X.shape[1])]
            row.extend(deviations)
        table.add_row(*row)
    console.print(table)

def print_date_anomalies(date_anomalies) -> None:
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
            safe_get(contest, "title", ""),
            str(safe_get(contest, "year", "")),
            safe_get(contest, "state", ""),
            safe_get(contest, "county", "")
        )
    console.print(table)

def print_auto_tune_result(contamination) -> None:
    if contamination is None:
        console.print(Panel("Auto-tuned contamination: [bold yellow]N/A[/bold yellow]", title="IsolationForest Auto-Tune"))
    else:
        console.print(Panel(f"Auto-tuned contamination: [bold green]{contamination:.4f}[/bold green]", title="IsolationForest Auto-Tune"))

def print_analyze_contests(results) -> None:
    print_issues_table(safe_get(results, "integrity_issues", []), title="Integrity Issues")
    print_date_anomalies(safe_get(results, "date_anomalies", []))
    print_ml_anomalies(safe_get(results, "ml_anomalies", []), safe_get(results, "contests", []))
    flagged = safe_get(results, "flagged_suspicious", [])
    if flagged:
        console.print(Panel(f"[yellow]{len(flagged)} suspicious contests flagged[/yellow]: {flagged}", title="Suspicious Contests"))
    else:
        console.print("[bold green]No suspicious contests flagged.[/bold green]")

# --- Real-Time Monitoring (unchanged) ---

def monitor_db_for_alerts(poll_interval: int = 10) -> None:
    """
    Monitor the alerts table in PostgreSQL for new alerts in real time using SQLAlchemy.
    Adds type checking for .id, .scalars().all(), and .execute.
    """
    last_alert_id = 0
    def monitor():
        nonlocal last_alert_id
        while True:
            try:
                with get_session() as session:
                    stmt = select(Alert).where(Alert.id > last_alert_id).order_by(Alert.id.asc())
                    result = safe_execute(session, stmt)
                    if result is None:
                        console.print("[MONITOR] Session object missing or failed 'execute' method.")
                        time.sleep(poll_interval)
                        continue
                    scalars = getattr(result, "scalars", None)
                    if not callable(scalars):
                        console.print("[MONITOR] Result object missing 'scalars' method.")
                        time.sleep(poll_interval)
                        continue
                    rows = scalars()
                    alerts = safe_all(rows)
                    if alerts is None:
                        console.print("[MONITOR] Scalars object missing or failed 'all' method.")
                        time.sleep(poll_interval)
                        continue
                    for row in alerts:
                        row_id = getattr(row, "id", None)
                        if not isinstance(row_id, int):
                            console.print(f"[MONITOR] Alert row missing valid 'id': {row}")
                            continue
                        last_alert_id = row_id
                        msg = getattr(row, "msg", "")
                        context = getattr(row, "context", "")
                        level = getattr(row, "level", "")
                        console.print(f"[REAL-TIME ALERT][{level}] {msg} | Context: {context} | ALERT_TYPE: {level}")
            except Exception as e:
                console.print(f"[MONITOR] Error in real-time alert monitor: {e}")
            time.sleep(poll_interval)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

# --- Utility: Audit Logging (unchanged) ---

def log_integrity_issues(issues: List[Tuple[str, Dict[str, Any]]], log_path: str = None) -> None:
    # Use .jsonl extension and safe path
    default_name = "integrity_issues.jsonl"
    if log_path:
        # Ensure .jsonl extension
        if not log_path.endswith(".jsonl"):
            log_path = re.sub(r"\.[^.]+$", "", log_path) + ".jsonl"
        log_path = misc_utils.safe_db_path(log_path)
    else:
        log_path = str((Path(CONTEXT_DB_PATH).parent / default_name).resolve())
    # Write each issue as a JSON object per line
    with open(log_path, "ab") as f:
        for issue_type, contest in issues:
            obj = {"issue": issue_type, "contest": clean_for_json(contest)}
            obj = _cap_log_value(obj)
            f.write(orjson.dumps(obj) + b"\n")

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
# results = analyze_contests(contests)
# print_analyze_contests(results)
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


def print_integrity_summary(contests, expected_year=None, X=None) -> None:
    """
    Print a full integrity summary using rich tables and panels.
    - contests: list of contest dicts
    - expected_year: optional, for date anomaly checks
    - X: optional, feature matrix for auto_tune_contamination
    """
    # Analyze contest titles (integrity, date, ML, suspicious)
    results = analyze_contests(contests, expected_year=expected_year)
    # Add contests to results for ML anomaly printing
    results["contests"] = contests

    console.rule("[bold blue]Election Data Integrity Summary[/bold blue]")

    # Print integrity issues, date anomalies, ML anomalies, suspicious contests
    print_analyze_contests(results)

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
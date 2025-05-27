import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
import matplotlib
# Ensure matplotlib uses a non-GUI backend for environments without display
# matplotlib.use('Agg')  # Use Agg backend for non-GUI environments
import matplotlib.pyplot as plt
import threading
import sqlite3
import json
import time
from ..utils.shared_logger import rprint
from typing import List, Dict, Any, Tuple, Optional
from ..utils.spacy_utils import extract_dates   

DB_PATH = "context_elections.db"  # Update path as needed

# --- Anomaly Detection & Clustering ---

conn = sqlite3.connect("context_elections.db")
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

def find_date_anomalies(contests, expected_year=None):
    """
    Find contests whose extracted dates do not match the expected year.
    """
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
    """
    Detect anomalies and clusters in contest data using IsolationForest and DBSCAN.
    Returns indices of anomalies and cluster labels.
    """
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
def plot_clusters(
    X: np.ndarray,
    clusters: np.ndarray,
    anomalies: Optional[List[int]] = None,
    title: str = "PCA Cluster Visualization",
    save_path: Optional[str] = None,
    show: bool = True,
    pca_components: int = 2,
    feature_indices: Optional[List[int]] = None,
    feature_names: Optional[List[str]] = None,
    highlight_color: str = "#D55E00",
    cluster_palette: str = "tab20",
    style: str = "seaborn-v0_8-darkgrid",
    annotate_points: bool = False,
    context: Optional[dict] = None,
    **kwargs
):
    """
    Robust cluster plot with PCA, feature selection, context-aware titles/labels, and anomaly highlighting.
    - feature_indices: which columns of X to use for PCA (default: all)
    - feature_names: names of features (for axis labels)
    - annotate_points: if True, annotate each point with its index
    - context: dict with keys like 'state', 'county', 'year' to enrich title
    - kwargs: passed to plt.scatter for further customization
    """
    import matplotlib.pyplot as plt
    import threading
    from sklearn.decomposition import PCA

    plt.style.use(style)
    # Select features for plotting
    if feature_indices is not None:
        X_plot = X[:, feature_indices]
        used_feature_names = [feature_names[i] for i in feature_indices] if feature_names else [f"Feature {i}" for i in feature_indices]
    else:
        X_plot = X
        used_feature_names = feature_names if feature_names else [f"Feature {i}" for i in range(X.shape[1])]

    # PCA for dimensionality reduction
    pca = PCA(n_components=pca_components)
    X_pca = pca.fit_transform(X_plot)

    plt.figure(figsize=kwargs.get("figsize", (10, 8)))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=clusters, cmap=cluster_palette, alpha=0.8, s=90, edgecolor='k', label='Cluster', **kwargs
    )

    # Highlight anomalies
    if anomalies is not None and len(anomalies) > 0:
        plt.scatter(
            X_pca[anomalies, 0], X_pca[anomalies, 1],
            color=highlight_color, marker='X', s=200, linewidths=2, label='Anomaly', zorder=5, edgecolor='black'
        )

    # Annotate points if requested
    if annotate_points:
        for idx, (x, y) in enumerate(zip(X_pca[:, 0], X_pca[:, 1])):
            plt.annotate(str(idx), (x, y), fontsize=8, alpha=0.7)

    # Build context-aware title
    context_title = title
    if context:
        parts = []
        for k in ('state', 'county', 'year'):
            v = context.get(k)
            if v:
                parts.append(str(v))
        if parts:
            context_title += " (" + ", ".join(parts) + ")"
    plt.title(context_title, fontsize=17, fontweight='bold')

    # Axis labels
    xlabel = f"PCA Component 1 ({', '.join(used_feature_names)})"
    ylabel = "PCA Component 2"
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Colorbar with cluster labels
    unique_clusters = set(clusters)
    if len(unique_clusters) > 1 or (-1 in unique_clusters):
        cbar = plt.colorbar(scatter, ticks=sorted([c for c in unique_clusters if c != -1]))
        cbar.set_label('Cluster Label')
        if -1 in unique_clusters:
            cbar.ax.plot([], [], color='gray', label='Noise/Outlier')

    # Thread-safe display/save
    if save_path or threading.current_thread() is not threading.main_thread() or not show:
        out_path = save_path or "cluster_plot.png"
        plt.savefig(out_path, dpi=180)
        print(f"[INFO] Cluster plot saved to {out_path}")
    else:
        plt.show()
    plt.close()

def plot_clusters(
    X: np.ndarray,
    clusters: np.ndarray,
    anomalies: Optional[List[int]] = None,
    title: str = "PCA Cluster Visualization",
    save_path: Optional[str] = None,
    show: bool = True,
    pca_components: int = 2,
    feature_indices: Optional[List[int]] = None,
    highlight_color: str = "#D55E00",
    cluster_palette: str = "plasma",
    style: str = "seaborn-v0_8-darkgrid",
    **kwargs
):
    """
    Plot clusters and highlight anomalies using PCA for dimensionality reduction.
    - anomalies: list of indices to highlight (e.g., detected anomalies)
    - feature_indices: which features to use for PCA (default: all)
    - save_path: if provided, saves the plot to this file
    - show: if True, attempts to show the plot (main thread only)
    - style: matplotlib style to use
    - kwargs: passed to plt.scatter for further customization
    """
    import matplotlib.pyplot as plt
    import threading
    from sklearn.decomposition import PCA

    plt.style.use(style)
    if feature_indices is not None:
        X_plot = X[:, feature_indices]
    else:
        X_plot = X

    pca = PCA(n_components=pca_components)
    X_pca = pca.fit_transform(X_plot)

    plt.figure(figsize=kwargs.get("figsize", (10, 8)))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=clusters, cmap=cluster_palette, alpha=0.75, s=80, edgecolor='k', label='Cluster', **kwargs
    )

    if anomalies is not None and len(anomalies) > 0:
        plt.scatter(
            X_pca[anomalies, 0], X_pca[anomalies, 1],
            color=highlight_color, marker='X', s=180, linewidths=2, label='Anomaly', zorder=5, edgecolor='black'
        )

    plt.title(title, fontsize=17, fontweight='bold')
    plt.xlabel('PCA Component 1', fontsize=13)
    plt.ylabel('PCA Component 2', fontsize=13)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Add colorbar for clusters if more than 1 cluster
    if len(set(clusters)) > 1:
        cbar = plt.colorbar(scatter, ticks=range(int(np.max(clusters)) + 1))
        cbar.set_label('Cluster Label')

    if save_path or threading.current_thread() is not threading.main_thread() or not show:
        out_path = save_path or "cluster_plot.png"
        plt.savefig(out_path, dpi=180)
        print(f"[INFO] Cluster plot saved to {out_path}")
    else:
        plt.show()
    plt.close()
    """
    **Example Usage:**
    ## for the _plot_anomaly_scores_ and _plot_clusters_ functions, you can use them as follows:
    
    ## To show interactively (main thread)
    
    plot_anomaly_scores(scores, cutoff=0.5, highlight=anomaly_indices)
    plot_clusters(X, clusters, anomalies=anomaly_indices)
    
    
    ## To save to files without showing (background thread)
    
    plot_anomaly_scores(scores, cutoff=0.5, highlight=anomaly_indices, save_path="my_scores.png", show=False)
    plot_clusters(X, clusters, anomalies=anomaly_indices, save_path="my_clusters.png", show=False)

    ## To customize style
    plot_anomaly_scores(scores, color='green', marker='s', figsize=(14,6))
    plot_clusters(X, clusters, cluster_palette="plasma", highlight_color="red")

    """    
def plot_anomaly_scores(
    scores: np.ndarray,
    cutoff: Optional[float] = None,
    highlight: Optional[List[int]] = None,
    title: str = "IsolationForest Anomaly Scores",
    xlabel: str = "Sample Index (sorted)",
    ylabel: str = "Anomaly Score",
    save_path: Optional[str] = None,
    show: bool = True,
    style: str = "seaborn-v0_8-darkgrid",
    highlight_color: str = "#CC79A7",
    score_color: str = "#0072B2",
    cutoff_color: str = "#D55E00",
    **kwargs
):
    """
    Plot anomaly scores for visual inspection.
    - highlight: list of indices to highlight (e.g., detected anomalies)
    - save_path: if provided, saves the plot to this file
    - show: if True, attempts to show the plot (main thread only)
    - style: matplotlib style to use
    - kwargs: passed to plt.plot for further customization
    """
    import matplotlib.pyplot as plt
    import threading

    plt.style.use(style)
    sorted_indices = np.argsort(scores)
    sorted_scores = np.array(scores)[sorted_indices]

    plt.figure(figsize=kwargs.get("figsize", (13, 6)))
    plt.plot(sorted_scores, marker='o', linestyle='-', color=score_color, label='Anomaly Score', **kwargs)

    if cutoff is not None:
        plt.axhline(cutoff, color=cutoff_color, linestyle='--', linewidth=2, label=f'Cutoff ({cutoff:.2f})')

    if highlight is not None and len(highlight) > 0:
        highlight_sorted = [np.where(sorted_indices == idx)[0][0] for idx in highlight if idx in sorted_indices]
        plt.scatter(highlight_sorted, sorted_scores[highlight_sorted], color=highlight_color, s=100, marker='X', label='Anomaly', zorder=5, edgecolor='black')

    plt.title(title, fontsize=17, fontweight='bold')
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    if save_path or threading.current_thread() is not threading.main_thread() or not show:
        out_path = save_path or "anomaly_scores.png"
        plt.savefig(out_path, dpi=180)
        print(f"[INFO] Anomaly score plot saved to {out_path}")
    else:
        plt.show()
    plt.close()    

def plot_clusters(
    X: np.ndarray,
    clusters: np.ndarray,
    anomalies: Optional[List[int]] = None,
    title: str = "PCA Cluster Visualization",
    save_path: Optional[str] = None,
    show: bool = True,
    pca_components: int = 2,
    highlight_color: str = "#D55E00",
    cluster_palette: str = "tab10",
    **kwargs
):
    """
    Plot clusters and highlight anomalies using PCA for dimensionality reduction.
    - anomalies: list of indices to highlight (e.g., detected anomalies)
    - save_path: if provided, saves the plot to this file
    - show: if True, attempts to show the plot (main thread only)
    - kwargs: passed to plt.scatter for further customization
    """
    import matplotlib.pyplot as plt
    import threading
    from sklearn.decomposition import PCA

    pca = PCA(n_components=pca_components)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=kwargs.get("figsize", (9, 7)))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=clusters, cmap=cluster_palette, alpha=0.7, s=60, edgecolor='k', label='Cluster', **kwargs
    )

    if anomalies is not None and len(anomalies) > 0:
        plt.scatter(
            X_pca[anomalies, 0], X_pca[anomalies, 1],
            color=highlight_color, marker='x', s=120, linewidths=2, label='Anomaly'
        )

    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel('PCA Component 1', fontsize=12)
    plt.ylabel('PCA Component 2', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    # Add colorbar for clusters if more than 1 cluster
    if len(set(clusters)) > 1:
        cbar = plt.colorbar(scatter, ticks=range(int(np.max(clusters)) + 1))
        cbar.set_label('Cluster Label')

    if save_path or threading.current_thread() is not threading.main_thread() or not show:
        out_path = save_path or "cluster_plot.png"
        plt.savefig(out_path, dpi=150)
        print(f"[INFO] Cluster plot saved to {out_path}")
    else:
        plt.show()
    plt.close()
    
def auto_tune_contamination(
    X: np.ndarray,
    initial_contamination: float = 0.2,
    min_contamination: float = 0.01,
    max_contamination: float = 0.2,
    plot: bool = False
) -> float:
    """
    Automatically tune the contamination parameter for IsolationForest.
    """
    clf = IsolationForest(contamination=initial_contamination, random_state=42)
    clf.fit(X)
    scores = -clf.decision_function(X)
    cutoff = np.percentile(scores, 90)
    n_anomalies = np.sum(scores >= cutoff)
    contamination = n_anomalies / len(scores)
    contamination = max(min_contamination, min(max_contamination, contamination))
    if plot:
        plot_anomaly_scores(scores, cutoff=cutoff)
    return contamination

# --- Integrity Checks ---

def election_integrity_checks(contests: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Basic integrity checks for contest data.
    Returns a list of (issue_type, contest) tuples.
    """
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
    """
    Advanced cross-field validation for logical inconsistencies.
    Returns a list of (issue_type, contest) tuples.
    """
    issues = []
    for c in contests:
        # Example: Check for logical inconsistencies
        if c.get("type") == "Presidential" and c.get("state") not in ("us", "USA", "United States"):
            issues.append(("presidential_state_mismatch", c))
        # Add more advanced checks as needed
        # Example: Check for negative vote counts
        if "votes" in c and isinstance(c["votes"], (int, float)) and c["votes"] < 0:
            issues.append(("negative_votes", c))
    return issues

# --- Real-Time Monitoring ---

def summarize_context_entities(contests):
    """
    Summarize named entities found in contest titles.
    Returns a dict: entity label -> count.
    """
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
    flagged = flag_suspicious_contests(contests, context_library_path=context_library_path)
    return {
        "integrity_issues": integrity_issues,
        "date_anomalies": date_anomalies,
        "ml_anomalies": anomalies,
        "clusters": clusters.tolist() if hasattr(clusters, "tolist") else clusters,
        "flagged_suspicious": flagged,
    }

def monitor_db_for_alerts(db_path: str = DB_PATH, poll_interval: int = 10):
    """
    Monitor the alerts table in the database for new alerts and print them in real time.
    """
    if db_path is None:
        from ..utils.db_utils import DB_PATH
        db_path = DB_PATH
    def monitor():
        last_alert_id = 0
        while True:
            try:
                conn = sqlite3.connect(db_path)
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

# --- Utility: Audit Logging ---

def log_integrity_issues(issues: List[Tuple[str, Dict[str, Any]]], log_path: str = "integrity_issues.log"):
    """
    Log integrity issues to a file for audit purposes.
    """
    with open(log_path, "a", encoding="utf-8") as f:
        for issue_type, contest in issues:
            f.write(json.dumps({"issue": issue_type, "contest": contest}) + "\n")

# --- Utility: Statistical Outlier Detection ---

def detect_statistical_outliers(
    values: List[float],
    threshold: float = 3.0
) -> List[int]:
    """
    Detect outliers in a list of numerical values using z-score.
    Returns indices of outliers.
    """
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
# Import and use these functions in context_organizer or context_coordinator as needed.
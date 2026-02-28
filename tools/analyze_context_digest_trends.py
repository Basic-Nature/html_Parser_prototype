#!/usr/bin/env python3
"""Analyze context digest trend deltas for ML/NLP drift alerts.

Usage:
  python tools/analyze_context_digest_trends.py
  python tools/analyze_context_digest_trends.py --window 20 --recent 5
  python tools/analyze_context_digest_trends.py --json-out tools/debug_headless_output/trend_alerts.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _load_trends(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Trend file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    parsed = json.loads(raw) if raw.strip() else []
    if not isinstance(parsed, list):
        raise ValueError("Trend file must contain a JSON list")
    return [item for item in parsed if isinstance(item, dict)]


def _slice_baseline_and_recent(trends: list[dict[str, Any]], window: int, recent: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(trends) <= recent:
        return trends[:-1], trends[-1:]
    recent_slice = trends[-recent:]
    baseline_end = len(trends) - recent
    baseline_start = max(0, baseline_end - window)
    baseline_slice = trends[baseline_start:baseline_end]
    return baseline_slice, recent_slice


def _extract_metrics(entries: list[dict[str, Any]]) -> dict[str, float]:
    if not entries:
        return {
            "confidence_avg": 0.0,
            "unknown_ratio": 0.0,
            "segments_review": 0.0,
            "pattern_kb_matches": 0.0,
        }

    conf_vals = []
    unknown_vals = []
    review_vals = []
    kb_vals = []

    for item in entries:
        confidence = item.get("confidence") if isinstance(item.get("confidence"), dict) else {}
        review = item.get("review_signals") if isinstance(item.get("review_signals"), dict) else {}

        conf_vals.append(_safe_float(confidence.get("avg"), 0.0))
        unknown_vals.append(_safe_float(item.get("unknown_ratio"), 0.0))
        review_vals.append(_safe_float(review.get("segments_needing_review"), 0.0))
        kb_vals.append(_safe_float(review.get("pattern_kb_matches"), 0.0))

    return {
        "confidence_avg": mean(conf_vals) if conf_vals else 0.0,
        "unknown_ratio": mean(unknown_vals) if unknown_vals else 0.0,
        "segments_review": mean(review_vals) if review_vals else 0.0,
        "pattern_kb_matches": mean(kb_vals) if kb_vals else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze context digest trend deltas")
    parser.add_argument(
        "--input",
        default="tools/debug_headless_output/context_digest_trends.json",
        help="Path to rolling trend JSON file",
    )
    parser.add_argument("--window", type=int, default=30, help="Baseline window size")
    parser.add_argument("--recent", type=int, default=5, help="Recent window size")
    parser.add_argument(
        "--conf-drop-threshold",
        type=float,
        default=0.08,
        help="Alert if confidence avg drops by at least this amount",
    )
    parser.add_argument(
        "--unknown-spike-threshold",
        type=float,
        default=0.10,
        help="Alert if unknown ratio increases by at least this amount",
    )
    parser.add_argument(
        "--review-spike-threshold",
        type=float,
        default=5.0,
        help="Alert if avg segments_needing_review increases by at least this amount",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write computed analysis JSON",
    )

    args = parser.parse_args()

    trends = _load_trends(Path(args.input))
    if len(trends) < 2:
        print("[INFO] Not enough trend entries to compute deltas.")
        return 0

    baseline_slice, recent_slice = _slice_baseline_and_recent(trends, max(1, args.window), max(1, args.recent))
    baseline = _extract_metrics(baseline_slice)
    recent = _extract_metrics(recent_slice)

    deltas = {
        "confidence_avg_delta": recent["confidence_avg"] - baseline["confidence_avg"],
        "unknown_ratio_delta": recent["unknown_ratio"] - baseline["unknown_ratio"],
        "segments_review_delta": recent["segments_review"] - baseline["segments_review"],
        "pattern_kb_matches_delta": recent["pattern_kb_matches"] - baseline["pattern_kb_matches"],
    }

    alerts = []
    if deltas["confidence_avg_delta"] <= -abs(args.conf_drop_threshold):
        alerts.append(
            {
                "type": "confidence_drop",
                "severity": "warning",
                "message": f"Confidence avg dropped by {abs(deltas['confidence_avg_delta']):.3f}",
            }
        )
    if deltas["unknown_ratio_delta"] >= abs(args.unknown_spike_threshold):
        alerts.append(
            {
                "type": "unknown_spike",
                "severity": "warning",
                "message": f"Unknown ratio increased by {deltas['unknown_ratio_delta']:.3f}",
            }
        )
    if deltas["segments_review_delta"] >= abs(args.review_spike_threshold):
        alerts.append(
            {
                "type": "review_spike",
                "severity": "warning",
                "message": f"Segments needing review increased by {deltas['segments_review_delta']:.2f}",
            }
        )

    result = {
        "input": args.input,
        "entry_count": len(trends),
        "baseline_window": len(baseline_slice),
        "recent_window": len(recent_slice),
        "baseline": baseline,
        "recent": recent,
        "deltas": deltas,
        "alerts": alerts,
        "status": "alert" if alerts else "ok",
    }

    print("[TREND] baseline:", baseline)
    print("[TREND] recent:", recent)
    print("[TREND] deltas:", deltas)
    if alerts:
        for alert in alerts:
            print(f"[ALERT] {alert['type']}: {alert['message']}")
    else:
        print("[ALERT] none")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote analysis JSON: {out_path}")

    return 0


def compute_integrity_signal(
    trend_file: str | Path = "tools/debug_headless_output/context_digest_trends.json",
    window: int = 30,
    recent: int = 5,
    conf_drop_threshold: float = 0.08,
    unknown_spike_threshold: float = 0.10,
    review_spike_threshold: float = 5.0,
) -> dict[str, Any]:
    """Compute integrity signal from trend deltas.
    
    Returns dict with baseline, recent, deltas, alerts, status.
    Safe to call inline during pipeline execution.
    """
    try:
        trends = _load_trends(Path(trend_file))
        if len(trends) < 2:
            return {
                "status": "insufficient_data",
                "entry_count": len(trends),
                "alerts": [],
            }
        
        baseline_slice, recent_slice = _slice_baseline_and_recent(trends, max(1, window), max(1, recent))
        baseline = _extract_metrics(baseline_slice)
        recent = _extract_metrics(recent_slice)
        
        deltas = {
            "confidence_avg_delta": recent["confidence_avg"] - baseline["confidence_avg"],
            "unknown_ratio_delta": recent["unknown_ratio"] - baseline["unknown_ratio"],
            "segments_review_delta": recent["segments_review"] - baseline["segments_review"],
            "pattern_kb_matches_delta": recent["pattern_kb_matches"] - baseline["pattern_kb_matches"],
        }
        
        alerts = []
        if deltas["confidence_avg_delta"] <= -abs(conf_drop_threshold):
            alerts.append({
                "type": "confidence_drop",
                "severity": "warning",
                "message": f"Confidence avg dropped by {abs(deltas['confidence_avg_delta']):.3f}",
            })
        if deltas["unknown_ratio_delta"] >= abs(unknown_spike_threshold):
            alerts.append({
                "type": "unknown_spike",
                "severity": "warning",
                "message": f"Unknown ratio increased by {deltas['unknown_ratio_delta']:.3f}",
            })
        if deltas["segments_review_delta"] >= abs(review_spike_threshold):
            alerts.append({
                "type": "review_spike",
                "severity": "warning",
                "message": f"Segments needing review increased by {deltas['segments_review_delta']:.2f}",
            })
        
        return {
            "entry_count": len(trends),
            "baseline_window": len(baseline_slice),
            "recent_window": len(recent_slice),
            "baseline": baseline,
            "recent": recent,
            "deltas": deltas,
            "alerts": alerts,
            "status": "alert" if alerts else "ok",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "alerts": [],
        }


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import orjson

from ..config import LOG_DIR

DEFAULT_NAV_LOG = Path(LOG_DIR) / "navigation_learning_log.jsonl"


def iter_navigation_feedback(log_path: str | Path = DEFAULT_NAV_LOG) -> Iterable[dict]:
    path = Path(log_path)
    if not path.exists() or not path.is_file():
        return []
    def _generator():
        with path.open("rb") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    entry = orjson.loads(line)
                except Exception:
                    continue
                if isinstance(entry, dict):
                    yield entry
    return _generator()


def build_training_dataset(
    *,
    log_path: str | Path = DEFAULT_NAV_LOG,
    limit: Optional[int] = None,
) -> List[dict]:
    samples: List[dict] = []
    for entry in iter_navigation_feedback(log_path):
        sample = {
            "script_id": entry.get("script_id"),
            "success": bool(entry.get("success")),
            "pre_context": entry.get("context_before") or {},
            "post_context": entry.get("context_after") or {},
            "telemetry": entry.get("telemetry") or [],
            "metadata": entry.get("metadata") or {},
        }
        sample["action_count"] = len(sample["telemetry"])
        sample["state"] = sample["post_context"].get("state") or sample["pre_context"].get("state")
        sample["county"] = sample["post_context"].get("county") or sample["pre_context"].get("county")
        samples.append(sample)
        if limit and len(samples) >= limit:
            break
    return samples


def export_training_dataset(
    output_path: str | Path,
    *,
    log_path: str | Path = DEFAULT_NAV_LOG,
    limit: Optional[int] = None,
) -> List[dict]:
    dataset = build_training_dataset(log_path=log_path, limit=limit)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(orjson.dumps(dataset, option=orjson.OPT_INDENT_2))
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Export navigation instruction training data.")
    parser.add_argument("--output", help="Optional path to write the dataset as JSON.")
    parser.add_argument("--log-path", default=str(DEFAULT_NAV_LOG), help="Navigation log to read.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit.")
    args = parser.parse_args()

    dataset = build_training_dataset(log_path=args.log_path, limit=args.limit)
    if args.output:
        export_training_dataset(args.output, log_path=args.log_path, limit=args.limit)
    else:
        print(orjson.dumps(dataset, option=orjson.OPT_INDENT_2).decode("utf-8"))


if __name__ == "__main__":
    main()

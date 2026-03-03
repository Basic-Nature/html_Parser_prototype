#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import orjson
except Exception:  # pragma: no cover
    orjson = None

from webapp.parser.config import OUTPUT_DIR
from webapp.parser.utils.data_comparator import DataComparator


def _load_json(path: Path) -> Any:
    raw = path.read_bytes()
    if orjson is not None:
        return orjson.loads(raw)
    return json.loads(raw.decode("utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if orjson is not None:
        path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate DL1 vs DL2 regression comparison report.")
    parser.add_argument("--dl1", required=True, help="Path to DL1 ground truth JSON")
    parser.add_argument("--dl2", required=True, help="Path to DL2 parser-output JSON")
    parser.add_argument(
        "--out",
        default=str(Path(OUTPUT_DIR) / "reports" / "data_comparison_latest.json"),
        help="Output report path",
    )
    parser.add_argument("--min-accuracy", type=float, default=0.95, help="Minimum required accuracy")
    parser.add_argument("--max-mismatches", type=int, default=0, help="Maximum allowed mismatches")
    parser.add_argument("--soft", action="store_true", help="Always exit 0 even when gate fails")

    args = parser.parse_args()

    dl1_path = Path(args.dl1).expanduser().resolve()
    dl2_path = Path(args.dl2).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    comparator = DataComparator()

    try:
        dl1_data = _load_json(dl1_path)
        dl2_data = _load_json(dl2_path)
    except Exception as exc:
        failure_report = {
            "schema_version": "1.0",
            "status": "error",
            "error": str(exc),
            "dl1": str(dl1_path),
            "dl2": str(dl2_path),
        }
        _write_json(out_path, failure_report)
        return 1

    result = comparator.compare_datasets(dl1_data, dl2_data)
    report = comparator.build_regression_report(
        result,
        context={
            "dl1_path": str(dl1_path),
            "dl2_path": str(dl2_path),
        },
        min_accuracy=args.min_accuracy,
        max_mismatches=args.max_mismatches,
    )

    _write_json(out_path, report)

    failed = report.get("gate", {}).get("status") == "fail"
    if failed and not args.soft:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

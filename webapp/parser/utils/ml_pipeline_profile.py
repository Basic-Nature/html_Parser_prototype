from __future__ import annotations

from pathlib import Path
from typing import Any

from webapp.parser.config import LOG_DIR, PROJECT_ROOT


def _count_jsonl_rows(path: Path, *, max_lines: int = 200_000) -> int:
    if not path.exists() or not path.is_file():
        return 0
    count = 0
    try:
        with path.open("rb") as handle:
            for line in handle:
                if not line.strip():
                    continue
                count += 1
                if count >= max_lines:
                    break
    except Exception:
        return 0
    return count


def _count_vocab_entries(vocab_root: Path, subdir: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "files": {},
        "file_count": 0,
        "entry_count": 0,
    }
    target_dir = vocab_root / subdir
    if not target_dir.exists() or not target_dir.is_dir():
        return result

    txt_files = sorted([p for p in target_dir.glob("*.txt") if p.is_file()])
    result["file_count"] = len(txt_files)

    total_entries = 0
    for path in txt_files:
        entry_count = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                for raw in handle:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    entry_count += 1
        except Exception:
            entry_count = 0

        total_entries += entry_count
        result["files"][path.name] = entry_count

    result["entry_count"] = total_entries
    return result


def get_ml_pipeline_profile() -> dict[str, Any]:
    """Return a compact profile of ML logic ingestion and tuning inputs."""
    from webapp.parser.utils.ml_telemetry import get_ml_telemetry_snapshot

    vocab_root = PROJECT_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab"
    profile = {
        "telemetry": get_ml_telemetry_snapshot(include_recent=True, limit=50),
        "training_inputs": {
            "structure_feedback_log_rows": _count_jsonl_rows(LOG_DIR / "structure_feedback_log.jsonl"),
            "spacy_ner_train_rows": _count_jsonl_rows(LOG_DIR / "spacy_ner_train_data.jsonl"),
            "spacy_ner_misaligned_rows": _count_jsonl_rows(LOG_DIR / "spacy_ner_misaligned.jsonl"),
            "ml_usage_telemetry_rows": _count_jsonl_rows(LOG_DIR / "ml_usage_telemetry.jsonl"),
        },
        "mapping_catalog": {
            "vocab_root": str(vocab_root),
            "entities": _count_vocab_entries(vocab_root, "entities"),
            "validators": _count_vocab_entries(vocab_root, "validators"),
            "sources": _count_vocab_entries(vocab_root, "sources"),
            "scoring": _count_vocab_entries(vocab_root, "scoring"),
        },
    }
    return profile

"""Small utility for safe, idempotent patch application used by the assistant.

This module provides a simple, deterministic helper that accepts a mapping
of file paths -> new content (str/bytes) and applies them atomically (per-file).
It detects no-op updates (identical content), records a short history, and
returns a structured result suitable for planners to decide whether to retry.

Note: This is intentionally simple and avoids git plumbing to keep behavior
consistent across environments (Windows, CI). It writes files directly and
records outcomes to `tools/patch_history.jsonl`.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


HISTORY_PATH = Path(__file__).parent / "patch_history.jsonl"


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _read_file_bytes(p: Path) -> Optional[bytes]:
    try:
        with open(p, "rb") as f:
            return f.read()
    except FileNotFoundError:
        return None
    except Exception:
        return None


def record_apply_result(record: dict, history_path: Path | None = None) -> None:
    target = history_path or HISTORY_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "ab") as fh:
            fh.write(json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n")
    except Exception:
        # Best-effort only; do not raise in helper
        pass


def apply_patch_safe(
    files: Dict[str, bytes | str],
    *,
    max_retries: int = 3,
    backoff_sec: float = 1.0,
    auto_apply: bool = True,
    patch_id: str | None = None,
    history_path: Path | None = None,
    history_extra: Dict[str, object] | None = None,
) -> dict:
    """Apply the provided file content mapping.

    Args:
      files: mapping of relative file paths -> new content (str or bytes).
      max_retries: number of attempts on transient errors.
      backoff_sec: seconds to wait between retries.
      auto_apply: if False, do not write files; still perform no-op detection.

    Returns a dict: {
      "applied": bool,
      "changed_files": [path,...],
      "noop_files": [path,...],
      "errors": [str,...],
      "patch_id": str,
      "timestamp": float,
    }
    """
    # Normalize inputs and compute patch id
    normalized: Dict[str, bytes] = {}
    for p, content in files.items():
        if isinstance(content, str):
            b = content.encode("utf-8")
        else:
            b = content
        normalized[str(p).replace("\\", "/")] = b

    concat = b"".join([p.encode("utf-8") + b":" + normalized[p] for p in sorted(normalized)])
    computed_patch_id = _sha256_bytes(concat)
    patch_id = patch_id or computed_patch_id

    attempt = 0
    errors: List[str] = []
    changed: List[str] = []
    noop: List[str] = []

    while attempt < max_retries:
        attempt += 1
        transient = False
        try:
            # Per-file check & write
            for rel_path, new_b in normalized.items():
                p = Path(rel_path)
                old_b = _read_file_bytes(p)
                old_hash = _sha256_bytes(old_b) if old_b is not None else None
                new_hash = _sha256_bytes(new_b)
                if old_hash == new_hash:
                    noop.append(rel_path)
                    continue

                if not auto_apply:
                    changed.append(rel_path)
                    continue

                # Ensure parent exists
                try:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    # write atomically using temp file in same dir
                    tmp = p.with_suffix(p.suffix + ".tmp")
                    with open(tmp, "wb") as tf:
                        tf.write(new_b)
                        tf.flush()
                        os.fsync(tf.fileno())
                    os.replace(tmp, p)
                    changed.append(rel_path)
                except Exception as e:
                    errors.append(f"write_failed:{rel_path}:{e}")
                    transient = True
                    break

            # If we reached here without transient error, break
            if transient:
                time.sleep(backoff_sec)
                continue
            break
        except Exception as e:
            errors.append(str(e))
            transient = True
            time.sleep(backoff_sec)
            continue

    result = {
        "applied": len(changed) > 0,
        "changed_files": changed,
        "noop_files": noop,
        "errors": errors,
        "patch_id": patch_id,
        "timestamp": time.time(),
        "attempts": attempt,
    }
    # Treat fully no-op case as applied=False but mark noop_files
    if not result["applied"] and result["noop_files"] and not result["errors"]:
        result["applied"] = True

    # Record short history for dedupe and inspection
    try:
        record = {
            "patch_id": patch_id,
            "applied": result["applied"],
            "changed_files": result["changed_files"],
            "noop_files": result["noop_files"],
            "errors": result["errors"],
            "timestamp": result["timestamp"],
            "tool": "agent_patch_utils.apply_patch_safe",
        }
        if history_extra:
            record.update(history_extra)
        record_apply_result(record, history_path=history_path)
    except Exception:
        pass

    return result


if __name__ == "__main__":
    # Quick manual demo when run directly
    print("agent_patch_utils demo: create small file and run no-op check")
    demo_path = Path("tools/_demo_patch_file.txt")
    r1 = apply_patch_safe({str(demo_path): "hello world\n"}, auto_apply=True)
    print("first apply:", r1)
    r2 = apply_patch_safe({str(demo_path): "hello world\n"}, auto_apply=True)
    print("second (no-op) apply:", r2)

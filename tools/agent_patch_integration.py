"""Integration helpers for higher-level agent patch workflows.

Provides a simple check-before-apply API that consults the patch history
to avoid re-applying identical patches, adds basic safety screening, and
delegates actual writes to `agent_patch_utils.apply_patch_safe`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set

from tools.agent_patch_utils import apply_patch_safe, _sha256_bytes


HISTORY_PATH = Path(__file__).parent / "patch_history.jsonl"
SUSPICIOUS_EXT = {".html", ".htm", ".js", ".css"}
SUSPICIOUS_TOKENS = {
    "static/",
    "templates/",
    "webapp/static",
    "public/",
}


def _load_applied_patch_ids(history_path: Path) -> Set[str]:
    out = set()
    if not history_path.exists():
        return out
    try:
        with open(history_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("applied"):
                    pid = obj.get("patch_id")
                    if pid:
                        out.add(pid)
    except Exception:
        return out
    return out


def _trim_history(history_path: Path, max_entries: int = 200) -> None:
    if max_entries <= 0 or not history_path.exists():
        return
    try:
        with open(history_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        if len(lines) <= max_entries:
            return
        trimmed = lines[-max_entries:]
        with open(history_path, "w", encoding="utf-8") as fh:
            fh.writelines(trimmed)
    except Exception:
        # Non-fatal; history trimming is best-effort
        return


def compute_patch_id(files: Dict[str, bytes | str]) -> str:
    # reuse the same patch id logic used by apply_patch_safe
    normalized = {}
    for p, content in files.items():
        b = content.encode("utf-8") if isinstance(content, str) else content
        normalized[str(p).replace("\\", "/")] = b
    concat = b"".join([p.encode("utf-8") + b":" + normalized[p] for p in sorted(normalized)])
    return _sha256_bytes(concat)


def _detect_suspicious_files(files: Dict[str, bytes | str]) -> List[str]:
    suspicious: List[str] = []
    for rel_path in files.keys():
        norm = str(rel_path).replace("\\", "/").lower()
        ext = Path(norm).suffix.lower()
        if ext in SUSPICIOUS_EXT:
            suspicious.append(rel_path)
            continue
        if any(token in norm for token in SUSPICIOUS_TOKENS):
            suspicious.append(rel_path)
    return suspicious


def apply_patch_if_needed(
    files: Dict[str, bytes | str], *, auto_apply: bool = True, max_retries: int = 3, backoff_sec: float = 1.0,
    allow_suspicious: bool = False, trim_history_max: int = 200, principal: str | None = None
) -> dict:
    """Apply files only if an identical patch hasn't been applied before.

    Returns the same result structure as `apply_patch_safe`. If the patch
    was previously applied (patch_id present in history), returns a small
    no-op result indicating it was skipped.
    """
    base_pid = compute_patch_id(files)
    principal_token = (principal or "").strip()
    if principal_token:
        principal_pid = _sha256_bytes((principal_token + ":" + base_pid).encode("utf-8"))
    else:
        principal_pid = base_pid
    pid = principal_pid
    # Safety check for static/html-like updates unless explicitly allowed
    suspicious = _detect_suspicious_files(files)
    if suspicious and not allow_suspicious:
        return {
            "applied": False,
            "changed_files": [],
            "noop_files": [],
            "errors": [f"suspicious_file:{p}" for p in suspicious],
            "patch_id": pid,
            "timestamp": None,
            "attempts": 0,
            "skipped": True,
            "reason": "suspicious_files",
            "terminate": True,
        }

    # Per-principal history file for isolation
    history_path = HISTORY_PATH if not principal_token else HISTORY_PATH.with_name(f"patch_history.{principal_token}.jsonl")

    applied_ids = _load_applied_patch_ids(history_path)
    if pid in applied_ids:
        return {
            "applied": True,
            "changed_files": [],
            "noop_files": list(files.keys()),
            "errors": [],
            "patch_id": pid,
            "timestamp": None,
            "attempts": 0,
            "skipped": True,
            "reason": "already_applied",
        }

    # Otherwise delegate to safe applier
    res = apply_patch_safe(
        files,
        max_retries=max_retries,
        backoff_sec=backoff_sec,
        auto_apply=auto_apply,
        patch_id=pid,
        history_path=history_path,
        history_extra={"principal": principal_token} if principal_token else None,
    )
    # Trim history to keep the file small and reduce parsing overhead
    _trim_history(history_path, max_entries=trim_history_max)
    return res

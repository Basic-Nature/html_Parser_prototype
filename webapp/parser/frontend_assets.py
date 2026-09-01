from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any


_STATIC_ROOT = Path(__file__).resolve().parents[1] / "static"
_F2_DIST_REL = PurePosixPath("dist/ballot-lens-f2")
_F2_MANIFEST_REL = _F2_DIST_REL / "manifest.json"


def _safe_manifest_asset(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("F2 manifest asset must be a non-empty string")

    normalized = value.replace("\\", "/").strip()
    candidate = PurePosixPath(normalized)

    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("F2 manifest asset escaped the build directory")

    return candidate.as_posix()


def _append_css(
    styles: list[str],
    seen: set[str],
    values: Any,
) -> None:
    if values is None:
        return
    if not isinstance(values, list):
        raise ValueError("F2 Vite manifest css field must be a list")

    for value in values:
        stylesheet = _safe_manifest_asset(value)
        if not stylesheet.lower().endswith(".css"):
            raise ValueError("F2 manifest css field contains non-CSS asset")
        if stylesheet not in seen:
            seen.add(stylesheet)
            styles.append(stylesheet)


def _collect_manifest_styles(
    payload: dict[str, Any],
    *,
    entry_key: str,
) -> list[str]:
    """Collect entry/import CSS plus Vite standalone CSS asset chunks."""
    entry = payload.get(entry_key)
    if not isinstance(entry, dict):
        raise ValueError("F2 Vite entry record is invalid")

    styles: list[str] = []
    seen_styles: set[str] = set()
    seen_chunks: set[str] = set()

    def walk_chunk(key: str) -> None:
        if key in seen_chunks:
            return
        seen_chunks.add(key)

        record = payload.get(key)
        if not isinstance(record, dict):
            raise ValueError("F2 Vite manifest import record is invalid")

        _append_css(styles, seen_styles, record.get("css"))

        imports = record.get("imports") or []
        if not isinstance(imports, list):
            raise ValueError("F2 Vite manifest imports field must be a list")
        for import_key in imports:
            if not isinstance(import_key, str) or not import_key:
                raise ValueError("F2 Vite manifest import key is invalid")
            walk_chunk(import_key)

    walk_chunk(entry_key)

    # Vite backend integration: with cssCodeSplit=false, the single extracted
    # stylesheet is represented as its own CSS manifest chunk (style.css).
    for key, record in payload.items():
        if not isinstance(key, str) or not isinstance(record, dict):
            continue

        src_value = record.get("src")
        css_record = (
            key.lower().endswith(".css")
            or (
                isinstance(src_value, str)
                and src_value.lower().endswith(".css")
            )
        )
        if not css_record:
            continue

        stylesheet = _safe_manifest_asset(record.get("file"))
        if not stylesheet.lower().endswith(".css"):
            raise ValueError(
                "F2 standalone CSS manifest record points to non-CSS asset"
            )
        if stylesheet not in seen_styles:
            seen_styles.add(stylesheet)
            styles.append(stylesheet)

    return styles


def load_ballot_lens_f2_assets(
    *,
    static_root: Path | None = None,
) -> dict[str, Any]:
    root = (static_root or _STATIC_ROOT).resolve()
    manifest_path = root / Path(*_F2_MANIFEST_REL.parts)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("F2 Vite manifest must be an object")

    entry_keys = [
        key
        for key, record in payload.items()
        if isinstance(record, dict) and record.get("isEntry") is True
    ]
    if len(entry_keys) != 1:
        raise ValueError("F2 Vite manifest must contain exactly one entry")

    entry_key = entry_keys[0]
    entry = payload[entry_key]
    script = _safe_manifest_asset(entry.get("file"))
    if not script.lower().endswith(".js"):
        raise ValueError("F2 Vite entry must resolve to JavaScript")

    styles = _collect_manifest_styles(payload, entry_key=entry_key)
    if not styles:
        raise ValueError("F2 Vite manifest did not expose a stylesheet")

    resolved_script = root / Path(*_F2_DIST_REL.parts) / Path(script)
    if not resolved_script.is_file():
        raise FileNotFoundError(resolved_script)

    for stylesheet in styles:
        resolved_style = root / Path(*_F2_DIST_REL.parts) / Path(stylesheet)
        if not resolved_style.is_file():
            raise FileNotFoundError(resolved_style)

    return {
        "script": (_F2_DIST_REL / script).as_posix(),
        "styles": [
            (_F2_DIST_REL / stylesheet).as_posix()
            for stylesheet in styles
        ],
    }

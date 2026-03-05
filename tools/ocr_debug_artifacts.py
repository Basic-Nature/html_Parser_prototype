#!/usr/bin/env python3
"""OCR helper for debug artifacts.

Usage:
  python tools/ocr_debug_artifacts.py --input tools/debug_headless_output
  python tools/ocr_debug_artifacts.py --input tools/debug_headless_output --prune-days 14
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

try:
    import pytesseract
    from PIL import Image
except Exception:
    Image = None
    pytesseract = None


def _utcnow() -> datetime:
    return datetime.utcnow()


def _iter_png_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.png"))


def _ocr_image(path: Path) -> str:
    if Image is None or pytesseract is None:
        raise RuntimeError("pytesseract/Pillow not available")
    with Image.open(path) as img:
        return pytesseract.image_to_string(img)


def _write_text(out_dir: Path, image_path: Path, text: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_path.stem}.txt"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def _prune_old_files(root: Path, cutoff: datetime) -> int:
    removed = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            mtime = datetime.utcfromtimestamp(path.stat().st_mtime)
        except Exception:
            continue
        if mtime < cutoff:
            try:
                path.unlink()
                removed += 1
            except Exception:
                continue
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR debug PNG artifacts.")
    parser.add_argument("--input", default="tools/debug_headless_output", help="Input directory with PNGs")
    parser.add_argument("--output", default="tools/debug_headless_output/ocr", help="Output directory for OCR text")
    parser.add_argument("--prune-days", type=int, default=0, help="Delete files older than N days")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return 2

    if args.prune_days > 0:
        cutoff = _utcnow() - timedelta(days=args.prune_days)
        removed = _prune_old_files(input_dir, cutoff)
        print(f"[INFO] Pruned {removed} file(s) older than {args.prune_days} days")

    png_files = _iter_png_files(input_dir)
    if not png_files:
        print("[INFO] No PNG files found.")
        return 0

    if Image is None or pytesseract is None:
        print("[ERROR] Missing dependencies. Install with: pip install pytesseract pillow")
        return 2

    summary = {
        "input": str(input_dir),
        "output": str(output_dir),
        "processed": [],
        "errors": [],
        "timestamp": _utcnow().isoformat() + "Z",
    }

    for png_path in png_files:
        try:
            text = _ocr_image(png_path)
            out_path = _write_text(output_dir, png_path, text)
            summary["processed"].append({
                "png": str(png_path),
                "text": str(out_path),
                "chars": len(text),
            })
            print(f"[OK] OCR: {png_path.name} -> {out_path.name}")
        except Exception as exc:
            summary["errors"].append({"png": str(png_path), "error": str(exc)})
            print(f"[WARN] OCR failed for {png_path.name}: {exc}")

    summary_path = output_dir / "ocr_summary.json"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[INFO] Summary written: {summary_path}")
    except Exception as exc:
        print(f"[WARN] Failed to write summary: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

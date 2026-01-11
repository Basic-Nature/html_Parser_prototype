import os
import sys
from pathlib import Path

# Minimal CLI to force OCR on a single PDF placed in uploads/
# Usage (PowerShell):
#   $env:ENABLE_OCR="1"; $env:ENABLE_OCR_FORCE="1"; $env:PDF_FAST_MODE="1"; \
#   python scripts/run_pdf_ocr_force.py "MyFile.pdf"
#
# Optional env vars on Windows:
#   $env:TESSERACT_CMD="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
#   $env:POPPLER_PATH="C:\\Program Files\\poppler-24.08.0\\Library\\bin"

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
UPLOADS_DIR = WORKSPACE_ROOT / "uploads"


def _resolve_input_path(arg: str) -> tuple[Path, str]:
    p = Path(arg)
    if p.is_file():
        # Accept absolute or relative paths
        return p.resolve(), p.suffix.lstrip(".").lower() or "pdf"
    # Try within uploads/
    in_uploads = UPLOADS_DIR / arg
    if in_uploads.is_file():
        return in_uploads.resolve(), in_uploads.suffix.lstrip(".").lower() or "pdf"
    # Try plain stem match in uploads
    candidates = list(UPLOADS_DIR.glob(f"**/{arg}"))
    if candidates:
        c = candidates[0]
        return c.resolve(), c.suffix.lstrip(".").lower() or "pdf"
    raise FileNotFoundError(f"Input file not found: {arg} (looked in uploads/ as well)")


def _set_default_env_flags():
    os.environ.setdefault("ENABLE_OCR", "1")
    os.environ.setdefault("ENABLE_OCR_FORCE", "1")
    os.environ.setdefault("PDF_FAST_MODE", "1")
    os.environ.setdefault("PDF_PROBE_MAX_PAGES", "8")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_pdf_ocr_force.py <file-in-uploads-or-path>")
        # List a few uploads to help
        if UPLOADS_DIR.exists():
            files = sorted([p.name for p in UPLOADS_DIR.iterdir() if p.is_file()])[:15]
            if files:
                print("\nUploads/ candidates:")
                for f in files:
                    print(" -", f)
        sys.exit(2)

    input_arg = sys.argv[1]
    input_path, ext = _resolve_input_path(input_arg)

    # Helpful Windows hints
    if os.name == "nt":
        tess = os.environ.get("TESSERACT_CMD")
        if not tess:
            print("[hint] Set TESSERACT_CMD if Tesseract is installed in a non-standard location.")
        poppler = os.environ.get("POPPLER_PATH")
        if not poppler:
            print("[hint] Set POPPLER_PATH if pdf2image is used and Poppler is installed.")

    _set_default_env_flags()

    # Import after env flags are set
    from webapp.parser.utils.logger_singleton import logger
    from webapp.parser.html_election_parser import main as parser_main

    logger.set_mode("cli")

    rel_forced = None
    try:
        # Prefer path relative to uploads for safe handlers
        abs_uploads = UPLOADS_DIR.resolve()
        if str(input_path).startswith(str(abs_uploads)):
            rel_forced = str(input_path.relative_to(abs_uploads)).replace("\\", "/")
        else:
            # If file is outside uploads, still pass absolute (handlers guard paths)
            rel_forced = str(input_path)
    except Exception:
        rel_forced = str(input_path)

    print(f"[run] Forcing OCR on: {rel_forced} (ext={ext})")
    try:
        parser_main(
            manual_source='uploads',
            force_parse_input_file=rel_forced,
            force_parse_format=ext or 'pdf',
            output_bypass=False,
            skip_url_prompt=True,
            url_source_label='run_pdf_ocr_force',
        )
    except SystemExit as se:
        # Allow graceful exit from parser if it calls sys.exit
        rc = int(getattr(se, 'code', 0) or 0)
        sys.exit(rc)
    except Exception as exc:
        print(f"[error] Parser run failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

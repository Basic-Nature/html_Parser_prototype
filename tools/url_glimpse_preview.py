from __future__ import annotations

import argparse
import json
from pathlib import Path
from webapp.parser.utils.url_glimpse import capture_url_glimpse


DEFAULT_OUT_DIR = Path("tools") / "debug_headless_output"


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture screenshot + DOM/table glimpse for a candidate results URL.")
    parser.add_argument("--url", required=True, help="URL to preflight preview before ingestion")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Artifact output directory")
    parser.add_argument("--timeout-ms", type=int, default=45000, help="Navigation timeout in ms")
    parser.add_argument("--wait-ms", type=int, default=1800, help="Post-load wait in ms")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    result = capture_url_glimpse(args.url, out_dir=out_dir, timeout_ms=args.timeout_ms, wait_ms=args.wait_ms)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

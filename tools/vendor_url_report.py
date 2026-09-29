import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).parent.parent))

from webapp.parser.config import URL_LIST_FILE
from webapp.parser.services.source_registry_runtime import load_trusted_url_library_view

URLS_PATH = URL_LIST_FILE
OUT_DIR = Path("output")
CSV_OUT = OUT_DIR / "vendor_url_report.csv"
MD_OUT = OUT_DIR / "vendor_url_report.md"

VENDOR_HOST_PATTERNS = {
    "clarity": re.compile(r"clarityelections\\.com", re.I),
    "voteworks": re.compile(r"voteworks\\.com", re.I),
    "dominion": re.compile(r"dominionvoting\\.com", re.I),
}

CANDIDATE_PATTERNS = {
    "enhancedvoting": re.compile(r"enhancedvoting\\.com", re.I),
    "enr": re.compile(r"\\benr\\b|enr[-.]", re.I),
}

VENDOR_HINT_PATTERNS = {
    "clarity": re.compile(r"clarity", re.I),
    "voteworks": re.compile(r"voteworks", re.I),
    "dominion": re.compile(r"dominion", re.I),
}


@dataclass
class UrlRow:
    year: str
    contest: str
    state: str
    scope: str
    fmt: str
    notes: str
    url: str


@dataclass
class UrlCheck:
    year: str
    state: str
    contest: str
    notes: str
    vendor: str
    confidence: str
    url: str
    resolved_url: str
    resolved_host: str
    http_status: str
    evidence: str


def parse_urls(path: Path) -> List[UrlRow]:
    rows: List[UrlRow] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        year, contest, state, scope, fmt, notes, url = parts[:7]
        rows.append(
            UrlRow(
                year=year.strip(),
                contest=contest.strip(),
                state=state.strip(),
                scope=scope.strip(),
                fmt=fmt.strip(),
                notes=notes.strip(),
                url=url.strip(),
            )
        )
    return rows


def parse_authority_rows(path: Path = URL_LIST_FILE) -> List[UrlRow]:
    # Project exact URL values from the authority-aware Source Registry view.
    entries, _diagnostics = load_trusted_url_library_view(path)
    rows: List[UrlRow] = []
    for entry in entries:
        url = entry.get("url")
        if not isinstance(url, str) or not url:
            continue
        rows.append(
            UrlRow(
                year=str(entry.get("year") or ""),
                contest=str(entry.get("contest") or ""),
                state=str(entry.get("state") or ""),
                scope=str(entry.get("scope") or ""),
                fmt=str(entry.get("format") or ""),
                notes=str(entry.get("notes") or ""),
                url=url,
            )
        )
    return rows


def dedupe_rows(rows: Iterable[UrlRow]) -> List[UrlRow]:
    seen = set()
    out: List[UrlRow] = []
    for row in rows:
        if row.url in seen:
            continue
        seen.add(row.url)
        out.append(row)
    return out


def classify_vendor(row: UrlRow) -> Optional[Tuple[str, str, str]]:
    url = row.url
    notes = row.notes or ""

    for vendor, pattern in VENDOR_HOST_PATTERNS.items():
        if pattern.search(url):
            return vendor, "confirmed", f"host:{vendor}"

    for vendor, pattern in CANDIDATE_PATTERNS.items():
        if pattern.search(url) or pattern.search(notes):
            return vendor, "candidate", f"candidate:{vendor}"

    for vendor, pattern in VENDOR_HINT_PATTERNS.items():
        if pattern.search(notes):
            return vendor, "candidate", f"notes:{vendor}"

    return None


def http_check(url: str, timeout: int = 15) -> Tuple[str, str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        req = Request(url, headers=headers, method="HEAD")
        with urlopen(req, timeout=timeout) as resp:
            resolved_url = resp.geturl()
            status = str(getattr(resp, "status", ""))
    except Exception:
        try:
            req = Request(url, headers=headers, method="GET")
            with urlopen(req, timeout=timeout) as resp:
                resolved_url = resp.geturl()
                status = str(getattr(resp, "status", ""))
        except Exception as exc:
            return "", "", f"error:{type(exc).__name__}"

    host = urlparse(resolved_url).netloc if resolved_url else ""
    return resolved_url, host, status or "unknown"


def browser_check(url: str, timeout_ms: int = 20000) -> Tuple[str, str, str]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return "", "", "error:playwright_unavailable"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            response = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            resolved_url = page.url or ""
            status = str(response.status) if response is not None else "unknown"
            context.close()
            browser.close()
    except Exception as exc:
        return "", "", f"error:{type(exc).__name__}"

    host = urlparse(resolved_url).netloc if resolved_url else ""
    return resolved_url, host, status or "unknown"


def build_report(rows: List[UrlRow], browser_fetch: bool = False, browser_timeout_ms: int = 20000) -> List[UrlCheck]:
    checks: List[UrlCheck] = []
    for row in rows:
        classification = classify_vendor(row)
        if not classification:
            continue
        vendor, confidence, evidence = classification
        resolved_url, host, status = http_check(row.url)
        if browser_fetch and status.startswith("error:"):
            resolved_url, host, status = browser_check(row.url, timeout_ms=browser_timeout_ms)
        checks.append(
            UrlCheck(
                year=row.year,
                state=row.state,
                contest=row.contest,
                notes=row.notes,
                vendor=vendor,
                confidence=confidence,
                url=row.url,
                resolved_url=resolved_url,
                resolved_host=host,
                http_status=status,
                evidence=evidence,
            )
        )
    return checks


def write_csv(checks: List[UrlCheck], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "year",
                "state",
                "contest",
                "vendor",
                "confidence",
                "url",
                "resolved_url",
                "resolved_host",
                "http_status",
                "notes",
                "evidence",
            ]
        )
        for item in checks:
            writer.writerow(
                [
                    item.year,
                    item.state,
                    item.contest,
                    item.vendor,
                    item.confidence,
                    item.url,
                    item.resolved_url,
                    item.resolved_host,
                    item.http_status,
                    item.notes,
                    item.evidence,
                ]
            )


def write_md(checks: List[UrlCheck], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Vendor URL Verification Report",
        "",
        "| Year | State | Vendor | Confidence | HTTP | Resolved Host | URL | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in checks:
        lines.append(
            f"| {item.year} | {item.state} | {item.vendor} | {item.confidence} | "
            f"{item.http_status} | {item.resolved_host} | {item.url} | {item.notes} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify vendor-related URLs.")
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Optional explicit noncanonical tab-delimited URL-list file. "
            "When omitted, the authority-aware Source Registry view is used."
        ),
    )
    parser.add_argument("--csv", default=str(CSV_OUT), help="CSV output path.")
    parser.add_argument("--md", default=str(MD_OUT), help="Markdown output path.")
    parser.add_argument(
        "--browser-fetch",
        action="store_true",
        help="Use Playwright to fetch URLs when basic HTTP fails.",
    )
    parser.add_argument(
        "--browser-timeout-ms",
        type=int,
        default=20000,
        help="Playwright navigation timeout in milliseconds.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    md_path = Path(args.md)

    if args.input is None:
        rows = parse_authority_rows(URL_LIST_FILE)
    else:
        input_path = Path(args.input)
        if input_path.resolve() == URL_LIST_FILE.resolve():
            print(
                "Canonical urls.txt is not a direct file-input authority. "
                "Omit --input to use the Source Registry authority view."
            )
            return 2
        if not input_path.exists():
            print(f"Missing explicit input file: {input_path}")
            return 1
        rows = parse_urls(input_path)

    rows = dedupe_rows(rows)
    checks = build_report(
        rows,
        browser_fetch=bool(args.browser_fetch),
        browser_timeout_ms=int(args.browser_timeout_ms),
    )

    if not checks:
        print("No vendor-related URLs found.")
        return 0

    write_csv(checks, csv_path)
    write_md(checks, md_path)
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote MD: {md_path}")
    print(f"Entries: {len(checks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

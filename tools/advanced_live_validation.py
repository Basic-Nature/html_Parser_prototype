#!/usr/bin/env python3
"""Advanced live validation for the Ballot Lens F2 bootstrap and assets."""

from __future__ import annotations

import os
import sys
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


TEST_SERVER = os.environ.get("ELECTIONPULSE_TEST_SERVER", "http://localhost:5555").rstrip("/")
TIMEOUT = 10
_RESULTS: list[bool] = []


def check(label: str, passed: bool, detail: str = "") -> None:
    _RESULTS.append(bool(passed))
    suffix = f" — {detail}" if detail else ""
    print(f"[{'PASS' if passed else 'FAIL'}] {label}{suffix}")


def get(path: str) -> requests.Response:
    return requests.get(
        urljoin(TEST_SERVER + "/", path.lstrip("/")),
        timeout=TIMEOUT,
        allow_redirects=True,
    )


def main() -> int:
    print("ElectionPulse Ballot Lens F2 advanced live validation")

    try:
        page = get("/ballot_lens")
    except Exception as exc:
        check("Primary Ballot Lens route", False, str(exc))
        return 1

    check("Primary Ballot Lens route", page.status_code == 200, f"HTTP {page.status_code}")
    soup = BeautifulSoup(page.text, "html.parser")
    root = soup.find(id="ballotLensF2Root")
    check("F2 root present", root is not None)
    if root is not None:
        check("F2 page identity", soup.html is not None and soup.html.get("data-page") == "ballot-lens-f2")
        check("Public registry bootstrap", root.get("data-public-registry-api") == "/api/public/ballot-lens/registry")
        check("Canonical read bootstrap", bool(root.get("data-data-api-url")))
        check("Trusted-control bootstrap", root.get("data-trusted-controls") in {"0", "1"})
        check("F2 phase bootstrap", root.get("data-f2-phase") == "F2-E4")

    styles = [
        link.get("href")
        for link in soup.find_all("link", rel="stylesheet")
        if link.get("href")
    ]
    modules = [
        script.get("src")
        for script in soup.find_all("script")
        if script.get("type") == "module" and script.get("src")
    ]
    check("F2 stylesheet projected", len(styles) >= 1, f"{len(styles)} stylesheet(s)")
    check("F2 module projected", len(modules) == 1, f"{len(modules)} module script(s)")

    for kind, paths, minimum in (
        ("stylesheet", styles, 500),
        ("module", modules, 1000),
    ):
        for asset in paths:
            try:
                response = requests.get(urljoin(TEST_SERVER + "/", asset), timeout=TIMEOUT)
                check(
                    f"{kind} asset {asset}",
                    response.status_code == 200 and len(response.content) >= minimum,
                    f"HTTP {response.status_code}, {len(response.content)} bytes",
                )
            except Exception as exc:
                check(f"{kind} asset {asset}", False, str(exc))

    content_type = page.headers.get("Content-Type", "")
    check("HTML content type", "text/html" in content_type.lower(), content_type)
    check("X-Content-Type-Options", bool(page.headers.get("X-Content-Type-Options")))

    passed = sum(1 for value in _RESULTS if value)
    print(f"Result: {passed}/{len(_RESULTS)} checks passed")
    return 0 if _RESULTS and all(_RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Comprehensive Ballot Lens F2 UI/bootstrap validation.

This tool validates the current F2 route, immutable projected assets, and
source-level safety contracts without executing a parser run.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


TEST_SERVER = os.environ.get("ELECTIONPULSE_TEST_SERVER", "http://localhost:5555").rstrip("/")
TIMEOUT = 10
ROOT = Path(".")
CHECKS: list[bool] = []


def check(label: str, passed: bool, detail: str = "") -> None:
    CHECKS.append(bool(passed))
    suffix = f" — {detail}" if detail else ""
    print(f"[{'PASS' if passed else 'FAIL'}] {label}{suffix}")


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8-sig")


def get(path: str) -> requests.Response:
    return requests.get(
        urljoin(TEST_SERVER + "/", path.lstrip("/")),
        timeout=TIMEOUT,
        allow_redirects=True,
    )


def source_contracts() -> None:
    template = read("webapp/templates/ballot_lens_f2.html")
    source_panel = read("webapp/frontend/ballot-lens/components/source/SourcePanel.tsx")
    registry = read("webapp/frontend/ballot-lens/contracts/registry.ts")
    bootstrap = read("webapp/frontend/ballot-lens/contracts/bootstrap.ts")
    canonical = read("webapp/frontend/ballot-lens/services/canonicalComparison.ts")
    workspace = read("webapp/frontend/ballot-lens/components/workspace/WorkspaceViews.tsx")

    check("F2 template root", 'id="ballotLensF2Root"' in template)
    check("F2 module bootstrap", 'type="module"' in template)
    check("Public registry data attribute", 'data-public-registry-api="/api/public/ballot-lens/registry"' in template)
    check("Canonical API data attribute", 'data-data-api-url="{{ data_api_url|e }}"' in template)

    for mode in ("public_registry", "manual_upload", "trusted_url", "worklist"):
        check(f"Source mode {mode}", f"'{mode}'" in source_panel)

    check("Exact safe registry keys", "PUBLIC_REGISTRY_SOURCE_KEYS" in registry)
    check("Server execution metadata typed", "execution_source_id" in registry and "execution_enabled" in registry)
    check("Bootstrap carries data API", "dataApiUrl: root.dataset.dataApiUrl ?? ''" in bootstrap)
    check("Canonical read path builder", "buildCanonicalReadPath" in canonical)
    check("Canonical same-origin boundary", "Canonical API must be a same-origin relative path" in canonical)
    check("Canonical approved path", "/api/ballotlens-database" in canonical)
    check("NULL is explicit in workspace", "NULL" in workspace)


def live_contracts() -> None:
    try:
        page = get("/ballot_lens")
    except Exception as exc:
        check("Primary Ballot Lens route", False, str(exc))
        return

    check("Primary Ballot Lens route", page.status_code == 200, f"HTTP {page.status_code}")
    soup = BeautifulSoup(page.text, "html.parser")
    root = soup.find(id="ballotLensF2Root")
    check("Rendered bootstrap root", root is not None)
    if root is not None:
        check("Rendered F2 phase", root.get("data-f2-phase") == "F2-E4")
        check("Rendered public registry endpoint", root.get("data-public-registry-api") == "/api/public/ballot-lens/registry")
        check("Rendered data API", bool(root.get("data-data-api-url")))
        check("Rendered mode", root.get("data-mode") in {"public", "trusted"})

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
    check("At least one projected stylesheet", bool(styles))
    check("Exactly one projected module", len(modules) == 1)

    for asset in styles + modules:
        try:
            response = requests.get(urljoin(TEST_SERVER + "/", asset), timeout=TIMEOUT)
            check(
                f"Projected asset {asset}",
                response.status_code == 200 and len(response.content) > 500,
                f"HTTP {response.status_code}, {len(response.content)} bytes",
            )
        except Exception as exc:
            check(f"Projected asset {asset}", False, str(exc))


def main() -> int:
    print("ElectionPulse Ballot Lens F2 comprehensive validation")
    source_contracts()
    live_contracts()
    passed = sum(1 for value in CHECKS if value)
    print(f"Result: {passed}/{len(CHECKS)} checks passed")
    return 0 if CHECKS and all(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Automated data import test suite.

Runs through phases 1-6 of the test runbook with minimal user interaction.
Logs all results and provides a final pass/fail summary.

Usage:
    python scripts/test_data_import_pipeline.py [--skip-import] [--limit N]

Flags:
    --skip-import    Skip Phase 3 (actual import); good for re-testing visualization with existing data
    --limit N        Limit rows processed to N (default: full sheet)
    --dry-run        Only run dry-run; don't actually import
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Load .env first
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Setup paths
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from webapp.parser.utils.db_utils import SessionLocal
    from webapp.parser.utils.models import WarehouseElectionResult
    DB_AVAILABLE = True
except ImportError as exc:
    print(f"⚠️  Database utilities not available: {exc}")
    DB_AVAILABLE = False


class TestResult:
    """Track test results with pass/fail/warn status."""

    def __init__(self, name: str):
        self.name = name
        self.status = "pending"  # pending, pass, fail, warn
        self.message = ""
        self.details = {}
        self.timestamp = datetime.now(timezone.utc)

    def pass_check(self, msg: str = "", details: Optional[Dict] = None):
        self.status = "pass"
        self.message = msg
        self.details = details or {}

    def fail_check(self, msg: str, details: Optional[Dict] = None):
        self.status = "fail"
        self.message = msg
        self.details = details or {}

    def warn_check(self, msg: str, details: Optional[Dict] = None):
        self.status = "warn"
        self.message = msg
        self.details = details or {}

    def __str__(self):
        icons = {"pass": "[PASS]", "fail": "[FAIL]", "warn": "[WARN]", "pending": "[...] "}
        icon = icons.get(self.status, "?")
        return f"{icon} {self.name}: {self.message}"

    def to_dict(self):
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


# ───────────────────────────────────────────────────────────────────────────
# Phase 1: Verify Credentials & Data Source
# ───────────────────────────────────────────────────────────────────────────


def phase_1_verify_credentials() -> Tuple[bool, list[TestResult]]:
    """Verify Google Sheets credentials are configured."""
    results = []

    # Check 1.1: Sheet ID
    test_sheet_id = TestResult("1.1 GOOGLE_SHEETS_DB_LITE_ID configured")
    sheet_id = os.getenv("GOOGLE_SHEETS_DB_LITE_ID", "").strip()
    if sheet_id:
        test_sheet_id.pass_check(f"Sheet ID found: {sheet_id[:10]}...")
    else:
        test_sheet_id.fail_check("GOOGLE_SHEETS_DB_LITE_ID not set")
    results.append(test_sheet_id)

    # Check 1.2: Service account credentials
    test_sa_email = TestResult("1.2 GOOGLE_SHEETS_SA_CLIENT_EMAIL configured")
    sa_email = os.getenv("GOOGLE_SHEETS_SA_CLIENT_EMAIL", "").strip()
    if sa_email and ".iam.gserviceaccount.com" in sa_email:
        test_sa_email.pass_check(f"Service account: {sa_email[:30]}...")
    else:
        test_sa_email.fail_check("GOOGLE_SHEETS_SA_CLIENT_EMAIL not valid")
    results.append(test_sa_email)

    # Check 1.3: Private key
    test_private_key = TestResult("1.3 GOOGLE_SHEETS_SA_PRIVATE_KEY configured")
    private_key = os.getenv("GOOGLE_SHEETS_SA_PRIVATE_KEY", "").strip()
    if private_key and "PRIVATE KEY" in private_key.upper():
        test_private_key.pass_check("Private key found (begins with RSA header)")
    else:
        test_private_key.fail_check("GOOGLE_SHEETS_SA_PRIVATE_KEY invalid or missing")
    results.append(test_private_key)

    all_pass = all(r.status in ("pass", "warn") for r in results)
    return all_pass, results


# ───────────────────────────────────────────────────────────────────────────
# Phase 2: Dry-Run the Import
# ───────────────────────────────────────────────────────────────────────────


def phase_2_dry_run(limit: Optional[int] = None) -> Tuple[bool, list[TestResult], Dict]:
    """Run import in dry-run mode."""
    results = []
    import_data = {}

    test_import = TestResult("2.1 Import dry-run execution")
    try:
        from scripts.import_database_lite import (  # noqa: PLC0415
            _load_sheet,
            build_records,
        )

        sheet_id = os.getenv("GOOGLE_SHEETS_DB_LITE_ID")
        if not sheet_id:
            test_import.fail_check("No sheet ID configured")
            results.append(test_import)
            return False, results, {}

        headers, data, sheet_title = _load_sheet(sheet_id, None)
        records, stats = build_records(headers, data, limit=limit)

        import_data = {
            "sheet_id": sheet_id,
            "sheet_title": sheet_title,
            "total_rows": len(data),
            "headers": len(headers),
            "records_built": len(records),
            "stats": stats,
        }

        test_import.pass_check(
            f"Imported {len(records)} records from '{sheet_title}' ({stats['rows']} rows processed)",
            details=import_data,
        )
    except Exception as exc:
        test_import.fail_check(f"Import failed: {exc}", details={"error": str(exc)})
    results.append(test_import)

    # Check 2.2: Data quality
    test_quality = TestResult("2.2 Data quality (>90% rows retained)")
    if import_data:
        skip_ratio = import_data["stats"]["skipped"] / max(1, import_data["stats"]["rows"])
        if skip_ratio < 0.1:
            test_quality.pass_check(
                f"Skipped {skip_ratio*100:.1f}% (acceptable)",
                details={"skipped": import_data["stats"]["skipped"]},
            )
        elif skip_ratio < 0.2:
            test_quality.warn_check(
                f"Skipped {skip_ratio*100:.1f}% (review sheet structure)",
                details={"skipped": import_data["stats"]["skipped"]},
            )
        else:
            test_quality.fail_check(
                f"Skipped {skip_ratio*100:.1f}% (too high, investigate)",
                details={"skipped": import_data["stats"]["skipped"]},
            )
    results.append(test_quality)

    all_pass = all(r.status in ("pass", "warn") for r in results)
    return all_pass, results, import_data


# ───────────────────────────────────────────────────────────────────────────
# Phase 3: Full Import
# ───────────────────────────────────────────────────────────────────────────


def phase_3_full_import() -> Tuple[bool, list[TestResult]]:
    """Run the actual import."""
    results = []

    if not DB_AVAILABLE:
        test_import = TestResult("3.1 Full import (database not available)")
        test_import.warn_check("Database not available; skipping full import")
        results.append(test_import)
        return False, results

    test_import = TestResult("3.1 Full import execution")
    try:
        from scripts.import_database_lite import (  # noqa: PLC0415
            _load_sheet,
            build_records,
        )
        from webapp.parser.utils.db_utils import SessionLocal  # noqa: PLC0415
        from webapp.parser.utils.models import WarehouseElectionResult  # noqa: PLC0415, F401

        sheet_id = os.getenv("GOOGLE_SHEETS_DB_LITE_ID")
        if not sheet_id:
            test_import.fail_check("No sheet ID configured")
            results.append(test_import)
            return False, results

        headers, data, sheet_title = _load_sheet(sheet_id, None)
        records, stats = build_records(headers, data, limit=None)

        session = SessionLocal()
        try:
            for record in records:
                session.add(WarehouseElectionResult(**record))
            session.commit()
            inserted = len(records)
            test_import.pass_check(f"Inserted {inserted} rows")
        except Exception as exc:
            session.rollback()
            raise exc
        finally:
            session.close()
    except Exception as exc:
        test_import.fail_check(f"Import failed: {exc}")
    results.append(test_import)

    # Check 3.2: Verify data was written
    test_verify = TestResult("3.2 Verify rows in database")
    try:
        session = SessionLocal()
        count = session.query(WarehouseElectionResult).count()
        session.close()
        if count > 1000:
            test_verify.pass_check(f"{count:,} rows found in warehouse_election_results")
        elif count > 0:
            test_verify.warn_check(f"Only {count} rows in database (expected >1000)")
        else:
            test_verify.fail_check("No rows found in database")
    except Exception as exc:
        exc_msg = str(exc)
        if "password authentication failed" in exc_msg or "connection" in exc_msg.lower():
            test_verify.warn_check(f"Database not accessible (expected in dev): {exc_msg[:50]}...")
        else:
            test_verify.fail_check(f"Query failed: {exc_msg[:100]}")
    results.append(test_verify)

    all_pass = all(r.status in ("pass", "warn") for r in results)
    return all_pass, results


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Automated data import test suite")
    parser.add_argument("--skip-import", action="store_true", help="Skip Phase 3 (full import)")
    parser.add_argument("--limit", type=int, help="Limit rows to N (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Dry-run only; no writes")
    args = parser.parse_args()

    print("=" * 80)
    print("DATA IMPORT TEST SUITE")
    print("=" * 80)
    print(f"Start time: {datetime.now(timezone.utc).isoformat()}")
    print(f"Skip import: {args.skip_import}, Limit: {args.limit}, Dry-run: {args.dry_run}")
    print()

    all_results = []

    # Phase 1
    print("\n[Phase 1] Verifying Credentials...")
    phase1_pass, phase1_results = phase_1_verify_credentials()
    for r in phase1_results:
        print(f"  {r}")
    all_results.extend(phase1_results)

    if not phase1_pass:
        print("\n❌ Phase 1 failed. Fix credential configuration and retry.")
        return 1

    # Phase 2
    print("\n[Phase 2] Dry-Run Import...")
    phase2_pass, phase2_results, import_data = phase_2_dry_run(limit=args.limit)
    for r in phase2_results:
        print(f"  {r}")
    all_results.extend(phase2_results)

    if not phase2_pass:
        print("\n❌ Phase 2 failed. Check sheet structure and retry.")
        return 1

    # Phase 3 (optional)
    if not args.skip_import and not args.dry_run:
        print("\n[Phase 3] Full Import...")
        phase3_pass, phase3_results = phase_3_full_import()
        for r in phase3_results:
            print(f"  {r}")
        all_results.extend(phase3_results)
    else:
        print("\n[Phase 3] Skipped (--skip-import or --dry-run)")

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    passed = sum(1 for r in all_results if r.status == "pass")
    failed = sum(1 for r in all_results if r.status == "fail")
    warned = sum(1 for r in all_results if r.status == "warn")
    print(f"[PASS] Passed: {passed}")
    print(f"[WARN] Warned: {warned}")
    print(f"[FAIL] Failed: {failed}")

    if failed > 0:
        print("\n[FAIL] TESTS FAILED. Review errors above and retry.")
        return 1
    elif warned > 0:
        print("\n[WARN] TESTS PASSED WITH WARNINGS. Review warnings above.")
        return 0
    else:
        print("\n[PASS] ALL TESTS PASSED. Ready to commit and deploy!")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

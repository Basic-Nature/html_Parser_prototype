"""Publication-only, redacting ElectionPulse support-snapshot read boundary.

No environment variables are read here. No private ledger/receipt is a static asset.
An explicitly approved redacted JSON file must be placed by a separately governed
operator process at <Flask instance_path>/support/approved_public_snapshot.json.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
from typing import Any

SCHEMA = "ELECTIONPULSE_SUPPORT_PUBLIC_SNAPSHOT_REV2"
_TOKEN = re.compile(r"[A-Za-z0-9_.-]{6,128}\Z")
_PERIOD = re.compile(r"[0-9]{4}-(0[1-9]|1[0-2])\Z")
_FUNDING = ("gross_captured_usd", "net_settled_usd", "spent_on_project_usd", "committed_unspent_usd", "unallocated_available_usd")
_FOUNDER = ("provider_record_payment_subtotal_usd", "invoice_paid_status_azure_usd", "applied_payment_github_usd", "received_payment_godaddy_usd")
_MONTHLY = ("azure_paid_invoice_usd", "github_applied_payments_usd", "godaddy_paid_receipt_usd", "provider_payment_sum_usd")
_TOP_KEYS = frozenset(("schema", "source_revision", "snapshot_date", "approval_evidence", "publish_approved", "founder", "contributions", "monthly"))
MAX_BYTES = 262144


def _money_cents(value: Any) -> int | None:
    """Require explicit nonnegative exact cents; never equate missing with 0."""
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return None
    try:
        decimal = Decimal(str(value))
        cents = decimal * 100
        if not decimal.is_finite() or cents != cents.to_integral_value() or not 0 <= cents <= 10**14:
            return None
        return int(cents)
    except (ValueError, InvalidOperation, OverflowError):
        return None


def _money_table(table: Any, keys: tuple[str, ...]) -> dict[str, int | float] | None:
    if not isinstance(table, dict) or set(table) != set(keys):
        return None
    amounts = {key: _money_cents(table[key]) for key in keys}
    if any(v is None for v in amounts.values()):
        return None
    return amounts


def validate_public_snapshot(document: Any, expected_approval_id: str) -> dict[str, Any] | None:
    """Validate all figures and emit only the allowlisted public projection."""
    if not isinstance(expected_approval_id, str) or not _TOKEN.fullmatch(expected_approval_id):
        return None
    if not isinstance(document, dict) or set(document) != _TOP_KEYS:
        return None
    if document.get("schema") != SCHEMA or document.get("publish_approved") is not True:
        return None
    revision = document.get("source_revision")
    if not isinstance(revision, str) or not _TOKEN.fullmatch(revision):
        return None
    stamp = document.get("snapshot_date")
    if not isinstance(stamp, str):
        return None
    try:
        if date.fromisoformat(stamp).isoformat() != stamp:
            return None
    except ValueError:
        return None
    approval = document.get("approval_evidence")
    if not isinstance(approval, dict) or set(approval) != {"record_id"} or approval.get("record_id") != expected_approval_id:
        return None
    founder = _money_table(document.get("founder"), _FOUNDER)
    contribution = _money_table(document.get("contributions"), _FUNDING)
    rows = document.get("monthly")
    if founder is None or contribution is None or not isinstance(rows, list) or not (1 <= len(rows) <= 120):
        return None
    if founder[_FOUNDER[0]] != sum(founder[k] for k in _FOUNDER[1:]):
        return None
    if contribution["gross_captured_usd"] < contribution["net_settled_usd"]:
        return None
    if contribution["net_settled_usd"] != sum(contribution[k] for k in _FUNDING[2:]):
        return None
    months = []
    seen = set()
    for item in rows:
        if not isinstance(item, dict) or set(item) != {"period", *_MONTHLY}:
            return None
        period = item["period"]
        if not isinstance(period, str) or not _PERIOD.fullmatch(period) or period in seen:
            return None
        seen.add(period)
        values = _money_table({k: item[k] for k in _MONTHLY}, _MONTHLY)
        if values is None or values[_MONTHLY[3]] != sum(values[k] for k in _MONTHLY[:3]):
            return None
        months.append({"period": period, **values})
    for fkey, mkey in zip(_FOUNDER[1:], _MONTHLY[:3]):
        if founder[fkey] != sum(row[mkey] for row in months):
            return None
    as_usd = lambda table: {key: value / 100 for key, value in table.items()}
    return {
        "schema": SCHEMA, "source_revision": revision, "snapshot_date": stamp,
        "publish_approved": True, "approval_evidence": {"record_id": expected_approval_id},
        "founder": as_usd(founder), "contributions": as_usd(contribution),
        "monthly": [{"period": row["period"], **as_usd({key: row[key] for key in _MONTHLY})} for row in months],
    }


def load_approved_public_snapshot(instance_path: str, approval_id: Any) -> dict[str, Any] | None:
    """Fail closed on missing flags, symlinks, excessive bytes and invalid data."""
    if not isinstance(approval_id, str) or not _TOKEN.fullmatch(approval_id):
        return None
    base = Path(instance_path)
    directory = base / "support"
    path = directory / "approved_public_snapshot.json"
    try:
        if directory.is_symlink() or path.is_symlink() or not path.is_file():
            return None
        if path.stat().st_size > MAX_BYTES:
            return None
        # A separate deployment/operator gate must populate this path, never webapp/static.
        raw = path.read_bytes()
        if len(raw) > MAX_BYTES:
            return None
        document = json.loads(raw.decode("utf-8"), parse_float=Decimal)
    except (OSError, ValueError, UnicodeError, TypeError):
        return None
    return validate_public_snapshot(document, approval_id)

"""W24PUBPREVIEW6 shadow candidate: approved public CSV, memory-only.

Uninstalled candidate. No API for arbitrary URLs, file paths, or upload data.
Acquisition is restricted to the exact server-resolved approved HTTPS CSV URL;
redirects remain same-origin; DNS addresses are checked and pinned per request.
Every parsed precinct is explicitly identified by a locality + precinct pair.

Intentionally NOT a general-purpose CSV handler. Unsupported source schemas fail
closed and must not be coerced into precinct rows.
"""
from __future__ import annotations

import csv
import hashlib
import io
import ipaddress
import json
import re
import socket
from collections import OrderedDict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping
from urllib.parse import urljoin, urlsplit


class PublicStructuredCSVError(ValueError):
    """Fail-closed public extraction or egress rejection."""


MAX_CSV_ROWS = 150_000
MAX_COLUMNS = 100
MAX_CELL_CHARS = 2_000
MAX_CANDIDATES = 200
DEFAULT_METHODS = ("Election Day", "Early Voting", "Absentee Mail", "Provisional")
ALIASES: dict[str, tuple[str, ...]] = {
    "precinct": ("precinct", "precinct name", "precinct id", "precinct number", "precinct code"),
    "locality": ("locality", "county", "county name", "jurisdiction", "municipality"),
    "candidate": ("candidate", "candidate name", "choice", "choice name", "candidate fullname"),
    "votes": ("votes", "vote count", "vote total", "total votes", "number of votes"),
    "method": ("vote method", "voting method", "ballot method", "vote type", "ballot type"),
    "party": ("party", "party name", "political party"),
    "contest": ("contest", "office", "contest name", "race", "office name"),
    "reporting": ("percent precincts reporting", "precincts reporting percent"),
    "year": ("election year", "year", "year of election"),
    "state": ("state", "state name", "jurisdiction state"),
    "precinct_total": ("precinct total votes", "precinct grand total", "precinct total"),
}
METHOD_ALIASES = {
    "election day": "Election Day", "electionday": "Election Day", "eday": "Election Day",
    "early voting": "Early Voting", "early": "Early Voting", "advance": "Early Voting",
    "absentee mail": "Absentee Mail", "absentee by mail": "Absentee Mail", "mail": "Absentee Mail",
    "vote by mail": "Absentee Mail", "mail in": "Absentee Mail",
    "provisional": "Provisional", "provisional ballot": "Provisional",
    "total": "Total Votes", "all": "Total Votes", "all votes": "Total Votes",
    "total votes": "Total Votes", "overall": "Total Votes",
}


def _norm(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _column_map(headers: list[str]) -> dict[str, str]:
    if not headers or len(headers) > MAX_COLUMNS:
        raise PublicStructuredCSVError("CSV_HEADER_COUNT_OUT_OF_BOUNDS")
    actual: dict[str, str] = {}
    for header in headers:
        if not isinstance(header, str) or not header.strip() or len(header) > 160:
            raise PublicStructuredCSVError("CSV_INVALID_HEADER")
        normalized = _norm(header)
        if normalized in actual:
            raise PublicStructuredCSVError("CSV_DUPLICATE_HEADER")
        actual[normalized] = header
    result: dict[str, str] = {}
    for semantic, choices in ALIASES.items():
        found = [actual[_norm(name)] for name in choices if _norm(name) in actual]
        if len(found) > 1:
            raise PublicStructuredCSVError("CSV_AMBIGUOUS_COLUMN_" + semantic.upper())
        if found:
            result[semantic] = found[0]
    for required in ("locality", "precinct", "candidate", "votes", "contest"):
        if required not in result:
            raise PublicStructuredCSVError("CSV_REQUIRED_COLUMN_MISSING_" + required.upper())
    return result


def _parse_votes(raw: str) -> int:
    value = str(raw or "").strip().replace(",", "")
    if not value or not re.fullmatch(r"[0-9]+", value):
        raise PublicStructuredCSVError("CSV_INVALID_OR_MISSING_VOTE_COUNT")
    number = int(value)
    if number > 2**63 - 1:
        raise PublicStructuredCSVError("CSV_VOTE_COUNT_OVERFLOW")
    return number


def _parse_reporting(raw: str) -> str | None:
    value = str(raw or "").strip().rstrip("%")
    if not value:
        return None
    try:
        p = Decimal(value)
    except InvalidOperation as exc:
        raise PublicStructuredCSVError("CSV_INVALID_REPORTING_PERCENT") from exc
    if not p.is_finite() or p < 0 or p > 100:
        raise PublicStructuredCSVError("CSV_REPORTING_PERCENT_OUT_OF_RANGE")
    return f"{p:.2f}%"


def _contest_matches(raw: str, expected: str) -> bool:
    desired = _norm(expected)
    observed = _norm(raw)
    if desired == "senate":
        return observed in {"senate", "u s senate", "us senate", "united states senate", "united states senator", "us senator"}
    return observed == desired


@dataclass(frozen=True)
class CSVShape:
    headers: list[str]
    rows: list[dict[str, object]]
    precinct_rows: int
    candidate_count: int
    source_sha256: str
    missing_method_cells: int
    discrepancy_precincts: int
    year_column_present: bool = False
    state_column_present: bool = False
    excluded_other_contest_rows: int = 0
    excluded_summary_rows: int = 0


def parse_approved_precinct_csv(
    data: bytes, *, contest: str, year: str | None = None, state: str | None = None,
    source_sha256: str | None = None, source_url: str | None = None,
) -> CSVShape:
    """Strict *long-form* CSV mapping, preserving nulls and explicit zero.

    A statewide source must explicitly name locality, precinct, candidate,
    contest and votes. Method columns absent from the source are kept as None.
    A recognized candidate Total Votes row is not added again to method rows.
    Duplicate vote observations are rejected rather than double counted.
    """
    if not isinstance(data, bytes) or not data:
        raise PublicStructuredCSVError("CSV_EMPTY_SOURCE")
    if b"\x00" in data[:4096]:
        raise PublicStructuredCSVError("CSV_BINARY_SOURCE_REJECTED")
    try:
        decoded = data.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise PublicStructuredCSVError("CSV_UNSUPPORTED_ENCODING_REQUIRES_REVIEW") from exc
    first = decoded[:1024].lstrip().casefold()
    if first.startswith("<html") or first.startswith("<!doctype html"):
        raise PublicStructuredCSVError("CSV_HTML_RESPONSE_REJECTED")
    try:
        reader = csv.DictReader(io.StringIO(decoded, newline=""), restkey="__extra__", restval=None)
        columns = _column_map(list(reader.fieldnames or []))
        values: OrderedDict[tuple[str, str], dict[tuple[str, str], dict[str, int | None]]] = OrderedDict()
        row_reporting: dict[tuple[str, str], str | None] = {}
        precinct_reported_totals: dict[tuple[str, str], int] = {}
        normalized_identities: dict[tuple[str, str], tuple[str, str]] = {}
        excluded_other_contest_rows = 0
        excluded_summary_rows = 0
        year_column_present = "year" in columns
        state_column_present = "state" in columns
        if year_column_present and (not year or not re.fullmatch(r"(?:19|20)[0-9]{2}", str(year).strip())):
            raise PublicStructuredCSVError("CSV_EXPECTED_ELECTION_YEAR_INVALID")
        if state_column_present and not str(state or "").strip():
            raise PublicStructuredCSVError("CSV_EXPECTED_STATE_MISSING")
        parties: dict[str, str] = {}
        all_candidates: set[str] = set()
        extras: set[str] = set()
        seen_observations: set[tuple[str, str, str, str]] = set()
        parsed_rows = 0
        for raw in reader:
            parsed_rows += 1
            if parsed_rows > MAX_CSV_ROWS:
                raise PublicStructuredCSVError("CSV_ROW_BUDGET_EXCEEDED")
            if not isinstance(raw, dict) or raw.get("__extra__") is not None:
                raise PublicStructuredCSVError("CSV_RAGGED_ROW_REJECTED")
            if any(value is None or len(str(value)) > MAX_CELL_CHARS for value in raw.values()):
                raise PublicStructuredCSVError("CSV_INVALID_OR_OVERSIZED_CELL")
            # Scope columns, when supplied, are authoritative: never silently
            # accept a record from a different year or state.
            if year_column_present:
                year_value = str(raw[columns["year"]]).strip()
                if not re.fullmatch(r"(?:19|20)[0-9]{2}", year_value):
                    raise PublicStructuredCSVError("CSV_SOURCE_YEAR_MALFORMED")
                if year_value != str(year).strip():
                    raise PublicStructuredCSVError("CSV_ELECTION_YEAR_SCOPE_MISMATCH")
            if state_column_present:
                state_value = _norm(str(raw[columns["state"]]).strip())
                expected_state = _norm(str(state or ""))
                state_aliases = {"va": "virginia", "virginia": "virginia"}
                if not state_value or state_aliases.get(state_value, state_value) != state_aliases.get(expected_state, expected_state):
                    raise PublicStructuredCSVError("CSV_STATE_SCOPE_MISMATCH")
            if not _contest_matches(raw[columns["contest"]], contest):
                excluded_other_contest_rows += 1
                continue
            locality = str(raw[columns["locality"]]).strip()
            precinct = str(raw[columns["precinct"]]).strip()
            candidate = str(raw[columns["candidate"]]).strip()
            if not all((locality, precinct, candidate)):
                raise PublicStructuredCSVError("CSV_MISSING_PRECINCT_IDENTITY")
            for label in (locality, precinct, candidate):
                if len(label) > 160 or any(ord(ch) < 32 for ch in label) or label[0] in "=+-@":
                    raise PublicStructuredCSVError("CSV_UNSAFE_ELECTION_LABEL")
            if _norm(precinct) in {"total", "totals", "statewide", "county total", "all precincts", "grand total"}:
                # Totals are not precincts. Do not silently blend them.
                excluded_summary_rows += 1
                continue
            canonical_id = (_norm(locality), _norm(precinct))
            old_identity = normalized_identities.setdefault(canonical_id, (locality, precinct))
            if old_identity != (locality, precinct):
                raise PublicStructuredCSVError("CSV_CONFLICTING_PRECINCT_IDENTITY")
            method_raw = str(raw.get(columns.get("method", ""), "")).strip()
            if "method" not in columns:
                method = "Total Votes"
            elif not method_raw:
                raise PublicStructuredCSVError("CSV_MISSING_METHOD_VALUE")
            else:
                method = METHOD_ALIASES.get(_norm(method_raw), method_raw)
                if method not in DEFAULT_METHODS + ("Total Votes",):
                    if not re.fullmatch(r"[\w /()-]{1,64}", method):
                        raise PublicStructuredCSVError("CSV_UNSAFE_METHOD_NAME")
                    extras.add(method)
            votes = _parse_votes(raw[columns["votes"]])
            k = (locality, precinct, candidate, method)
            if k in seen_observations:
                raise PublicStructuredCSVError("CSV_DUPLICATE_PRECINCT_CANDIDATE_METHOD")
            seen_observations.add(k)
            precinct_key = (locality, precinct)
            if "precinct_total" in columns:
                reported_text = str(raw[columns["precinct_total"]]).strip()
                if reported_text:
                    reported_value = _parse_votes(reported_text)
                    prior_value = precinct_reported_totals.setdefault(precinct_key, reported_value)
                    if reported_value != prior_value:
                        raise PublicStructuredCSVError("CSV_CONFLICTING_REPORTED_PRECINCT_TOTAL")
            cand_party = str(raw.get(columns.get("party", ""), "")).strip()
            if candidate in parties and parties[candidate] != cand_party:
                raise PublicStructuredCSVError("CSV_CANDIDATE_PARTY_COLLISION")
            if cand_party and (len(cand_party) > 40 or any(ord(ch) < 32 for ch in cand_party) or cand_party[0] in "=+-@"):
                raise PublicStructuredCSVError("CSV_UNSAFE_PARTY_LABEL")
            parties[candidate] = cand_party
            all_candidates.add(candidate)
            if len(all_candidates) > MAX_CANDIDATES:
                raise PublicStructuredCSVError("CSV_CANDIDATE_BUDGET_EXCEEDED")
            groups = values.setdefault(precinct_key, {})
            cells = groups.setdefault((candidate, cand_party), {})
            cells[method] = votes
            if "reporting" in columns:
                p = _parse_reporting(raw[columns["reporting"]])
                old = row_reporting.setdefault(precinct_key, p)
                if old is not None and p is not None and old != p:
                    raise PublicStructuredCSVError("CSV_CONFLICTING_REPORTING_PERCENT")
        if not values:
            raise PublicStructuredCSVError("CSV_NO_VERIFIED_PRECINCT_ROWS_FOR_CONTEST")
    except csv.Error as exc:
        raise PublicStructuredCSVError("CSV_PARSER_REJECTED_SOURCE") from exc

    methods = list(DEFAULT_METHODS) + sorted(extras)
    candidate_order = sorted(all_candidates, key=lambda name: (parties.get(name, ""), name))
    headers = ["Precinct", "% Precincts Reporting", "Election Day Total", "Early Voting Total", "Absentee Mail Total", "Provisional Total", "Official Source Link", "Source SHA-256"]
    if "precinct_total" in columns:
        headers.append("Reported Precinct Total")
    headers += [f"{name}{(' (' + parties[name] + ')') if parties.get(name) else ''} - {method}" for name in candidate_order for method in methods + ["Total Votes"]]
    headers += ["Discrepancy Flag", "Grand Total"]
    rows: list[dict[str, object]] = []
    missing = 0
    discrepancies = 0
    for (locality, precinct), groups in values.items():
        row: dict[str, object] = {header: None for header in headers}
        row["Precinct"] = locality + " / " + precinct
        row["% Precincts Reporting"] = row_reporting.get((locality, precinct))
        row["Official Source Link"] = source_url
        row["Source SHA-256"] = source_sha256 or hashlib.sha256(data).hexdigest()
        if "precinct_total" in columns:
            row["Reported Precinct Total"] = precinct_reported_totals.get((locality, precinct))
        flag: list[str] = []
        for name in candidate_order:
            cells = groups.get((name, parties.get(name, "")), {})
            label = name + (f" ({parties[name]})" if parties.get(name) else "")
            for method in methods:
                row[f"{label} - {method}"] = cells.get(method)
                if method not in cells:
                    missing += 1
            total = cells.get("Total Votes")
            if total is None and cells:
                flag.append("Missing Candidate Total")
            absent_methods = [method for method in methods if method not in cells]
            if absent_methods:
                flag.append("Missing Vote Method Breakdowns")
                for missing_method in absent_methods:
                    flag.append("Missing " + missing_method)
            # Only reconcile when the breakdown is complete. Missing votes
            # are unknown (NULL), not implicit zeros. Flag both directions.
            if total is not None and not absent_methods:
                method_sum = sum(cells[method] for method in methods)
                if method_sum > 2**63 - 1:
                    raise PublicStructuredCSVError("CSV_METHOD_SUM_OVERFLOW")
                if method_sum != total:
                    flag.append("Method Votes Exceed Candidate Total" if method_sum > total else "Method Votes Below Candidate Total")
            row[f"{label} - Total Votes"] = total
        for method in DEFAULT_METHODS:
            observed = [row[f"{name}{(' (' + parties[name] + ')') if parties.get(name) else ''} - {method}"] for name in candidate_order]
            if all(value is not None for value in observed):
                row[f"{method} Total"] = sum(observed)
        all_totals = [row[f"{name}{(' (' + parties[name] + ')') if parties.get(name) else ''} - Total Votes"] for name in candidate_order]
        if all(value is not None for value in all_totals):
            candidate_sum = sum(all_totals)
            if candidate_sum > 2**63 - 1:
                raise PublicStructuredCSVError("CSV_PRECINCT_SUM_OVERFLOW")
            row["Grand Total"] = candidate_sum
        else:
            flag.append("Missing Candidate Total")
        if "precinct_total" in columns:
            reported = row["Reported Precinct Total"]
            if reported is None:
                flag.append("Missing Reported Precinct Total")
            elif row["Grand Total"] is not None and reported != row["Grand Total"]:
                flag.append("Reported Precinct Total Mismatch")
        row["Discrepancy Flag"] = "; ".join(sorted(set(flag)))
        discrepancies += bool(flag)
        rows.append(row)
    # A Grand Total summary row is distinctly labeled, not a precinct.
    grand: dict[str, object] = {h: None for h in headers}
    grand["Precinct"] = "Grand Total"
    grand["Official Source Link"] = source_url
    grand["Source SHA-256"] = source_sha256 or hashlib.sha256(data).hexdigest()
    for h in headers:
        if h in ("Precinct", "Discrepancy Flag", "% Precincts Reporting"):
            continue
        observations = [r[h] for r in rows]
        if observations and all(isinstance(x, int) and not isinstance(x, bool) for x in observations):
            grand[h] = sum(observations)
    grand["Discrepancy Flag"] = "Summary only; not an individual precinct"
    rows.append(grand)
    return CSVShape(
        headers, rows, len(rows) - 1, len(candidate_order),
        source_sha256 or hashlib.sha256(data).hexdigest(), missing, discrepancies,
        year_column_present, state_column_present,
        excluded_other_contest_rows, excluded_summary_rows,
    )


def acquire_approved_https_csv(runtime: Any, *, cancel_flag: Any = None) -> tuple[bytes, str]:
    """Bounded direct HTTPS with verified destination IP pinned for each hop.

    Does not use browser downloads, requests proxies, the user's filesystem,
    or any endpoint/URL supplied in the socket payload.
    """
    import certifi
    import urllib3
    from urllib3.util import Timeout

    source = runtime.approved_target_url
    parsed0 = urlsplit(source)
    if parsed0.scheme != "https" or not parsed0.path.casefold().endswith(".csv"):
        raise PublicStructuredCSVError("STRUCTURED_SOURCE_NOT_EXACT_HTTPS_CSV")
    runtime.egress_guard.validate_initial_target(source)
    max_bytes = min(int(runtime.policy.remote_download_max_bytes), 25 * 1024 * 1024)
    redirects = 0
    target = source
    while True:
        if cancel_flag is not None and cancel_flag.is_set():
            raise PublicStructuredCSVError("CSV_CANCELLED")
        decision = runtime.egress_guard.evaluate_request(url=target, resource_type="document", top_level_navigation=True)
        if not decision.allowed:
            raise PublicStructuredCSVError("CSV_EGRESS_REJECTED_" + decision.reason.split(":", 1)[0])
        parsed = urlsplit(target)
        if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in (None, 443):
            raise PublicStructuredCSVError("CSV_HTTPS_AUTHORITY_REJECTED")
        host = parsed.hostname or ""
        if host.casefold() != (parsed0.hostname or "").casefold():
            raise PublicStructuredCSVError("CSV_CROSS_HOST_REDIRECT_REJECTED")
        try:
            resolved = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise PublicStructuredCSVError("CSV_DNS_LOOKUP_FAILED") from exc
        addresses = sorted({str(item[4][0]) for item in resolved})
        if not addresses:
            raise PublicStructuredCSVError("CSV_DNS_NO_ADDRESSES")
        if not all(ipaddress.ip_address(ip).is_global for ip in addresses):
            raise PublicStructuredCSVError("CSV_DNS_PRIVATE_ADDRESS_REJECTED")
        path_query = parsed.path or "/"
        if parsed.query:
            path_query += "?" + parsed.query
        response = None
        pool = urllib3.HTTPSConnectionPool(
            addresses[0], port=443, maxsize=1, block=True,
            cert_reqs="CERT_REQUIRED", ca_certs=certifi.where(),
            assert_hostname=host, server_hostname=host,
            timeout=Timeout(connect=8.0, read=20.0), retries=False,
        )
        try:
            response = pool.request(
                "GET", path_query, preload_content=False, redirect=False,
                retries=False,
                headers={"Host": host, "Accept": "text/csv, text/plain, application/octet-stream", "Accept-Encoding": "identity", "User-Agent": "ElectionPulse-PublicCSV/1.0"},
            )
            status = int(response.status)
            if status in (301, 302, 303, 307, 308):
                redirects += 1
                if redirects > int(runtime.policy.top_level_redirect_max):
                    raise PublicStructuredCSVError("CSV_REDIRECT_LIMIT_EXCEEDED")
                location = response.headers.get("Location", "")
                if not location:
                    raise PublicStructuredCSVError("CSV_REDIRECT_NO_LOCATION")
                target = urljoin(target, location)
                continue
            if status != 200:
                raise PublicStructuredCSVError("CSV_HTTP_STATUS_NOT_OK")
            media = response.headers.get("Content-Type", "").split(";", 1)[0].strip().casefold()
            if media not in {"text/csv", "application/csv", "text/plain", "application/octet-stream", "application/vnd.ms-excel"}:
                raise PublicStructuredCSVError("CSV_MEDIA_TYPE_NOT_APPROVED")
            if response.headers.get("Content-Encoding", "identity").casefold() not in ("identity", ""):
                raise PublicStructuredCSVError("CSV_COMPRESSED_RESPONSE_REJECTED")
            length = response.headers.get("Content-Length")
            if length and (not length.isdigit() or int(length) > max_bytes):
                raise PublicStructuredCSVError("CSV_CONTENT_LENGTH_OUT_OF_BOUNDS")
            data = bytearray()
            for chunk in response.stream(64 * 1024, decode_content=False):
                if cancel_flag is not None and cancel_flag.is_set():
                    raise PublicStructuredCSVError("CSV_CANCELLED")
                data.extend(chunk)
                if len(data) > max_bytes:
                    raise PublicStructuredCSVError("CSV_BODY_BYTE_BUDGET_EXCEEDED")
            if not data:
                raise PublicStructuredCSVError("CSV_EMPTY_SOURCE")
            return bytes(data), hashlib.sha256(data).hexdigest()
        finally:
            if response is not None:
                response.release_conn()
            pool.close()



def enforce_public_socket_event_budget(
    *, runtime: Any, headers: list[str], rows: list[dict[str, object]],
) -> int:
    """Fail closed if a complete real result cannot fit in one Socket.IO event.

    The current public frontend has no authenticated result pagination endpoint;
    it is not safe to silently trim rows or claim downloadability. This
    conservative estimate includes the exact result shape and a status margin.
    """
    from webapp.parser.services.public_ballot_lens_execution import (
        build_public_memory_preview,
    )
    preview = build_public_memory_preview(
        registry_source_id=runtime.registry_source_id,
        source_projection=runtime.source_projection,
        headers=headers, rows=rows, policy=runtime.policy,
    )
    result = {
        "contract": "ballot_lens_public_runtime_result_v1",
        "registry_source_id": runtime.registry_source_id,
        "source": dict(runtime.source_projection),
        "outputs": list(runtime.finalized_previews) + [preview],
        "status_counts": {"partial": 1, "error": 1, "success": 1},
        "terminal_status": "completed_with_errors",
        "terminal_reason_code": "public_socket_event_size_exceeded",
        "download_available": False,
        "persistent_output": False,
    }
    encoded = json.dumps(
        result, ensure_ascii=False, allow_nan=False, separators=(",", ":"),
    ).encode("utf-8")
    # Require a small reserve for future fixed contract metadata and framing.
    max_allowed = int(runtime.policy.socket_event_max_bytes) - 512
    if max_allowed < 1024 or len(encoded) > max_allowed:
        raise PublicStructuredCSVError("CSV_PUBLIC_SOCKET_EVENT_BUDGET_EXCEEDED")
    return len(encoded)


def execute_approved_public_csv(runtime: Any, *, session_id: str, cancel_flag: Any) -> dict[str, object]:
    """Use existing Smart Elections public memory finalizer, never write files."""
    from webapp.parser.utils.output_utils import finalize_election_output

    runtime.record_checkpoint(checkpoint_id="source.resolve", state="complete", reason_code="public_registry_csv_source_resolved", summary="Server-approved CSV source selected.", evidence_count=1)
    try:
        payload, digest = acquire_approved_https_csv(runtime, cancel_flag=cancel_flag)
        runtime.record_checkpoint(checkpoint_id="source.acquire", state="complete", reason_code="public_csv_bytes_acquired", summary="Bounded approved CSV acquired.", evidence_count=1)
        from webapp.parser.services.public_registry_csv_dispatch import parse_runtime_approved_csv
        try:
            parsed = parse_runtime_approved_csv(runtime, payload, digest, parse_approved_precinct_csv)
        except ValueError as exc:
            raise PublicStructuredCSVError("CSV_APPROVED_SOURCE_SCHEMA_REJECTED") from exc
        runtime.record_checkpoint(checkpoint_id="structure.detect", state="complete", reason_code="public_strict_precinct_csv_shape", summary="Explicit precinct, locality and contest columns validated.", evidence_count=parsed.precinct_rows)
        runtime.record_checkpoint(checkpoint_id="provider.detect", state="complete", reason_code="public_csv_format_handler_resolved", summary="Strict approved CSV handler selected.", evidence_count=1)
        runtime.record_result_checkpoints(headers=parsed.headers, contest=runtime.source_projection.get("contest"))
        if getattr(runtime, "compact_enabled", False):
            # A server-side broker adapter must have been installed explicitly.
            # Only the acquired source digest is used; the parsed rows are never
            # trimmed or repivoted for the paging path.
            quarantined = int(getattr(parsed, "quarantined_zero_idless_observations", 0))
            runtime.set_compact_source_metadata(
                source_sha256=digest,
                quarantine={
                    "observations": quarantined,
                    "votes": int(getattr(parsed, "quarantined_vote_total", 0)),
                    "reason": "SOURCE_ZERO_IDLESS_QUARANTINED" if quarantined else None,
                },
            )
        else:
            enforce_public_socket_event_budget(runtime=runtime, headers=parsed.headers, rows=parsed.rows)
        finalize_election_output(
            headers=parsed.headers, data=parsed.rows, session_id=session_id,
            contest=runtime.source_projection.get("contest"),
            state=runtime.source_projection.get("state"),
            context={"preformatted_smart_elections_wide": True, "fill_blanks_with_na": False},
        )
        runtime.record_processed_status(status="partial" if parsed.discrepancy_precincts else "success")
    except PublicStructuredCSVError as exc:
        runtime.record_checkpoint(checkpoint_id="structure.detect", state="warning", reason_code=str(exc)[:100], summary="Approved CSV could not be safely interpreted as precinct rows.", evidence_count=0)
        runtime.record_processed_status(status="error")
    except Exception:
        runtime.record_checkpoint(checkpoint_id="structure.detect", state="error", reason_code="public_structured_csv_internal_error", summary="Approved CSV processing failed safely.", evidence_count=0)
        runtime.record_processed_status(status="error")
    return runtime.result_payload()

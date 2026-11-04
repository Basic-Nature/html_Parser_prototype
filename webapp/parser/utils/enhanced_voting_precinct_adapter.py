"""Non-network Enhanced Voting-style JSON precinct extraction.

W24PUBINGEST-AZ1B1 offline candidate. The caller must independently authorize,
resolve and retrieve the source. Structure detection is NOT source authorization.
One output row = (source county ID, source contest ID, source precinct ID).
Missing source values are None, never silently recoded as zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

STANDARD_METHODS = ("Election Day", "Early Voting", "Absentee Mail", "Provisional")


class EnhancedVotingPrecinctError(ValueError):
    """Raised when the requested source selection is ambiguous or malformed."""


@dataclass
class PrecinctTable:
    headers: list[str]
    rows: list[dict[str, Any]]
    metadata: dict[str, Any]
    contest: str
    state: str
    county: str


def matches_enhanced_voting_shape(value: Any) -> bool:
    """Recognize the observed public JSON structure; never confer authority."""
    return (
        isinstance(value, dict)
        and isinstance(value.get("results"), dict)
        and isinstance(value.get("localResults"), list)
        and isinstance(value.get("electionDate"), str)
        and isinstance(value.get("results", {}).get("ballotItems"), list)
    )


def _count(value: Any, where: str, issues: list[dict[str, str]]) -> int | None:
    if type(value) is int and value >= 0:
        return value
    if value is None:
        issues.append({"code": "MISSING_COUNT", "location": where})
    else:
        issues.append({"code": "INVALID_COUNT", "location": where})
    return None


def _source_label(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EnhancedVotingPrecinctError("missing source label: " + where)
    return value.strip()


def _selected_counties(payload: Mapping[str, Any], county: str | None) -> list[dict]:
    selected = []
    for entry in payload["localResults"]:
        if not isinstance(entry, dict):
            raise EnhancedVotingPrecinctError("county is not an object")
        if county is None or str(entry.get("id")) == county or entry.get("name") == county:
            selected.append(entry)
    if not selected:
        raise EnhancedVotingPrecinctError("county was not found in this snapshot")
    if county is not None and len(selected) != 1:
        raise EnhancedVotingPrecinctError("ambiguous county selector")
    return selected


def build_precinct_table(
    payload: Mapping[str, Any],
    *,
    contest_id: str,
    county: str | None = None,
    source_sha256: str | None = None,
    source_link: str | None = None,
) -> PrecinctTable:
    """Return a selected election contest as Smart Elections precinct rows.

    Selection is by *source contest ID* and optional exact county name/ID.
    Results are never merged across different contests or elections. If county
    is None, include all counties that actually contain this contest ID.
    """
    if not matches_enhanced_voting_shape(payload):
        raise EnhancedVotingPrecinctError("unrecognized Enhanced Voting JSON structure")
    if not isinstance(contest_id, str) or not contest_id.strip():
        raise EnhancedVotingPrecinctError("an explicit contest ID is required")
    contest_id = contest_id.strip()
    if source_sha256 is not None and (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(c not in "0123456789abcdef" for c in source_sha256)
    ):
        raise EnhancedVotingPrecinctError("source hash must be lowercase SHA-256")
    # A source link must be supplied by the trusted caller, not derived from a
    # user-controlled filename or treated as proof of registry authorization.
    source_link = source_link or ""
    source_state = _source_label(payload["results"].get("name"), "state")
    election_date = _source_label(payload.get("electionDate"), "electionDate")
    snapshot_at = str(payload.get("createdAt") or "")
    selected_contests: list[tuple[dict, dict]] = []
    county_ids: set[str] = set()
    for county_obj in _selected_counties(payload, county):
        county_id = str(county_obj.get("id") or "")
        if not county_id or county_id in county_ids:
            raise EnhancedVotingPrecinctError("missing or duplicate county identifier")
        county_ids.add(county_id)
        matching = [
            c for c in (county_obj.get("ballotItems") or [])
            if isinstance(c, dict) and str(c.get("id")) == contest_id
        ]
        if len(matching) > 1:
            raise EnhancedVotingPrecinctError("duplicate county-contest identity")
        if matching:
            selected_contests.append((county_obj, matching[0]))
    if not selected_contests:
        raise EnhancedVotingPrecinctError("contest is absent in selected county scope")
    titles = {_source_label(c.get("name"), "contest") for _, c in selected_contests}
    if len(titles) != 1:
        raise EnhancedVotingPrecinctError("same contest ID has conflicting contest titles")
    title = titles.pop()

    candidate_names: dict[str, str] = {}
    candidate_parties: dict[str, dict[str, str]] = {}
    methods_seen: set[str] = set()
    issues: list[dict[str, str]] = []
    # First pass: discover the complete column universe across all selected
    # county-contest records, including candidates with zero votes.
    for county_obj, contest in selected_contests:
        cid = str(county_obj["id"])
        seen_candidate_ids: set[str] = set()
        for option in contest.get("ballotOptions") or []:
            if not isinstance(option, dict):
                raise EnhancedVotingPrecinctError("ballot option is not an object")
            oid = str(option.get("id") or "")
            name = _source_label(option.get("name"), "ballotOption.name")
            if not oid or oid in seen_candidate_ids:
                raise EnhancedVotingPrecinctError("missing or duplicate candidate ID")
            seen_candidate_ids.add(oid)
            if oid in candidate_names and candidate_names[oid] != name:
                raise EnhancedVotingPrecinctError("candidate ID changes label across counties")
            candidate_names[oid] = name
            candidate_parties.setdefault(oid, {})[cid] = str(option.get("politicalParty") or "")
            for precinct in option.get("precinctResults") or []:
                if not isinstance(precinct, dict):
                    raise EnhancedVotingPrecinctError("precinct result is not an object")
                for group in precinct.get("groupResults") or []:
                    if not isinstance(group, dict):
                        raise EnhancedVotingPrecinctError("vote group is not an object")
                    methods_seen.add(_source_label(group.get("groupName"), "groupName"))
    if not candidate_names:
        raise EnhancedVotingPrecinctError("contest has no ballot options")

    # Stable source identity is candidate ID, not source party strings (which
    # vary between county short codes and long labels).
    ordered_ids = sorted(candidate_names, key=lambda k: (0, int(k)) if k.isdigit() else (1, k))
    labels: dict[str, str] = {}
    used_labels: set[str] = set()
    for oid in ordered_ids:
        base = candidate_names[oid]
        label = base if base not in used_labels else f"{base} [ID {oid}]"
        used_labels.add(label)
        labels[oid] = label
    extra_methods = sorted(methods_seen - set(STANDARD_METHODS))
    methods = [*STANDARD_METHODS, *extra_methods]
    missing_in_feed = [m for m in STANDARD_METHODS if m not in methods_seen]

    headers = [
        "County", "County ID", "Precinct", "Precinct ID", "Is Virtual",
        "Reporting Status", "% Precincts Reporting", "Election Date",
        "Snapshot Created At", "Contest ID", "Source Link", "Source SHA256",
    ]
    headers.extend(f"{m} Total" for m in methods)
    for oid in ordered_ids:
        headers.extend(f"{labels[oid]} - {m}" for m in methods)
        headers.append(f"{labels[oid]} - Total Votes")
    headers.extend(["Grand Total", "Discrepancy Flag", "Missing Fields"])

    rows: list[dict[str, Any]] = []
    for county_obj, contest in selected_contests:
        county_name = _source_label(county_obj.get("name"), "county.name")
        county_id = str(county_obj["id"])
        participating = _count(contest.get("precinctsParticipating"),
                               f"{county_id}/{contest_id}/participating", issues)
        reporting = _count(contest.get("precinctsReporting"),
                           f"{county_id}/{contest_id}/reporting", issues)
        pct = (f"{100 * reporting / participating:.2f}%"
               if participating is not None and participating > 0 and reporting is not None else None)
        keyed: dict[str, dict[str, Any]] = {}
        seen_candidates: set[str] = set()
        for option in contest.get("ballotOptions") or []:
            oid = str(option["id"])
            seen_candidates.add(oid)
            reported_total = _count(option.get("voteCount"), f"{county_id}/{contest_id}/{oid}/total", issues)
            candidate_precinct_total = 0
            candidate_has_missing = False
            seen_precincts: set[str] = set()
            for precinct in option.get("precinctResults") or []:
                pid = str(precinct.get("id") or "")
                if not pid or pid in seen_precincts:
                    raise EnhancedVotingPrecinctError("missing or duplicate precinct ID within candidate")
                seen_precincts.add(pid)
                pname = _source_label(precinct.get("name"), "precinct.name")
                status = str(precinct.get("reportingStatus") or "")
                virtual = precinct.get("isVirtual")
                if pid not in keyed:
                    keyed[pid] = {
                        "County": county_name, "County ID": county_id,
                        "Precinct": pname, "Precinct ID": pid,
                        "Is Virtual": virtual,
                        "Reporting Status": status,
                        "% Precincts Reporting": pct,
                        "Election Date": election_date,
                        "Snapshot Created At": snapshot_at,
                        "Contest ID": contest_id,
                        "Source Link": source_link,
                        "Source SHA256": source_sha256,
                        "_flags": set(),
                    }
                row = keyed[pid]
                if row["Precinct"] != pname or row["Is Virtual"] != virtual:
                    row["_flags"].add("PRECINCT_METADATA_CONFLICT")
                if row["Reporting Status"] != status:
                    row["_flags"].add("STATUS_CONFLICT")
                seen_groups: set[str] = set()
                known_groups: dict[str, int | None] = {}
                for group in precinct.get("groupResults") or []:
                    group_name = _source_label(group.get("groupName"), "groupName")
                    if group_name in seen_groups:
                        row["_flags"].add("DUPLICATE_VOTE_METHOD")
                    seen_groups.add(group_name)
                    known_groups[group_name] = _count(
                        group.get("voteCount"), f"{county_id}/{contest_id}/{oid}/{pid}/{group_name}", issues
                    )
                for m in methods:
                    row[f"{labels[oid]} - {m}"] = known_groups.get(m)
                votes = _count(precinct.get("voteCount"), f"{county_id}/{contest_id}/{oid}/{pid}/total", issues)
                row[f"{labels[oid]} - Total Votes"] = votes
                if votes is None:
                    candidate_has_missing = True
                else:
                    candidate_precinct_total += votes
                if votes is not None and known_groups and all(v is not None for v in known_groups.values()):
                    if sum(known_groups.values()) != votes:
                        row["_flags"].add("CANDIDATE_METHOD_MISMATCH")
                        issues.append({"code": "CANDIDATE_METHOD_MISMATCH", "location": f"{county_id}/{contest_id}/{oid}/{pid}"})
            if not candidate_has_missing and reported_total is not None and candidate_precinct_total != reported_total:
                issues.append({"code": "CANDIDATE_PRECINCT_MISMATCH", "location": f"{county_id}/{contest_id}/{oid}"})
        for pid, row in keyed.items():
            absent_ids = [oid for oid in ordered_ids if f"{labels[oid]} - Total Votes" not in row]
            # A candidate in the same contest but absent from a particular
            # precinct is not known to have zero votes; keep NULL.
            if absent_ids:
                row["_flags"].add("MISSING_CANDIDATE_PRECINCT")
            for m in methods:
                vals = [row.get(f"{labels[oid]} - {m}") for oid in ordered_ids]
                row[f"{m} Total"] = sum(vals) if all(type(v) is int for v in vals) else None
            totals = [row.get(f"{labels[oid]} - Total Votes") for oid in ordered_ids]
            row["Grand Total"] = sum(totals) if all(type(v) is int for v in totals) else None
            if row["Grand Total"] is not None:
                complete_method_totals = [row.get(f"{m} Total") for m in methods_seen]
                if all(type(v) is int for v in complete_method_totals) and sum(complete_method_totals) != row["Grand Total"]:
                    row["_flags"].add("PRECINCT_METHOD_MISMATCH")
            row["Discrepancy Flag"] = " | ".join(sorted(row.pop("_flags")))
            absent_methods = [m for m in methods if row[f"{m} Total"] is None]
            row["Missing Fields"] = " | ".join(absent_methods)
            rows.append({h: row.get(h) for h in headers})
    rows.sort(key=lambda r: (r["County ID"], (0, int(r["Precinct ID"])) if r["Precinct ID"].isdigit() else (1, r["Precinct ID"])))
    # Summary is not a precinct and must be marked as such.
    summary: dict[str, Any] = {h: None for h in headers}
    summary["Precinct"] = "GRAND TOTAL (SUMMARY; NOT A PRECINCT)"
    for h in headers:
        if h.endswith(" - Total Votes") or h.endswith(" Total") or h == "Grand Total":
            vals = [r.get(h) for r in rows]
            summary[h] = sum(vals) if vals and all(type(v) is int for v in vals) else None
    summary["Discrepancy Flag"] = "SUMMARY_NOT_PRECINCT"
    rows.append(summary)
    meta = {
        "format_family": "enhanced_voting_precinct_json_v1",
        "state": source_state, "election_date": election_date,
        "snapshot_created_at": snapshot_at, "contest_id": contest_id,
        "contest_name": title, "county": county if county is not None else "ALL_MATCHING_COUNTIES",
        "county_count": len(selected_contests),
        "precinct_rows": len(rows) - 1,
        "candidate_ids": ordered_ids, "candidate_labels": labels,
        "source_candidate_parties_by_county": candidate_parties,
        "source_methods": sorted(methods_seen),
        "unreported_standard_methods": missing_in_feed,
        "issue_count": len(issues), "issues": issues[:250],
        "issue_overflow": max(0, len(issues) - 250),
        "missing_value_semantics": "None means unreported/not supplied, never zero; method absent from source stays None",
        "reporting_percent_semantics": "county-contest aggregate repeated for each precinct, separate from reportingStatus",
        "source_sha256": source_sha256, "source_link": source_link,
        "summary_row_explicitly_non_precinct": True,
        "source_approval_authority": "NOT_INFERRED_FROM_JSON_SHAPE",
    }
    return PrecinctTable(headers, rows, meta, title, source_state, county or "")

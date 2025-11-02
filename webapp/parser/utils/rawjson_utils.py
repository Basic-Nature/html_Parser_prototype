from __future__ import annotations

import os
from typing import Iterable

import orjson

__all__ = [
    "_rj_first",
    "_rj_as_dict",
    "_rj_ensure_list",
    "_infer_party_from_name",
    "extract_rawjson_enrichment_from_rows",
    "offload_rawjson_to_ndjson",
]

def _rj_first(obj: dict, *keys: Iterable[str]):
    """Return first present key from obj by trying provided alternatives."""
    for k in keys:
        if isinstance(k, (list, tuple)):
            for kk in k:
                if isinstance(obj, dict) and kk in obj:
                    return obj[kk]
        else:
            if isinstance(obj, dict) and k in obj:
                return obj[k]
    return None

def _rj_as_dict(raw):
    """Coerce RawJSON value to dict if it's a JSON string; else pass dict through."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return orjson.loads(raw)
        except Exception:
            try:
                import json
                return json.loads(raw)
            except Exception:
                return None
    return None

def _rj_ensure_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]

def _infer_party_from_name(label: str | None) -> str | None:
    if not isinstance(label, str):
        return None
    low = label.lower()
    for p in ("democratic", "democrat", "republican", "working families", "conservative", "libertarian", "green"):
        if p in low:
            return p.title()
    return None

def extract_rawjson_enrichment_from_rows(rows: list[dict]) -> dict | None:
    """
    Build a contest- and candidate-level enrichment object from any RawJSON blobs in the rows.
    Handles common naming variants across feeds.
    """
    if not rows:
        return None
    # Find first non-empty RawJSON
    rj_dict = None
    for r in rows:
        raw = r.get("RawJSON")
        rj_dict = _rj_as_dict(raw)
        if rj_dict:
            break
    if not rj_dict:
        return None

    # Contest-level fields
    contest_name = _rj_first(rj_dict, "name", "contest", "contest_name", "title")
    contest_id = _rj_first(rj_dict, "id", "contestId", "contest_id")
    contest_type = _rj_first(rj_dict, "contestType", "contest_type", "type")
    vote_for = _rj_first(rj_dict, "voteFor", "vote_for", "seats", "maxVotes")
    ballot_order = _rj_first(rj_dict, "ballotOrder", "ballot_order")
    precincts_part = _rj_first(rj_dict, "precinctsParticipating", "precincts_participating", "precinctsTotal", "precincts_total")
    precincts_rep = _rj_first(rj_dict, "precinctsReporting", "precincts_reporting", "precinctsReported", "precincts_reported")

    # Candidates array
    candidates = _rj_first(
        rj_dict,
        "ballotOptions", "ballot_options", "candidates", "options", "ballots"
    ) or []
    candidates = candidates if isinstance(candidates, list) else []

    enr_candidates = []
    group_totals = {}  # sum over candidates
    grand_total = 0

    for c in candidates:
        if not isinstance(c, dict):
            continue
        cid = _rj_first(c, "id", "candidateId", "candidate_id")
        name = _rj_first(c, "name", "label", "candidate", "candidateName", "candidate_name") or ""
        party = _rj_first(c, "politicalParty", "party", "partyName") or _infer_party_from_name(name)
        total = _rj_first(c, "voteCount", "votes", "totalVotes", "total", "reportedVotes") or 0
        try:
            total = int(total)
        except Exception:
            try:
                total = int(float(total))
            except Exception:
                total = 0
        grand_total += total

        # Group results
        groups = _rj_first(c, "groupResults", "group_results", "groups") or []
        gmap = {}
        if isinstance(groups, list):
            for g in groups:
                if not isinstance(g, dict):
                    continue
                gname = _rj_first(g, "groupName", "name", "group", "label")
                gval = _rj_first(g, "voteCount", "votes", "total", "count")
                if gname is None:
                    continue
                try:
                    gval = int(gval or 0)
                except Exception:
                    try:
                        gval = int(float(gval or 0))
                    except Exception:
                        gval = 0
                gmap[str(gname)] = gval
                group_totals[str(gname)] = group_totals.get(str(gname), 0) + gval

        enr_candidates.append({
            "id": cid,
            "label": name,
            "party": party,
            "total_votes_reported": total,
            "group_breakdown": gmap
        })

    # Precinct stats (lightweight)
    precincts = _rj_first(rj_dict, "precinctResults", "precincts", "precinct_results")
    precinct_count = len(precincts) if isinstance(precincts, list) else 0

    # Percent reporting if possible
    contest_reporting_percent = None
    try:
        if isinstance(precincts_rep, (int, float)) and isinstance(precincts_part, (int, float)) and precincts_part:
            contest_reporting_percent = round(precincts_rep / float(precincts_part) * 100.0, 3)
    except Exception:
        pass

    # Group percent distribution
    total_group_votes = sum(v for v in group_totals.values() if isinstance(v, (int, float)))
    group_percent_distribution = {}
    if total_group_votes:
        for g, v in group_totals.items():
            if isinstance(v, (int, float)):
                group_percent_distribution[g] = round(v / total_group_votes * 100.0, 3)

    enrichment = {
        "contest_id": contest_id,
        "contest_name": contest_name,
        "contest_type": contest_type,
        "vote_for": vote_for,
        "ballot_order": ballot_order,
        "precincts_participating": precincts_part,
        "precincts_reporting": precincts_rep,
        "contest_reporting_percent": contest_reporting_percent,
        "candidate_count": len(enr_candidates),
        "candidates": enr_candidates,
        "group_totals": group_totals,
        "group_percent_distribution": group_percent_distribution,
        "grand_total_candidate_votes": grand_total,
        "precincts_listed": precinct_count,
    }
    enrichment_slim = {
        "contest_reporting_percent": contest_reporting_percent,
        "candidate_count": len(enr_candidates),
        "groups_present": list(group_totals.keys()),
    }
    return {"extended": enrichment, "slim": enrichment_slim}

def offload_rawjson_to_ndjson(rows: list[dict], out_dir: str, structure_hash: str | None = None) -> tuple[list[dict], str | None]:
    """
    Write RawJSON blobs to NDJSON and replace row RawJSON with (RawJSONPath, RawId) pointer.
    Returns (rows_modified, ndjson_path or None).
    """
    if not rows:
        return rows, None
    has_any = any(isinstance(r, dict) and r.get("RawJSON") not in (None, "") for r in rows)
    if not has_any:
        return rows, None
    try:
        os.makedirs(os.path.join(out_dir, "_rawjson"), exist_ok=True)
        ndjson_path = os.path.join(out_dir, "_rawjson", f"{(structure_hash or 'raw')}.ndjson")
        next_id = 1
        if os.path.exists(ndjson_path):
            try:
                next_id = sum(1 for _ in open(ndjson_path, "rb")) + 1
            except Exception:
                pass
        with open(ndjson_path, "ab") as f:
            out_rows = []
            for r in rows:
                if not isinstance(r, dict):
                    out_rows.append(r)
                    continue
                raw = r.pop("RawJSON", None)
                if raw in (None, ""):
                    out_rows.append(r)
                    continue
                blob = _rj_as_dict(raw) or raw
                rid = r.get("RawId") or str(next_id)
                try:
                    f.write(orjson.dumps({"id": rid, "raw": blob}) + b"\n")
                    next_id += 1
                except Exception:
                    pass
                r["RawJSONPath"] = ndjson_path
                r["RawId"] = rid
                out_rows.append(r)
        return out_rows, ndjson_path
    except Exception:
        return rows, None
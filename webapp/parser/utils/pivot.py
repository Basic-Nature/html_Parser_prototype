"""
pivot.py
Wide table pivot with:
- Detector integration
- Fast path for already-wide tables
- Strict header ordering (by candidate total desc then alpha)
- Party column inclusion (Candidate - Party)
- Per-candidate % Vote (recomputed)
- Cumulative running Vote / % (Candidate - Cum Vote / Candidate - Cum %)
- Division Type column via STATE_TO_DIVISION_TYPE_MAP
- Optional aggregate "All Precincts" summary row

Context / flags (defaults shown):
  strict_header_order: True
  include_party_in_wide: True
  include_candidate_percent: True
  include_cumulative_columns: True
  include_all_precincts_row: True
  include_division_type_column: True
  state: (two-letter or full name; used to derive division type)
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional, Set
import re
import orjson
from .logger_singleton import logger
from .detect import (
    dynamic_detect_location_header,
    normalize_header,
    parse_numeric
)
from .shared_logic import safe_get, safe_strip
from ..Context_Integration.Context_Library.constants import (
    BALLOT_TYPES_SORT_ORDER,
    BALLOT_TYPES,
    TOTAL_KEYWORDS,
    STATE_TO_DIVISION_TYPE_MAP,
    BALLOT_NAME_CANON_MAP,
    BALLOT_GROUP_CANON_ORDER,             # (if future pivot modes need it)
    canonical_ballot_group,
    normalize_party_label,
    CANDIDATE_BALLOT_SPLIT_PATTERN,
    DIVISION_SUFFIXES,
    DIVISION_HEURISTIC_TERMS
)

_CAND_BT_RE = re.compile(CANDIDATE_BALLOT_SPLIT_PATTERN, re.UNICODE)
_TOTAL_LIKE = {normalize_header(t) for t in TOTAL_KEYWORDS} | {
    normalize_header("grand total"),
    normalize_header("total vote")
}

# --- Added natural sort + division heuristics (place here with other global regex/constants) ---
_SPLIT_NUM_RE = re.compile(r'(\d+)')

def _natural_key(s: str):
    if s is None:
        return []
    if s.lower().strip() in ("all precincts", "all districts", "total", "overall"):
        return [float('inf')]
    parts = _SPLIT_NUM_RE.split(s.lower())
    key = []
    for p in parts:
        key.append(int(p) if p.isdigit() else p)
    return key

def _sort_precincts(precincts: List[str], context: dict):
    mode = context.get("precinct_sort", "natural")
    aggregate_last = context.get("aggregate_last", True)
    aggs = {"all precincts", "all districts", "total", "overall"}
    if mode == "alpha":
        ordered = sorted(precincts, key=lambda x: x.lower())
    elif mode == "numeric":
        if all(p.isdigit() for p in precincts if p):
            ordered = sorted(precincts, key=lambda x: int(x))
        else:
            ordered = sorted(precincts, key=_natural_key)
    else:
        ordered = sorted(precincts, key=_natural_key)
    if aggregate_last and any(p.lower() in aggs for p in ordered):
        core = [p for p in ordered if p.lower() not in aggs]
        agg = [p for p in ordered if p.lower() in aggs]
        return core + agg
    return ordered

def _infer_division_type_by_suffix(original: str) -> str:
    if not original:
        return ""
    low = original.lower().strip()
    for term, dtype in DIVISION_HEURISTIC_TERMS:
        if f" {term}" in f" {low} " or low.endswith(term):
            return dtype
    return ""

def _numeric_ratio(values) -> float:
    non = [v for v in values if v not in ("", None)]
    if not non:
        return 0.0
    num = 0
    for v in non:
        s = str(v).replace(",", "").replace("%", "").strip()
        if s.replace(".", "", 1).lstrip("-").isdigit():
            num += 1
    return num / len(non)

def _is_numeric_column(h: str, data: List[Dict[str, Any]]) -> bool:
    vals = [safe_get(r, h, "") for r in data]
    return _numeric_ratio(vals) >= 0.5

def _fast_path_already_wide(headers: List[str], data: List[Dict[str, Any]]):
    candidate_map: Dict[str, Set[str]] = {}
    has_pattern = False
    for h in headers:
        if h in ("Precinct", "Percent Reported", "Grand Total", "Division Type"):
            continue
        m = _CAND_BT_RE.match(h)
        if not m:
            continue
        cand = m.group("cand").strip()
        bt = m.group("bt").strip()
        if cand and bt:
            candidate_map.setdefault(cand, set()).add(bt)
            has_pattern = True
    if not has_pattern or len(candidate_map) < 1:
        return None
    pattern_cols = [h for h in headers if _CAND_BT_RE.match(h)]
    if not pattern_cols:
        return None
    numeric_like = sum(1 for h in pattern_cols if _is_numeric_column(h, data))
    if numeric_like == 0:
        return None
    needs_total = "Grand Total" not in headers
    if needs_total:
        for row in data:
            gt = 0
            for h in pattern_cols:
                v = row.get(h)
                if isinstance(v, (int, float)):
                    gt += int(v)
                elif isinstance(v, str):
                    s = v.replace(",", "").strip()
                    if s.isdigit():
                        gt += int(s)
            row["Grand Total"] = gt
        headers = headers + ["Grand Total"]
    logger.info(f"[PIVOT] Fast path wide detected: candidates={len(candidate_map)} pattern_cols={len(pattern_cols)}")
    return headers, data

def debug_dump_pivot_state(tag: str, headers: List[str], data: List[Dict[str, Any]], limit: int = 3):
    sample = data[:limit]
    logger.debug({"pivot_debug": tag, "headers": headers, "row_samples": sample})

def _normalize_candidate_label(raw: str) -> str:
    """
    Strip leading short uppercase party token (<=4 chars) & collapse whitespace.
    DEM Jane Doe -> Jane Doe
    WOR Jane Doe -> Jane Doe
    """
    if not raw:
        return raw
    parts = raw.split()
    if parts and len(parts[0]) <= 4 and parts[0].isupper():
        core = " ".join(parts[1:]).strip()
        return core or raw
    return raw

def _collect_ballot_types(headers: List[str], data: List[Dict[str, Any]], detector=None, candidate_col: Optional[str] = None):
    if detector:
        bt = detector.detect_ballot_types(headers, data)
        if bt:
            ordered = [b for b in BALLOT_TYPES_SORT_ORDER if b in bt]
            for b in bt:
                if b not in ordered:
                    ordered.append(b)
            return ordered
    bt_set = set()
    norm_sorted = [normalize_header(bt2) for bt2 in BALLOT_TYPES_SORT_ORDER]
    for h in headers:
        nh = normalize_header(h)
        if nh in norm_sorted:
            bt_set.add(h)
    if not bt_set:
        for h in headers:
            if h in ("Precinct", "Percent Reported", candidate_col, "Division Type", "Party"):
                continue
            nh = normalize_header(h)
            if nh in _TOTAL_LIKE:
                continue
            if any(b.lower() == h.lower() for b in BALLOT_TYPES):
                bt_set.add(h)
                continue
            if _is_numeric_column(h, data):
                bt_set.add(h)
    ordered = [b for b in BALLOT_TYPES_SORT_ORDER if b in bt_set]
    for b in sorted(bt_set):
        if b not in ordered:
            ordered.append(b)
    return ordered

def _derive_party_map(candidate_col: Optional[str], data: List[Dict[str, Any]]) -> Dict[str, str]:
    if not candidate_col:
        return {}
    party_map: Dict[str, str] = {}
    for r in data:
        cand = safe_strip(safe_get(r, candidate_col, "")) or ""
        if not cand:
            continue
        party = safe_strip(safe_get(r, "Party", "")) or ""
        if not party:
            continue
        party_map.setdefault(cand, party)
    return party_map

def _normalize_division_name(name: str) -> str:
    n = name.strip().lower()
    for suf in DIVISION_SUFFIXES:
        if n.endswith(suf):
            n = n.removesuffix(suf)
            break
    return n

def _division_type_for(division: str, state: str | None) -> str:
    if not division:
        return ""
    if state:
        s = state.strip().lower()
        div_norm = _normalize_division_name(division)
        state_map = STATE_TO_DIVISION_TYPE_MAP.get(s)
        if state_map:
            t = state_map.get(div_norm)
            if t:
                return t
            t2 = state_map.get(division.strip().lower())
            if t2:
                return t2
    return _infer_division_type_by_suffix(division)

def pivot_to_wide(
    headers: List[str],
    data: List[Dict[str, Any]],
    entity_info: Dict[str, Any],
    coordinator=None,
    context: dict | None = None
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Enhanced wide pivot:
      - Cross-endorsement merge (merge_cross_endorsements)
      - Robust ballot type consolidation
      - Local + cumulative % calculations
      - Natural precinct sort
      - Fallback if already wide
      - Party inference preserved when labels normalized
    """
    context = context or {}
    headers = headers or []
    data = data or []
    detector = (context.get("detector") or entity_info.get("detector")) if entity_info else context.get("detector")

    # Flags
    strict_order = context.get("strict_header_order", True)
    include_party = context.get("include_party_in_wide", True)
    include_candidate_pct = context.get("include_candidate_percent", True)
    include_cumulative = context.get("include_cumulative_columns", True)
    include_all_row = context.get("include_all_precincts_row", True)
    include_division_type = context.get("include_division_type_column", True)
    merge_cross = context.get("merge_cross_endorsements", True)

    state = (context.get("state") or entity_info.get("state") if entity_info else None)
    if state:
        state = state.lower()

    # Location / percent headers
    location_header = (entity_info or {}).get("location_column") or (entity_info or {}).get("location_header")
    percent_header = (entity_info or {}).get("percent_column") or (entity_info or {}).get("percent_header")

    if coordinator and (not location_header or not percent_header):
        det_loc, det_pct, _ = dynamic_detect_location_header(headers, coordinator)
        location_header = location_header or det_loc
        percent_header = percent_header or det_pct

    # Ensure Precinct column
    if not location_header:
        location_header = "Precinct"
        default_loc = safe_get(context, "location_value", "") or safe_get(context, "contest", "") or "All"
        for r in data:
            if "Precinct" not in r:
                r["Precinct"] = default_loc
    if location_header != "Precinct":
        headers = ["Precinct" if h == location_header else h for h in headers]
        for r in data:
            r["Precinct"] = r.pop(location_header, r.get("Precinct", "")) or r.get("Precinct", "")
        location_header = "Precinct"

    # Normalize percent header
    if percent_header and normalize_header(percent_header) != normalize_header("Percent Reported"):
        for r in data:
            if percent_header in r and "Percent Reported" not in r:
                r["Percent Reported"] = r.pop(percent_header)
        headers = ["Percent Reported" if h == percent_header else h for h in headers]
        percent_header = "Percent Reported"

    # Fast path if already wide
    fast = _fast_path_already_wide(headers, data)
    if fast:
        wh, wd = fast
        if include_division_type and "Division Type" not in wh:
            for row in wd:
                row["Division Type"] = _division_type_for(row.get("Precinct", ""), state)
            wh.insert(1, "Division Type")
        if include_all_row:
            summary = {"Precinct": "All Precincts"}
            if "Percent Reported" in wh:
                summary["Percent Reported"] = ""
            if "Division Type" in wh:
                summary["Division Type"] = "aggregate"
            for h in wh:
                if h in ("Precinct", "Percent Reported", "Division Type"):
                    continue
                if h.endswith("% Vote"):
                    summary[h] = ""
                    continue
                total = 0
                for r in wd:
                    iv, _ = parse_numeric(r.get(h, ""))
                    if iv is not None:
                        total += iv
                summary[h] = str(total)
            wd.append(summary)
        return wh, wd

    # ---------------- Candidate column detection ----------------
    candidate_col = None
    if detector:
        try:
            candidate_col = detector.detect_candidate_column(headers, data)
        except Exception:
            candidate_col = None
    if not candidate_col:
        for h in headers:
            if normalize_header(h) == "candidate":
                candidate_col = h
                break
    if not candidate_col:
        # heuristic: first non-Precinct, non-percent, non-total textual column
        for h in headers:
            hn = normalize_header(h)
            if h in ("Precinct", "Percent Reported", "Division Type", "Party"):
                continue
            if hn in _TOTAL_LIKE:
                continue
            candidate_col = h
            break

    debug_dump_pivot_state("pre_wide", headers, data, limit=2)

    # If still none & pattern columns exist treat as already wide
    if not candidate_col and any(_CAND_BT_RE.match(h) for h in headers):
        fast_wide = _fast_path_already_wide(headers, data)
        if fast_wide:
            wh, wd = fast_wide
            return wh, wd

    # ---------------- Aggregate candidates (with cross endorsement normalization) -------------
    candidates_totals: Dict[str, int] = {}
    raw_to_norm: Dict[str, str] = {}
    if candidate_col:
        for r in data:
            raw_c = safe_strip(safe_get(r, candidate_col, "")) or ""
            if not raw_c:
                continue
            norm_c = _normalize_candidate_label(raw_c) if merge_cross else raw_c
            raw_to_norm[raw_c] = norm_c
            row_total = 0
            for k, v in r.items():
                if k in (candidate_col, "Precinct", "Percent Reported", "Party", "Division Type"):
                    continue
                nh = normalize_header(k)
                if nh in _TOTAL_LIKE or any(bt.lower() in k.lower() for bt in BALLOT_TYPES):
                    if isinstance(v, (int, float)):
                        row_total += int(v)
                    elif isinstance(v, str):
                        sv = v.replace(",", "").replace("%", "").strip()
                        if sv.isdigit():
                            row_total += int(sv)
            candidates_totals[norm_c] = candidates_totals.get(norm_c, 0) + row_total

    ballot_types = _collect_ballot_types(headers, data, detector=detector, candidate_col=candidate_col)
    party_map = _derive_party_map(candidate_col, data) if include_party else {}

    # ---------------- Fallback: unstructured numeric table ----------------
    if not candidates_totals and not ballot_types:
        wide_headers = ["Precinct"]
        if include_division_type:
            wide_headers.append("Division Type")
        if percent_header:
            wide_headers.append("Percent Reported")
        wide_headers.append("Grand Total")
        loc_values = {safe_get(r, "Precinct", "") for r in data if safe_get(r, "Precinct", "")} or {"All"}
        rows_out = []
        for loc in _sort_precincts(list(loc_values), context):
            row = {"Precinct": loc}
            if include_division_type:
                row["Division Type"] = _division_type_for(loc, state)
            if percent_header:
                row["Percent Reported"] = ""
            grand = 0
            for r in data:
                if safe_get(r, "Precinct", "") != loc:
                    continue
                for h in headers:
                    if h in ("Precinct", "Percent Reported", "Division Type"):
                        continue
                    iv, _ = parse_numeric(r.get(h, ""))
                    if iv is not None:
                        grand += iv
                if percent_header and "Percent Reported" in r and not row.get("Percent Reported"):
                    row["Percent Reported"] = r["Percent Reported"]
            row["Grand Total"] = str(grand)
            rows_out.append(row)
        if include_all_row and rows_out and "All Precincts" not in {r["Precinct"] for r in rows_out}:
            agg = {"Precinct": "All Precincts"}
            if include_division_type:
                agg["Division Type"] = "aggregate"
            if percent_header:
                agg["Percent Reported"] = ""
            agg["Grand Total"] = str(sum(int(r["Grand Total"]) for r in rows_out if r.get("Grand Total", "").isdigit()))
            rows_out.append(agg)
        logger.info("[PIVOT] Fallback simple wide applied (no candidates detected).")
        return wide_headers, rows_out

    # ---------------- Order candidates ----------------
    candidate_names = list(candidates_totals.keys())
    if strict_order and candidates_totals:
        candidate_names.sort(key=lambda c: (-candidates_totals.get(c, 0), c.lower()))
    else:
        candidate_names.sort()

    # ---------------- Build header list ----------------
    wide_headers: List[str] = ["Precinct"]
    if include_division_type:
        wide_headers.append("Division Type")
    if percent_header:
        wide_headers.append("Percent Reported")

    for cand in candidate_names:
        # Party header (normalized mapping)
        if include_party:
            party_val = None
            for raw_label, nlabel in raw_to_norm.items():
                if nlabel == cand:
                    party_val = party_map.get(raw_label) or party_map.get(cand)
                    if party_val:
                        break
            if party_val:
                wide_headers.append(f"{cand} - Party")
        for bt in ballot_types:
            wide_headers.append(f"{cand} - {bt}")
        wide_headers.append(f"{cand} - Total Vote")
        if include_candidate_pct:
            wide_headers.append(f"{cand} - % Vote")
        if include_cumulative:
            wide_headers.append(f"{cand} - Cum Vote")
            if include_candidate_pct:
                wide_headers.append(f"{cand} - Cum %")
    wide_headers.append("Grand Total")

    # ---------------- Build rows ----------------
    loc_values = _sort_precincts(
        list({safe_get(r, "Precinct", "") for r in data if safe_get(r, "Precinct", "")} or ["All"]),
        context
    )
    running_totals = {c: 0 for c in candidate_names}
    overall_grand_total = 0
    out_rows: List[Dict[str, Any]] = []

    for loc in loc_values:
        out = {h: "" for h in wide_headers}
        out["Precinct"] = loc
        if include_division_type:
            out["Division Type"] = _division_type_for(loc, state) or ""
        if percent_header and "Percent Reported" in wide_headers:
            for r in data:
                if safe_get(r, "Precinct", "") == loc and safe_get(r, percent_header, ""):
                    out["Percent Reported"] = r.get(percent_header)
                    break

        grand_total_loc = 0
        cand_totals_local: Dict[str, int] = {}

        # Accumulate from each raw row for this precinct
        for r in data:
            if safe_get(r, "Precinct", "") != loc:
                continue
            raw_c = safe_strip(safe_get(r, candidate_col, "")) if candidate_col else ""
            if not raw_c:
                continue
            norm_c = raw_to_norm.get(raw_c, raw_c)
            if norm_c not in candidate_names:
                continue
            cand_row_total = 0
            # ballot type values
            for bt in ballot_types:
                val = r.get(bt)
                if val in (None, "", "-"):
                    # approximate header match
                    for k, v in r.items():
                        if normalize_header(k) == normalize_header(bt):
                            val = v
                            break
                vc = 0
                if isinstance(val, (int, float)):
                    vc = int(val)
                elif isinstance(val, str):
                    sv = val.replace(",", "").strip()
                    if sv.isdigit():
                        vc = int(sv)
                if vc:
                    out[f"{norm_c} - {bt}"] = out.get(f"{norm_c} - {bt}", 0) + vc
                    cand_row_total += vc

            # Explicit total overrides
            total_cell = None
            for k, v in r.items():
                if normalize_header(k) in _TOTAL_LIKE:
                    total_cell = v
                    break
            if total_cell not in (None, "", "-"):
                sv = str(total_cell).replace(",", "").strip()
                if sv.isdigit():
                    cand_row_total = int(sv)

            out[f"{norm_c} - Total Vote"] = out.get(f"{norm_c} - Total Vote", 0) + cand_row_total
            cand_totals_local[norm_c] = cand_totals_local.get(norm_c, 0) + cand_row_total
            grand_total_loc += cand_row_total

            if include_party:
                party_val = party_map.get(raw_c) or party_map.get(norm_c)
                if party_val and f"{norm_c} - Party" in out and not out[f"{norm_c} - Party"]:
                    out[f"{norm_c} - Party"] = party_val

        # Local % per candidate
        if include_candidate_pct and grand_total_loc:
            for cand in candidate_names:
                ct = cand_totals_local.get(cand, 0)
                out[f"{cand} - % Vote"] = f"{(ct / grand_total_loc) * 100:.2f}%"

        # Cumulative
        if include_cumulative:
            overall_grand_total += grand_total_loc
            for cand in candidate_names:
                running_totals[cand] += cand_totals_local.get(cand, 0)
                out[f"{cand} - Cum Vote"] = running_totals[cand]
                if include_candidate_pct and overall_grand_total:
                    out[f"{cand} - Cum %"] = f"{(running_totals[cand] / overall_grand_total) * 100:.2f}%"

        out["Grand Total"] = grand_total_loc
        out_rows.append(out)

    # Aggregate row
    if include_all_row and out_rows:
        agg = {h: "" for h in wide_headers}
        agg["Precinct"] = "All Precincts"
        if include_division_type:
            agg["Division Type"] = "aggregate"
        grand_total_all = 0
        for cand in candidate_names:
            col = f"{cand} - Total Vote"
            total = sum(
                int(r[col]) if isinstance(r.get(col), int)
                else int(str(r.get(col)).replace(",", "")) if str(r.get(col, "")).replace(",", "").isdigit()
                else 0
                for r in out_rows
            )
            agg[col] = total
            grand_total_all += total
            if include_candidate_pct:
                pct = (total / grand_total_all * 100) if grand_total_all else 0
                agg[f"{cand} - % Vote"] = f"{pct:.2f}%"
            if include_cumulative:
                agg[f"{cand} - Cum Vote"] = total
                if include_candidate_pct:
                    agg[f"{cand} - Cum %"] = agg[f"{cand} - % Vote"]
            party_col = f"{cand} - Party"
            if party_col in wide_headers:
                for r in out_rows:
                    if r.get(party_col):
                        agg[party_col] = r[party_col]
                        break
            for bt in ballot_types:
                bt_col = f"{cand} - {bt}"
                bt_sum = 0
                for r in out_rows:
                    v = r.get(bt_col)
                    if isinstance(v, int):
                        bt_sum += v
                    elif isinstance(v, str) and v.replace(",", "").isdigit():
                        bt_sum += int(v.replace(",", ""))
                agg[bt_col] = bt_sum
        agg["Grand Total"] = grand_total_all
        out_rows.append(agg)

    logger.info(f"[PIVOT] wide rows={len(out_rows)} cols={len(wide_headers)} candidates={len(candidate_names)} bt={len(ballot_types)}")
    if not candidate_names:
        logger.warning("[PIVOT] No candidates detected – verify headers and candidate column extraction.")
    return wide_headers, out_rows

# ---------------- RawJSON candidate groups pivot fix -----------------

def pivot_candidate_groups_from_rawjson(
    headers: List[str],
    data: List[Dict[str, Any]],
    context: dict | None = None,
    drop_rawjson: bool = True
) -> tuple[List[str] | None, List[Dict[str, Any]] | None]:
    from .detect import emit_metric
    context = context or {}
    if "RawJSON" not in headers:
        return None, None

    raw_payload = None
    for row in data:
        val = row.get("RawJSON")
        if isinstance(val, str) and val.strip().startswith("{"):
            raw_payload = val
            break
    if not raw_payload:
        return None, None

    try:
        contest = orjson.loads(raw_payload)
    except Exception:
        return None, None

    precincts_participating = contest.get("precinctsParticipating")
    precincts_reporting = contest.get("precinctsReporting")
    contest_reporting_percent = None
    try:
        if precincts_participating and precincts_reporting is not None and precincts_participating > 0:
            pct_val = (precincts_reporting / precincts_participating) * 100.0
            contest_reporting_percent = f"{pct_val:.1f}".rstrip("0").rstrip(".")
    except Exception:
        pass

    ballot_options = (contest or {}).get("ballotOptions") or []
    if not ballot_options:
        return None, None

    observed_groups = set()
    candidates_meta = []
    candidate_enrichment = []

    for opt in ballot_options:
        raw_name = (opt.get("name") or "").strip()
        party = normalize_party_label(opt.get("politicalParty"))
        short = _normalize_candidate_label(raw_name)
        label = f"{party}: {short}"
        for gr in (opt.get("groupResults") or []):
            observed_groups.add(canonical_ballot_group(gr.get("groupName", "")))
        candidates_meta.append({"label": label, "option": opt, "party": party, "short": short})
        candidate_enrichment.append({
            "label": label,
            "party": party,
            "raw_name": raw_name,
            "total_votes_reported": opt.get("voteCount"),
            "ballot_order": opt.get("ballotOrder"),
            "group_breakdown": {}
        })

    present_groups: list[str] = []
    seen_pg = set()
    for g in BALLOT_GROUP_CANON_ORDER:
        canon = canonical_ballot_group(g)
        if canon in observed_groups and canon not in seen_pg:
            present_groups.append(canon)
            seen_pg.add(canon)

    precinct_rows: Dict[str, Dict[str, Any]] = {}
    group_totals = {g: 0 for g in present_groups}

    for meta in candidates_meta:
        opt = meta["option"]
        label = meta["label"]
        per_cand_group_counts = {g: 0 for g in present_groups}
        for pr in (opt.get("precinctResults") or []):
            pname = pr.get("name") or f"Precinct {pr.get('id')}"
            prow = precinct_rows.setdefault(pname, {"Precinct": pname, "Percent Reported": "", "_cand_totals": {}})
            status = (pr.get("reportingStatus") or "").lower()
            if "fully" in status:
                prow["Percent Reported"] = prow.get("Percent Reported") or "100"
            for gr in (pr.get("groupResults") or []):
                canon_g = canonical_ballot_group(gr.get("groupName", ""))
                if canon_g not in present_groups:
                    continue
                vc = gr.get("voteCount") or 0
                per_cand_group_counts[canon_g] += vc
                col = f"{label} - {canon_g}"
                prow[col] = prow.get(col, 0) + vc
            candidate_total = sum(per_cand_group_counts.values())
            prow[f"{label} - Total Reported"] = prow.get(f"{label} - Total Reported", 0) + candidate_total
            prow["_cand_totals"][label] = prow["_cand_totals"].get(label, 0) + candidate_total
        for g, v in per_cand_group_counts.items():
            group_totals[g] = group_totals.get(g, 0) + v
        for ce in candidate_enrichment:
            if ce["label"] == label:
                ce["group_breakdown"] = per_cand_group_counts
                break

    base_headers = ["Precinct", "Total Ballots Reported", "Percent Reported"]
    candidate_headers = []
    for meta in candidates_meta:
        label = meta["label"]
        for g in present_groups:
            candidate_headers.append(f"{label} - {g}")
        candidate_headers.append(f"{label} - Total Reported")

    dedup = []
    s = set()
    for h in candidate_headers:
        if h not in s:
            s.add(h)
            dedup.append(h)
    final_headers = base_headers + dedup

    out_rows = []
    for prow in precinct_rows.values():
        total_ballots = sum(prow["_cand_totals"].values())
        row_out = {h: "" for h in final_headers}
        row_out["Precinct"] = prow["Precinct"]
        pr_val = prow.get("Percent Reported") or (contest_reporting_percent or "")
        row_out["Percent Reported"] = pr_val
        row_out["Total Ballots Reported"] = total_ballots
        for h in dedup:
            row_out[h] = prow.get(h, 0 if h.endswith("Reported") else 0)
        out_rows.append(row_out)

    out_rows.sort(key=lambda r: r["Precinct"])

    emit_metric("pivot_candidate_groups_rawjson", rows=len(out_rows), cols=len(final_headers))
    context.setdefault("pivot_modes", []).append("candidate_groups_rawjson")

    # hierarchical headers
    try:
        r1, r2 = [], []
        for h in final_headers:
            if h in ("Precinct", "Total Ballots Reported", "Percent Reported"):
                r1.append(h); r2.append("")
            else:
                cand_part, grp_part = h.rsplit(" - ", 1)
                if grp_part == "Total Reported":
                    grp_part = "Total"
                r1.append(cand_part); r2.append(grp_part)
        context["hierarchical_headers"] = {"rows": [r1, r2], "style_hint": "candidate_group_pivot_v1"}
    except Exception:
        pass

    # enrichment
    context["rawjson_enrichment"] = {
        "contest_id": contest.get("id"),
        "contest_name": contest.get("name"),
        "contest_type": contest.get("contestType"),
        "vote_for": contest.get("voteFor"),
        "precincts_participating": precincts_participating,
        "precincts_reporting": precincts_reporting,
        "contest_reporting_percent": contest_reporting_percent,
        "candidate_count": len(candidates_meta),
        "candidates": candidate_enrichment,
        "ballot_groups_present": present_groups,
        "group_totals": group_totals
    }

    if context.get("rawjson_include_all_precincts", True) and out_rows:
        agg = {h: "" for h in final_headers}
        agg["Precinct"] = "All Precincts"
        if contest_reporting_percent:
            agg["Percent Reported"] = contest_reporting_percent
        tb_tot = 0
        for r in out_rows:
            tb_tot += r.get("Total Ballots Reported", 0) if isinstance(r.get("Total Ballots Reported", 0), int) else int(str(r.get("Total Ballots Reported")).replace(",", "") or 0)
        agg["Total Ballots Reported"] = tb_tot
        for h in dedup:
            if h.endswith("Total Reported") or " - " in h:
                agg[h] = sum(
                    rr.get(h, 0) if isinstance(rr.get(h, 0), int)
                    else int(str(rr.get(h, 0)).replace(",", "") or 0)
                    for rr in out_rows
                )
        out_rows.append(agg)

    return final_headers, out_rows
"""
salvage.py
Row/column salvage, merging, RawJSON flatten, footer pruning.
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import re, json, orjson
from .shared_logic import (
    safe_get, safe_values, safe_keys, safe_pop, safe_append,
    safe_strip, safe_lower, safe_replace, safe_items
)
from ..Context_Integration.Context_Library.constants import (
    RAWJSON_COLUMN_ALIASES, BALLOT_TYPES_SORT_ORDER, LOCATION_ABBREVIATIONS,
    LOCATION_KEYWORDS, GROUP_RENAME_MAP, CANDIDATE_VALUE_KEYS,
    TOTAL_VALUE_KEYS, PERCENT_KEYWORDS, BALLOT_TYPES, TOTAL_KEYWORDS,
    MISC_FOOTER_KEYWORDS, PARTY_KEYWORDS
)
from .logger_singleton import logger
from .detect import normalize_text

def merge_multiline_candidate_rows(headers, data):
    if "Candidate" not in headers:
        return headers, data
    has_precinct = "Precinct" in headers or any("Precinct" in safe_keys(r) for r in data)
    has_percent = "Percent Reported" in headers or any("Percent Reported" in safe_keys(r) for r in data)
    if has_precinct and "Precinct" not in headers:
        headers.append("Precinct")
    if has_percent and "Percent Reported" not in headers:
        headers.append("Percent Reported")
    out=[]
    i=0
    while i < len(data):
        row = data[i]
        cand = safe_get(row,"Candidate","")
        if "\n" in cand:
            parts=[p.strip() for p in cand.split("\n") if p.strip()]
            if len(parts)==2:
                row["Candidate"], row["Party"]=parts
            elif len(parts)>2:
                row["Candidate"]=parts[0]; row["Party"]=" ".join(parts[1:])
        out.append(row)
        i+=1
    if any("Party" in safe_keys(r) for r in out) and "Party" not in headers:
        headers.append("Party")
    return headers, out

def combine_panel_tables_by_precinct(tables: List[Tuple[List[str], List[Dict[str,Any]]]]):
    hdrs=set(); rows=[]
    for h,d in tables:
        hdrs.update(h); rows.extend(d)
    return list(hdrs), rows

def remove_footer_and_summary_rows(rows, headers):
    total_cols=[h for h in headers if any(k in h.lower() for k in TOTAL_KEYWORDS|MISC_FOOTER_KEYWORDS)]
    out=[]
    for r in rows:
        vals=list(safe_values(r))
        if not any(v not in ("",None) for v in vals):
            continue
        rm=False
        for tc in total_cols:
            v=safe_get(r,tc,"")
            if any(k in str(v).lower() for k in TOTAL_KEYWORDS|MISC_FOOTER_KEYWORDS):
                rm=True; break
        if not rm:
            out.append(r)
    return out

def remove_outlier_and_empty_rows(rows, min_non_empty=2):
    out=[]
    for r in rows:
        vals=list(safe_values(r))
        non=[v for v in vals if v not in ("",None)]
        if len(non)>=min_non_empty:
            out.append(r)
    return out

def _coerce_json(v):
    if isinstance(v,dict): return v
    if isinstance(v,str):
        for loader in (orjson.loads, json.loads):
            try: return loader(v)
            except Exception: continue
    return None

def _salvage_rows_from_rawjson(headers: List[str], data: List[Dict[str,Any]]):
    raw_col = next((h for h in headers if normalize_text(h) in RAWJSON_COLUMN_ALIASES), None)
    if not raw_col:
        return headers,data
    norm_bt = {normalize_text(bt): bt for bt in BALLOT_TYPES_SORT_ORDER}
    norm_pct = {normalize_text(x) for x in PERCENT_KEYWORDS}
    norm_loc = {normalize_text(x) for x in LOCATION_KEYWORDS}
    abbrev = set(LOCATION_ABBREVIATIONS.keys())

    def to_num(x):
        try:
            s=str(x).replace(",","").replace("%","").strip()
            return float(s) if s.replace(".", "",1).lstrip("-").isdigit() else None
        except Exception:
            return None

    flat=[]
    for row in data:
        blob=_coerce_json(row.get(raw_col))
        if not blob: continue
        stack=[blob]
        loc_candidates=[]; percent_value=""
        items=[]
        while stack:
            cur=stack.pop()
            if isinstance(cur,dict):
                nk={k:normalize_text(k) for k in cur}
                for k,kn in nk.items():
                    if kn in norm_loc or kn in abbrev or kn.endswith(" id") or kn.endswith(" name"):
                        sval=str(cur.get(k,"")).strip()
                        if sval: loc_candidates.append(sval)
                for k,kn in nk.items():
                    v=cur.get(k,"")
                    if kn in norm_pct and isinstance(v,(str,int,float)):
                        sv=str(v)
                        if "%" in sv or "reported" in sv.lower():
                            percent_value=sv
                    elif isinstance(v,str) and "%" in v and any(p in kn for p in norm_pct):
                        percent_value=v
                cand_key = next((k for k,kn in nk.items() if kn in CANDIDATE_VALUE_KEYS or "candidate" in kn or kn=="name"), None)
                cand = cur.get(cand_key,"").strip() if cand_key and isinstance(cur.get(cand_key),str) else ""
                party_key = next((k for k in cur if any(pk in k.lower() for pk in PARTY_KEYWORDS)), None)
                party_val = cur.get(party_key,"").strip() if party_key and isinstance(cur.get(party_key),str) else ""
                if cand:
                    bt_map={}; total_seen=None
                    for k,v in cur.items():
                        kn=normalize_text(k)
                        if kn in GROUP_RENAME_MAP:
                            num=to_num(v)
                            if num is not None:
                                bt_map[GROUP_RENAME_MAP[kn]]=bt_map.get(GROUP_RENAME_MAP[kn],0)+num
                        elif kn in norm_bt:
                            num=to_num(v)
                            if num is not None:
                                bt_map[norm_bt[kn]]=bt_map.get(norm_bt[kn],0)+num
                        if kn in TOTAL_VALUE_KEYS or "total" in kn:
                            num=to_num(v)
                            if num is not None:
                                total_seen=max(total_seen or 0,num)
                    if bt_map or total_seen is not None:
                        items.append((cand,bt_map,party_val))
                for v in cur.values():
                    if isinstance(v,(dict,list)): stack.append(v)
            elif isinstance(cur,list):
                for v in cur:
                    if isinstance(v,(dict,list)): stack.append(v)
        if items:
            loc_val = loc_candidates[-1] if loc_candidates else ""
            for cand, bts, party in items:
                out={"Precinct": loc_val or row.get("Precinct","") or row.get("_heading","") or "All",
                     "Candidate": cand}
                if party: out["Party"]=party
                if percent_value: out["Percent Reported"]=percent_value
                total=0.0
                for bt,val in bts.items():
                    out[bt]=str(int(val)) if float(val).is_integer() else str(val)
                    total+=val
                out["Total Vote"]=str(int(total)) if float(total).is_integer() else str(total)
                flat.append(out)
    if not flat:
        return headers,data
    all_headers=set(["Precinct","Candidate"])
    for r in flat: all_headers.update(r.keys())
    bt_order=[bt for bt in BALLOT_TYPES_SORT_ORDER if bt in all_headers]
    leading=["Precinct","Candidate"]
    if "Party" in all_headers: leading.append("Party")
    if "Percent Reported" in all_headers: leading.append("Percent Reported")
    others=[h for h in all_headers if h not in set(leading)|set(bt_order)|{"Total Vote"}]
    new_headers=leading+bt_order+["Total Vote"]+sorted(others)
    return new_headers, flat
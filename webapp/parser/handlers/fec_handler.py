from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional, Tuple

from webapp.parser.fec_lookup import find_candidate_by_name, get_candidate_by_id
from webapp.parser.utils.fec_utils import (
    canonicalize_headers,
    date_normalize,
    incumbent_normalize,
    money_normalize,
    party_normalize,
)

try:
    import pandas as pd  # optional, used for .xlsx/.xls support
except Exception:
    pd = None


def _normalize_source_missing(value: Any) -> Any:
    """Convert source-library missing sentinels to a real missing value.

    pandas represents blank spreadsheet cells as NaN/NA values. These sentinels
    must not escape as domain strings such as ``"nan"`` or normalize to
    substantive values such as ``OTHER``.
    """
    if value is None:
        return None

    if pd is not None:
        try:
            if bool(pd.isna(value)):
                return None
        except (TypeError, ValueError):
            pass

    return value


def _party_is_missing(value: Any) -> bool:
    """Return True when party data is absent or the UNKNOWN sentinel."""
    if not value:
        return True
    return str(value).strip().upper() == "UNKNOWN"


def parse(page, coordinator, context: Dict[str, Any] | None = None, session_id: Optional[str] = None, manual_file: Optional[str] = None, **kwargs) -> Optional[Tuple[List[str], List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Parse FEC-style CSV or Excel exported candidate summary. Returns (headers, rows, contest, metadata).

    Accepts `manual_file` pointing to a CSV or Excel (.xlsx/.xls). If pandas is unavailable
    and an Excel file is provided, the function will return None.
    """
    if not manual_file or not os.path.exists(manual_file):
        return None

    ext = os.path.splitext(manual_file)[1].lower().lstrip('.')
    rows: List[Dict[str, Any]] = []
    try:
        if ext in ('xlsx', 'xls'):
            if pd is None:
                return None
            # Read first sheet
            df = pd.read_excel(manual_file, sheet_name=0, dtype=str)
            raw_headers = list(df.columns.astype(str).tolist())
            canonical_order, mapping = canonicalize_headers(raw_headers)
            for _, r in df.iterrows():
                out: Dict[str, Any] = {}
                for orig in raw_headers:
                    val = _normalize_source_missing(r.get(orig))
                    key = mapping.get(str(orig).strip(), str(orig).strip())
                    if key in ("total_receipts", "total_disbursement", "cash_on_hand", "debt"):
                        out[key] = money_normalize(val)
                    elif key in ("coverage_end_date", "coverage_start_date"):
                        out[key] = date_normalize(val)
                    elif key == "party":
                        out[key] = party_normalize(val)
                    elif key == "incumbent_status":
                        out[key] = incumbent_normalize(val)
                    else:
                        out[key] = str(val).strip() if val is not None else ""
                rows.append(out)
                # attempt enrichment from candidate index (id first, then fuzzy name)
                try:
                    cand_id = out.get('candidate_id') or out.get('Cand_Id') or out.get('CAND_ID')
                    if cand_id:
                        cand = get_candidate_by_id(str(cand_id).strip())
                        if cand:
                            out['_fec_candidate'] = cand
                            # prefer filling missing party/name from candidate record
                            if _party_is_missing(out.get('party')):
                                party_token = cand.get('Cand_Party_Affiliation') or cand.get('Party') or cand.get('cand_party_affiliation')
                                out['party'] = party_normalize(party_token)
                            if not out.get('candidate_name'):
                                name_token = cand.get('Cand_Name') or cand.get('candidate_name')
                                if name_token:
                                    out['candidate_name'] = str(name_token).strip()
                    else:
                        name_token = out.get('candidate_name') or out.get('Cand_Name') or out.get('CandName') or ''
                        state_token = out.get('state') or out.get('Cand_Office_St') or out.get('Cand_State')
                        if name_token:
                            match = find_candidate_by_name(name_token, state=state_token, cutoff=80)
                            if match:
                                out['_fec_candidate_match'] = match
                                if (match.get('score') or 0) >= 85:
                                    rec = match.get('record')
                                    if rec:
                                        out['_fec_candidate'] = rec
                                        if _party_is_missing(out.get('party')):
                                            out['party'] = party_normalize(rec.get('Cand_Party_Affiliation') or rec.get('Party'))
                                        if not out.get('candidate_name'):
                                            out['candidate_name'] = str(rec.get('Cand_Name') or rec.get('candidate_name') or name_token).strip()
                except Exception:
                    pass
        else:
            with open(manual_file, 'r', encoding='utf-8', errors='replace', newline='') as fh:
                reader = csv.DictReader(fh)
                raw_headers = list(reader.fieldnames or [])
                canonical_order, mapping = canonicalize_headers(raw_headers)
                for r in reader:
                    out: Dict[str, Any] = {}
                    for orig, val in r.items():
                        if orig is None:
                            continue
                        val = _normalize_source_missing(val)
                        key = mapping.get(orig.strip(), orig.strip())
                        # normalization heuristics for important keys
                        if key in ("total_receipts", "total_disbursement", "cash_on_hand", "debt"):
                            out[key] = money_normalize(val)
                        elif key in ("coverage_end_date", "coverage_start_date"):
                            out[key] = date_normalize(val)
                        elif key == "party":
                            out[key] = party_normalize(val)
                        elif key == "incumbent_status":
                            out[key] = incumbent_normalize(val)
                        else:
                            out[key] = val.strip() if isinstance(val, str) else val
                    rows.append(out)
                    # enrichment for CSV-parsed rows
                    try:
                        cand_id = out.get('candidate_id') or out.get('Cand_Id') or out.get('CAND_ID')
                        if cand_id:
                            cand = get_candidate_by_id(str(cand_id).strip())
                            if cand:
                                out['_fec_candidate'] = cand
                                if _party_is_missing(out.get('party')):
                                    party_token = cand.get('Cand_Party_Affiliation') or cand.get('Party') or cand.get('cand_party_affiliation')
                                    out['party'] = party_normalize(party_token)
                                if not out.get('candidate_name'):
                                    name_token = cand.get('Cand_Name') or cand.get('candidate_name')
                                    if name_token:
                                        out['candidate_name'] = str(name_token).strip()
                            else:
                                # fuzzy match by name for CSV rows without id
                                try:
                                    name_token = out.get('candidate_name') or out.get('Cand_Name') or out.get('CandName') or ''
                                    state_token = out.get('state') or out.get('Cand_Office_St') or out.get('Cand_State')
                                    match = None
                                    if name_token:
                                        match = find_candidate_by_name(name_token, state=state_token, cutoff=80)
                                    if match:
                                        out['_fec_candidate_match'] = match
                                        if (match.get('score') or 0) >= 85:
                                            rec = match.get('record')
                                            if rec:
                                                out['_fec_candidate'] = rec
                                                if _party_is_missing(out.get('party')):
                                                    out['party'] = party_normalize(rec.get('Cand_Party_Affiliation') or rec.get('Party'))
                                                if not out.get('candidate_name'):
                                                    out['candidate_name'] = str(rec.get('Cand_Name') or rec.get('candidate_name') or name_token).strip()
                                except Exception:
                                    pass
                    except Exception:
                        pass

        contest_label = (context or {}).get('contest') or 'FEC Candidates'
        metadata = {
            'handler': 'fec_handler',
            'row_count': len(rows),
            'source_file': os.path.basename(manual_file)
        }
        # headers to return: canonical_order plus any extra keys discovered in rows
        extra_keys = []
        for r in rows:
            for k in r.keys():
                if k not in canonical_order:
                    extra_keys.append(k)
        headers_final = canonical_order + [k for k in extra_keys if k not in canonical_order]
        return headers_final, rows, contest_label, metadata
    except Exception:
        return None


__all__ = ["parse"]

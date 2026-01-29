from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from webapp.parser.config import LOG_DIR

HERE = os.path.dirname(__file__)
ALIASES_PATH = os.path.normpath(os.path.join(HERE, '..', 'handlers', 'fec_header_aliases.json'))
PARTY_MAP_PATH = os.path.normpath(os.path.join(HERE, '..', 'handlers', 'fec_party_map.json'))


def _load_json(path: str) -> Optional[Dict]:
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    except Exception:
        return None


_ALIASES = _load_json(ALIASES_PATH) or {
    "candidate_id": ["Cand_Id", "CAND_ID", "Cand_Id"],
    "candidate_name": ["Cand_Name", "CAND_NAME", "Cand_Name"],
    "candidate_url": ["Link_Image", "link_image", "candidate_url"],
    "office_type": ["Cand_Office", "CAND_OFFICE", "Cand_Office"],
    "state": ["Cand_Office_St", "Cand_State", "Cand_Office_St"],
    "district": ["Cand_Office_Dist", "Cand_Office_Dist"],
    "party": ["Cand_Party_Affiliation", "Party", "cand_party_affiliation"],
    "incumbent_status": ["Cand_Incumbent_Challenger_Open_Seat", "Incumbent", "Status"],
    "total_receipts": ["Total_Receipt", "Total_Receipts", "TTL_RECEIPTS"],
    "total_disbursement": ["Total_Disbursement", "Total_Disbursements", "TTL_DISB"],
    "cash_on_hand": ["Cash_On_Hand_COP", "Cash_On_Hand_BOP", "COH_COP", "COH_BOP"],
    "debt": ["Debt_Owed_By_Committee", "Debt_Owe_To_Committee"],
    "coverage_end_date": ["Coverage_End_Date", "CVG_END_DT"],
    "coverage_start_date": ["Coverage_Start_Date"],
    "address_street": ["Cand_Street_1", "Cand_Street_2", "Street"],
    "city": ["Cand_City", "City"],
    "zip": ["Cand_Zip", "Zip"],
    "cycle": ["cycle"]
}


_PARTY_MAP = _load_json(PARTY_MAP_PATH) or {
    "DEM": "DEM",
    "DEMOCRAT": "DEM",
    "D": "DEM",
    "REP": "REP",
    "GOP": "REP",
    "R": "REP",
    "IND": "IND",
    "INDEPENDENT": "IND",
    "LIB": "LIB",
    "LP": "LIB",
    "GRE": "GRE",
    "G": "GRE",
    "OTH": "OTHER",
    "OTHER": "OTHER",
}


def _append_ambiguous_log(kind: str, token: str) -> None:
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        path = os.path.join(LOG_DIR, 'fechandler_ambiguous_tokens.jsonl')
        with open(path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps({"kind": kind, "token": token}) + "\n")
    except Exception:
        pass


def canonicalize_headers(headers: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Map original headers to canonical keys.

    Returns (canonical_order, mapping_original_to_canonical)
    """
    mapping: Dict[str, str] = {}
    canonical_order: List[str] = []
    lowered = [h.strip() for h in headers]
    for orig in lowered:
        found = False
        for canon, variants in _ALIASES.items():
            for v in variants:
                if v.lower() == orig.lower():
                    mapping[orig] = canon
                    if canon not in canonical_order:
                        canonical_order.append(canon)
                    found = True
                    break
            if found:
                break
        if not found:
            # keep original as-is but normalized
            key = re.sub(r"[^A-Za-z0-9]+", "_", orig).strip('_') or orig
            mapping[orig] = key
            if key not in canonical_order:
                canonical_order.append(key)
            _append_ambiguous_log('header', orig)
    return canonical_order, mapping


def money_normalize(val: Optional[str]) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return None
    try:
        s = s.replace(',', '')
        # handle parentheses as negative
        if s.startswith('(') and s.endswith(')'):
            s = '-' + s[1:-1]
        return float(s)
    except Exception:
        _append_ambiguous_log('money', str(val))
        try:
            # as fallback, extract digits
            m = re.search(r"-?\d+[\d\.]*", str(val))
            return float(m.group(0)) if m else None
        except Exception:
            return None


def date_normalize(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    s = str(val).strip()
    # try common formats
    fmts = ["%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%Y/%m/%d"]
    for f in fmts:
        try:
            dt = datetime.strptime(s, f)
            return dt.date().isoformat()
        except Exception:
            continue
    # try to extract numbers
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", s)
    if m:
        mm, dd, yy = m.groups()
        yy = yy if len(yy) == 4 else ('20' + yy)
        try:
            dt = datetime(int(yy), int(mm), int(dd))
            return dt.date().isoformat()
        except Exception:
            pass
    _append_ambiguous_log('date', s)
    return None


def party_normalize(token: Optional[str]) -> str:
    if not token:
        return 'UNKNOWN'
    t = str(token).strip().upper()
    if t in _PARTY_MAP:
        return _PARTY_MAP[t]
    # common shorthand
    t2 = re.sub(r"[^A-Z]", "", t)
    if t2 in _PARTY_MAP:
        return _PARTY_MAP[t2]
    _append_ambiguous_log('party', t)
    return 'OTHER'


def incumbent_normalize(token: Optional[str]) -> str:
    if not token:
        return 'UNKNOWN'
    t = str(token).strip().upper()
    if 'INC' in t:
        return 'INCUMBENT'
    if 'OPEN' in t:
        return 'OPEN'
    if 'CHALL' in t or 'CHALLENGER' in t:
        return 'CHALLENGER'
    return t


__all__ = [
    'canonicalize_headers',
    'money_normalize',
    'date_normalize',
    'party_normalize',
    'incumbent_normalize',
]

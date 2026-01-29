from __future__ import annotations

import json
import os
from typing import Dict, Any, Optional
from .config import MIN_FUZZY_SCORE_MANUAL, FUZZY_SCORER

HERE = os.path.dirname(__file__)
FIXTURES = os.path.join(HERE, 'fixtures')
DEFAULT_INDEX = os.path.join(FIXTURES, 'candidate_summary_index.json')

_CACHE: Optional[Dict[str, Dict[str, Any]]] = None
_NAME_INDEX: Optional[list] = None


def _normalize_name(name: str) -> str:
    if not name:
        return ""
    s = str(name).strip()
    # common suffixes/titles to remove
    for tok in ("MR", "MRS", "MS", "DR", "SR", "JR", "MD", "ESQ"):
        s = s.replace(f", {tok}.", "").replace(f" {tok}.", "")
        s = s.replace(f", {tok}", "").replace(f" {tok}", "")
    s = s.replace('"', '')
    # If format is "LAST, FIRST ..." convert to "FIRST LAST"
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) >= 2:
            first = parts[1]
            last = parts[0]
            s = f"{first} {last}"
    # collapse whitespace and lower
    s = " ".join(s.split()).lower()
    return s


def load_fec_candidates(index_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    path = index_path or DEFAULT_INDEX
    if not os.path.exists(path):
        _CACHE = {}
        return _CACHE
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
            _CACHE = data
            return _CACHE
    except Exception:
        _CACHE = {}
        return _CACHE


def get_candidate_by_id(cand_id: str) -> Optional[Dict[str, Any]]:
    if not cand_id:
        return None
    data = load_fec_candidates()
    return data.get(cand_id)


def _build_name_index() -> list:
    """Build an in-memory list of tuples (cand_id, normalized_name, record)."""
    global _NAME_INDEX
    if _NAME_INDEX is not None:
        return _NAME_INDEX
    data = load_fec_candidates()
    out = []
    for cid, rec in (data or {}).items():
        name = rec.get('Cand_Name') or rec.get('candidate_name') or rec.get('CandName') or ''
        norm = _normalize_name(name)
        if norm:
            out.append((cid, norm, rec))
    _NAME_INDEX = out
    return _NAME_INDEX


def find_candidate_by_name(
    name: str,
    state: str | None = None,
    party: str | None = None,
    cutoff: int | None = None,
    scorer: str | None = None,
    top_k: int = 1,
) -> dict | None:
    """Find candidate matches by name using fuzzy matching.

    Parameters:
      name: query name
      state: optional state filter (not currently applied to scoring)
      party: optional party filter (not currently applied)
      cutoff: primary cutoff for high-confidence (0-100)
      scorer: 'auto'|'rapidfuzz'|'difflib' to select backend
      top_k: return top_k candidates (1 returns single best, >1 adds 'candidates' list)

    Returns a dict with keys: `cand_id`, `record`, `score`, `method`, and optionally
    `candidates` (list of {cand_id, record, score}) when `top_k` > 1. Returns None if no reasonable match.
    """
    if not name:
        return None
    idx = _build_name_index()
    if not idx:
        return None
    target = _normalize_name(name)

    # apply config defaults when caller did not specify
    if cutoff is None:
        cutoff = int(MIN_FUZZY_SCORE_MANUAL or 70)
    if scorer is None:
        scorer = FUZZY_SCORER or "auto"

    method_used = None
    candidates = []

    def _to_rec_list(matches):
        out = []
        for key, score in matches:
            rec = load_fec_candidates().get(key)
            out.append({"cand_id": key, "record": rec, "score": int(score)})
        return out

    # Try rapidfuzz when allowed
    try_rapid = scorer in ("auto", "rapidfuzz")
    tried_rapid = False
    if try_rapid:
        try:
            from rapidfuzz import process, fuzz  # type: ignore
            tried_rapid = True
            choices = {t[0]: t[1] for t in idx}
            scorer_fn = fuzz.token_sort_ratio
            # extract top_k matches
            if top_k == 1:
                best = process.extractOne(target, choices, scorer=scorer_fn)
                if best:
                    cand_id, score, _ = best
                    candidates = _to_rec_list([(cand_id, score)])
            else:
                bests = process.extract(target, choices, scorer=scorer_fn, limit=top_k)
                # bests: list of (key, score, index)
                candidates = _to_rec_list([(b[0], b[1]) for b in bests])
            method_used = "rapidfuzz"
        except Exception:
            tried_rapid = True
            # fallthrough to difflib if auto

    # If no candidates found via rapidfuzz (or requested difflib), try difflib
    if (not candidates) and (scorer in ("auto", "difflib") or (scorer == "rapidfuzz" and not tried_rapid)):
        try:
            import difflib
            names_map = {t[1]: t[0] for t in idx}  # norm -> cid
            names = list(names_map.keys())
            if top_k == 1:
                sm = difflib.get_close_matches(target, names, n=1, cutoff=cutoff / 100.0)
                if sm:
                    matched = sm[0]
                    cid = names_map.get(matched)
                    ratio = difflib.SequenceMatcher(None, target, matched).ratio()
                    score = int(ratio * 100)
                    candidates = [(cid, score)]
            else:
                # get more matches via simple scoring
                scored = []
                for cid, norm, rec in idx:
                    ratio = difflib.SequenceMatcher(None, target, norm).ratio()
                    scored.append((cid, int(ratio * 100)))
                scored.sort(key=lambda x: x[1], reverse=True)
                candidates = scored[:top_k]
            method_used = "difflib"
        except Exception:
            pass

    if not candidates:
        return None

    # convert to structured list
    cand_list = _to_rec_list(candidates if isinstance(candidates[0], tuple) else [(c['cand_id'], c['score']) for c in candidates]) if candidates else []

    best = cand_list[0]
    result = {
        "cand_id": best.get("cand_id"),
        "record": best.get("record"),
        "score": best.get("score"),
        "method": method_used or "unknown",
    }
    if top_k > 1:
        result["candidates"] = cand_list
    return result


__all__ = ['load_fec_candidates', 'get_candidate_by_id', 'find_candidate_by_name']

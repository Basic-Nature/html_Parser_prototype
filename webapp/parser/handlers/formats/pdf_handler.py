from __future__ import annotations
# ==============================================================
# 🗳️ Smart Elections: Universal PDF Election Results Parser
# ==============================================================
import os
import re
import csv
import time
import platform
import shutil
import importlib
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from ...config import (
    ENABLE_OCR, OUTPUT_DIR
)
import html
# Optional flags/paths; provide safe defaults if missing
try:
    from ...config import ENABLE_OCR_FORCE, OCR_DEBUG_DIR
except Exception:
    ENABLE_OCR_FORCE = False
    OCR_DEBUG_DIR = os.path.join(OUTPUT_DIR, "ocr_debug")
    os.makedirs(OCR_DEBUG_DIR, exist_ok=True)

try:
    from ...config import POPPLER_PATH as CONFIG_POPPLER_PATH
except Exception:
    CONFIG_POPPLER_PATH = None
try:
    from ...config import TESSERACT_CMD as CONFIG_TESSERACT_CMD
except Exception:
    CONFIG_TESSERACT_CMD = None

try:
    from ...config import ENABLE_CAMELOT
except Exception:
    ENABLE_CAMELOT = True

from ...utils.camelot_utils import (
    attempt_camelot_extraction,
    hybrid_fill_camelot,
)

# Added optional tuning flags (safe defaults if not in config)
try:
    from ...config import CAMELOT_MIN_SCORE, CAMELOT_HYBRID_FILL, CAMELOT_MERGE_COMPAT
except Exception:
    CAMELOT_MIN_SCORE = 0.9
    CAMELOT_HYBRID_FILL = True
    CAMELOT_MERGE_COMPAT = True

# Optional Camelot import
try:
    import camelot
    _CAMELOT_AVAILABLE = True
except Exception:
    _CAMELOT_AVAILABLE = False

from ...utils.logger_singleton import logger, prompt
from ...Context_Integration.Context_Library.constants import (
    LOCATION_KEYWORDS,
    CANDIDATE_KEYWORDS, BALLOT_TYPES, PARTY_KEYWORDS, TOTAL_KEYWORDS,
    MISC_FOOTER_KEYWORDS, CONTEST_KEYWORDS, CONTEST_TITLE_SKIP_PHRASES,
    CONTEST_HEADER_KEYWORDS, CONTEST_HEADER_PREFERENCE
)

def _camelot_signal_sets():
    # Build once (could memoize)
    base = set()
    for group in (
        CANDIDATE_KEYWORDS,
        PARTY_KEYWORDS,
        TOTAL_KEYWORDS,
        BALLOT_TYPES,
        {"percent", "%", "election day", "early", "absentee", "provisional", "total vote", "grand total"}
    ):
        base |= {str(x).lower() for x in group}
    noise = {str(x).lower() for x in MISC_FOOTER_KEYWORDS}
    return base, noise


def _prepare_output_context(base: dict | None, extra: dict | None = None) -> dict:
    """Merge context dictionaries while excluding non-serializable helpers."""
    ctx: dict = {}
    if isinstance(base, dict):
        for key, value in base.items():
            if key == "coordinator":
                continue
            ctx[key] = value
    if isinstance(extra, dict):
        ctx.update(extra)
    return ctx

from ...utils.table_core import harmonize_headers_and_data
from ...utils.location_helpers import (
    attach_precinct_column,
    collect_location_headers,
    is_strict_location_header,
)
import orjson
from ...utils.contest_selector import (
    select_contest_auto_first,
    resolve_selection_context
)
from ...utils.table_builder import build_table_noninteractive
from ...utils.output_utils import finalize_election_output
from ...utils.shared_logic import format_county_label, format_state_label, safe_get, safe_slug
from ...utils.pivot import expand_single_rawjson_row
from ...Context_Integration.context_coordinator import dynamic_state_county_detection
from ...Context_Integration.Context_Library.constants import normalize_party_label
from ...utils.table_core import robust_table_extraction
_FITZ_MODULE = None
_FITZ_IMPORT_WARNINGS: tuple[str, ...] = ()
_FITZ_PATCHED_TYPES: tuple[str, ...] = ()
_FITZ_PATCH_FAILURES: tuple[str, ...] = ()
_FITZ_WARNING_LOGGED = False
_SWIG_WARNING_PATTERN = re.compile(
    r"builtin type (?P<name>SwigPyObject|SwigPyPacked|swigvarlink) has no __module__ attribute"
)
_PYMUPDF_MIN_VERSION = (1, 26, 5)


def _check_pymupdf_version(module):
    """Emit guidance when PyMuPDF is behind the tested baseline."""
    version_str = getattr(module, "__version__", "0.0.0")
    try:
        parts = tuple(int(p) for p in version_str.split(".")[:3])
    except ValueError:
        parts = (0, 0, 0)
    if parts and parts < _PYMUPDF_MIN_VERSION:
        logger.warning({
            "level": "WARNING",
            "type": "dependency",
            "message": (
                "[WARN] Detected PyMuPDF %s. Upgrade to %s or newer to incorporate "
                "SWIG metadata fixes and avoid deprecation noise."
            )
            % (version_str, ".".join(str(p) for p in _PYMUPDF_MIN_VERSION)),
        })


def _patch_fitz_swig_types(module, warning_messages):
    """Assign a module name to SWIG-generated types to satisfy Python 3.12+ requirements."""
    patched: list[str] = []
    failures: list[str] = []
    for message in warning_messages:
        match = _SWIG_WARNING_PATTERN.search(message)
        if not match:
            continue
        type_name = match.group("name")
        target = getattr(module, type_name, None)
        if not isinstance(target, type):
            failures.append(type_name)
            continue
        module_name = getattr(target, "__module__", "")
        if module_name:
            patched.append(type_name)
            continue
        try:
            setattr(target, "__module__", module.__name__)
            patched.append(type_name)
        except (AttributeError, TypeError):
            failures.append(type_name)
    return patched, failures


try:
    import pandas as pd  # type: ignore
    _PANDAS_AVAILABLE = True
except Exception:  # pragma: no cover - pandas is optional but strongly recommended
    pd = None
    _PANDAS_AVAILABLE = False


def _ensure_fitz():
    """Import PyMuPDF while capturing its SWIG DeprecationWarnings safely."""
    global _FITZ_MODULE, _FITZ_IMPORT_WARNINGS, _FITZ_PATCHED_TYPES, _FITZ_PATCH_FAILURES
    if _FITZ_MODULE is not None:
        return _FITZ_MODULE

    captured_msgs: list[str] = []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        try:
            module = importlib.import_module("fitz")
        except ImportError as exc:
            raise ImportError("You must install PyMuPDF to use the PDF handler: pip install pymupdf") from exc
        captured_msgs = [str(warning.message) for warning in (caught or [])]

    _check_pymupdf_version(module)

    swig_msgs: list[str] = []
    other_msgs: list[str] = []
    if captured_msgs:
        for msg in captured_msgs:
            if _SWIG_WARNING_PATTERN.search(msg):
                swig_msgs.append(msg)
            else:
                other_msgs.append(msg)

    if swig_msgs:
        patched, failures = _patch_fitz_swig_types(module, swig_msgs)
        _FITZ_PATCHED_TYPES = tuple(sorted(set(patched)))
        _FITZ_PATCH_FAILURES = tuple(sorted(set(failures)))
        if _FITZ_PATCHED_TYPES:
            logger.info({
                "level": "INFO",
                "type": "handler",
                "message": "[INFO] Applied module metadata patches to PyMuPDF SWIG types.",
                "patched_types": list(_FITZ_PATCHED_TYPES),
            })
        # Retain warnings for any types we could not patch successfully
        unresolved = []
        for msg in swig_msgs:
            match = _SWIG_WARNING_PATTERN.search(msg)
            if match and match.group("name") not in patched:
                unresolved.append(msg)
        _FITZ_IMPORT_WARNINGS = tuple(sorted(set(unresolved)))
    else:
        _FITZ_IMPORT_WARNINGS = ()
        _FITZ_PATCHED_TYPES = ()
        _FITZ_PATCH_FAILURES = ()

    if other_msgs:
        logger.warning({
            "level": "WARNING",
            "type": "handler",
            "message": "[WARN] PyMuPDF import emitted unexpected warning(s).",
            "warning_details": other_msgs,
        })

    _FITZ_MODULE = module
    globals()["fitz"] = module
    return module


fitz = _ensure_fitz()

try:
    import pytesseract
    import pdf2image
    if CONFIG_TESSERACT_CMD:
        # Allow Windows to work without adding Tesseract to PATH
        try:
            pytesseract.pytesseract.tesseract_cmd = CONFIG_TESSERACT_CMD
        except Exception:
            pass
except ImportError:
    pytesseract = None
    pdf2image = None

# -------------------- Semantic contest block + candidate-line extractors --------------------

_CAMELOT_SIGNAL_SET, _CAMELOT_NOISE_SET = _camelot_signal_sets()

_NUM_TOKEN_RE = re.compile(r"^\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*$")
_PCT_TOKEN_RE = re.compile(r"^\s*\d{1,3}(?:\.\d+)?\s*%\s*$")

def _score_camelot_table(df) -> float:
    """
    Scoring components:
      header_signal = (# header tokens in keyword set / columns) * 2.0
      candidate_row_density = rows with >=1 header-signal token / rows * 1.5
      numeric_density = numeric-ish cells / total cells * 1.2
      percent_presence bonus
      width_penalties
    """
    try:
        import pandas as pd  # noqa
        if df is None or df.empty:
            return 0.0
        cols = [str(c).strip().lower() for c in df.columns]
        col_hits = sum(1 for c in cols if any(k in c for k in _CAMELOT_SIGNAL_SET))
        header_signal = (col_hits / max(1, len(cols))) * 2.0

        total_cells = int(df.shape[0] * df.shape[1])
        numeric_cells = 0
        candidate_like_rows = 0
        percent_cells = 0
        for _, row in df.iterrows():
            row_tokens = 0
            row_has_signal = False
            for v in row.tolist():
                vs = str(v).strip().lower()
                if not vs:
                    continue
                if _NUM_TOKEN_RE.match(vs):
                    numeric_cells += 1
                elif _PCT_TOKEN_RE.match(vs):
                    numeric_cells += 1
                    percent_cells += 1
                else:
                    # detect numeric with commas
                    core = vs.replace(",", "")
                    if core.isdigit():
                        numeric_cells += 1
                if any(k in vs for k in _CAMELOT_SIGNAL_SET):
                    row_has_signal = True
                row_tokens += 1
            if row_has_signal:
                candidate_like_rows += 1

        numeric_density = (numeric_cells / max(1, total_cells)) * 1.2
        candidate_row_density = (candidate_like_rows / max(1, df.shape[0])) * 1.5
        pct_bonus = 0.3 if percent_cells > 0 else 0.0

        width_penalty = 0.0
        if df.shape[1] > 40:
            width_penalty += 0.6
        elif df.shape[1] > 28:
            width_penalty += 0.3
        if df.shape[1] < 2:
            width_penalty += 0.6

        noise_penalty = 0.0
        if any(any(n in c for n in _CAMELOT_NOISE_SET) for c in cols):
            noise_penalty += 0.15

        score = header_signal + candidate_row_density + numeric_density + pct_bonus - width_penalty - noise_penalty
        return round(score, 4)
    except Exception:
        return 0.0

def _normalize_camelot_headers(raw_headers):
    norm = []
    seen = set()
    for i, h in enumerate(raw_headers):
        hs = (str(h) or "").strip() or f"Column {i+1}"
        hs = re.sub(r"\s+", " ", hs)
        base = hs
        k = 2
        while hs.lower() in seen:
            hs = f"{base}_{k}"
            k += 1
        seen.add(hs.lower())
        norm.append(hs)
    return norm

def _camelot_table_to_rows(table, session_id=None, limit_rows=1200):
    try:
        df = table.df
    except Exception:
        return [], []
    try:
        df = df.replace(r"^\s*$", "", regex=True)
    except Exception:
        pass

    # Promote first row to headers if it looks stronger
    orig_cols = [str(c).strip() for c in df.columns]
    first_row = [str(x).strip() for x in (df.iloc[0].tolist() if len(df) else [])]
    def _sig(lst): return sum(1 for c in lst if any(k in c.lower() for k in _CAMELOT_SIGNAL_SET))
    if first_row and _sig(first_row) > _sig(orig_cols):
        df.columns = first_row
        df = df.iloc[1:]

    headers = _normalize_camelot_headers(df.columns.tolist())
    out = []
    for _, r in df.head(limit_rows).iterrows():
        vals = r.tolist()
        rec = {}
        empty = True
        for h, v in zip(headers, vals):
            if isinstance(v, str):
                vs = v.strip()
                if vs.replace(",", "").replace("%", "").isdigit():
                    rec[h] = vs
                else:
                    rec[h] = vs
            else:
                rec[h] = v
            if rec[h] not in ("", None):
                empty = False
        if not empty:
            out.append(rec)
    return headers, out

def _merge_camelot_tables_if_compatible(entries: list[dict]) -> list[dict]:
    """
    Merge tables with identical normalized header sets (order-insensitive) to reduce fragmentation.
    """
    if not entries:
        return entries
    buckets = {}
    for e in entries:
        key = tuple(sorted([h.lower() for h in e["headers"]]))
        buckets.setdefault(key, []).append(e)
    merged = []
    for key, group in buckets.items():
        if len(group) == 1:
            merged.append(group[0])
            continue
        # Combine rows
        rows = []
        headers_ref = group[0]["headers"]
        for g in group:
            rows.extend(g["rows"])
        merged.append({
            "headers": headers_ref,
            "rows": rows,
            "score": max(g["score"] for g in group),
            "flavor": ",".join(sorted({g["flavor"] for g in group})),
            "page_range": ",".join(str(g.get("page_range") or "") for g in group)
        })
    # Preserve ordering by score
    merged.sort(key=lambda x: (-x["score"], -len(x["rows"])))
    return merged

def _extract_camelot_tables(pdf_path: str, session_id=None, max_tables=15, pages="1-end"):
    results = []
    if not _CAMELOT_AVAILABLE:
        return results
    for flavor in ("lattice", "stream"):
        try:
            tables = camelot.read_pdf(
                pdf_path,
                pages=pages,
                flavor=flavor,
                strip_text="\n",
                suppress_stdout=True
            )
        except Exception as e:
            logger.debug({
                "level": "DEBUG",
                "type": "handler",
                "message": f"[DEBUG] Camelot {flavor} failed: {e}",
                "session_id": session_id
            })
            continue
        for t in list(tables)[:max_tables]:
            try:
                score = _score_camelot_table(t.df)
            except Exception:
                score = 0.0
            h, r = _camelot_table_to_rows(t, session_id=session_id)
            if h and r:
                results.append({
                    "headers": h,
                    "rows": r,
                    "score": score,
                    "flavor": flavor,
                    "page_range": getattr(t, "page", None)
                })
        if len(results) >= max_tables:
            break
    results.sort(key=lambda x: (-x["score"], -len(x["rows"])))
    if CAMELOT_MERGE_COMPAT:
        results = _merge_camelot_tables_if_compatible(results)
    return results

# Hybrid fill: enrich Camelot rows using OCR text lines for missing numeric fields
def _hybrid_fill_camelot(top_table: dict, ocr_lines: list[str]):
    if not CAMELOT_HYBRID_FILL or not top_table or not ocr_lines:
        return
    headers = top_table["headers"]
    rows = top_table["rows"]
    numeric_cols = [h for h in headers if any(k in h.lower() for k in ("vote", "total", "absentee", "early", "election day", "provisional", "%"))]
    if not numeric_cols:
        return
    # Build fast search index of OCR lines
    ocr_index = {}
    for ln in ocr_lines:
        low = ln.lower()
        # candidate-like heuristic: contains alphabetic and a number
        if sum(ch.isalpha() for ch in low) >= 3 and re.search(r"\d", low):
            ocr_index[len(ocr_index)] = low
    for row in rows:
        name_candidate = None
        for k in ("Candidate", "Name"):
            if k in row and isinstance(row[k], str):
                name_candidate = row[k].strip()
                break
        if not name_candidate:
            continue
        low_name = name_candidate.lower()
        # Skip if row mostly filled
        empties = [c for c in numeric_cols if not str(row.get(c, "")).strip()]
        if not empties:
            continue
        # Find best OCR line containing name token(s)
        tokens = [t for t in re.findall(r"[a-z0-9]+", low_name) if len(t) > 2]
        best = None
        best_hits = 0
        for _, ln in ocr_index.items():
            hits = sum(1 for t in tokens if t in ln)
            if hits > best_hits:
                best_hits = hits
                best = ln
        if not best or best_hits == 0:
            continue
        # Extract numbers in sequence and map to empty columns heuristically
        nums = re.findall(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?%?", best)
        if not nums:
            continue
        # Simple left-to-right fill for empty numeric cols
        fill_iter = iter(nums)
        for col in empties:
            try:
                val = next(fill_iter)
            except StopIteration:
                break
            if val.endswith("%"):
                # Only put into percent-like column
                if "%" in col or "percent" in col.lower():
                    row[col] = val
                else:
                    # Skip placing % into a vote column
                    continue
            else:
                clean = val.replace(",", "")
                if clean.isdigit():
                    row[col] = int(clean)
                else:
                    row[col] = val

_NUM_RE = re.compile(r"(?P<num>\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_PCT_RE = re.compile(r"(?P<pct>\d{1,3}(?:\.\d+)?)\s*%")

def _norm_txt(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r'[\s\-_\/]+', ' ', s)
    return re.sub(r'[^a-z0-9 %]+', '', s)

def _token_set(s: str) -> set[str]:
    return set(re.findall(r'[a-z0-9]+', (s or "").lower()))

def _best_title_match_idx(lines: list[str], selected_title: str) -> int:
    """Find the index of the line that best matches the selected title by token overlap."""
    if not selected_title:
        return -1
    sel_tok = _token_set(selected_title)
    best = (-1.0, -1)
    for i, line in enumerate(lines[:600]):
        lt = _token_set(line)
        if not lt:
            continue
        inter = len(sel_tok & lt)
        union = len(sel_tok | lt) or 1
        jacc = inter / union
        if jacc > best[0]:
            best = (jacc, i)
    return best[1]

def _extract_contest_block(lines: list[str], selected_title: str) -> list[str]:
    """
    Extract lines starting from the best-matching title down to the next probable contest heading
    or a gap of 3+ blank lines.
    """
    if not lines:
        return []
    start_idx = _best_title_match_idx(lines, selected_title)
    if start_idx < 0:
        return []
    block = []
    blanks = 0
    for j in range(start_idx + 1, min(len(lines), start_idx + 800)):
        raw = lines[j].strip()
        low = raw.lower()
        if not raw:
            blanks += 1
            if blanks >= 3 and len(block) >= 2:
                break
            continue
        blanks = 0
        # stop at next contest-like heading
        if _CONTEST_RX.search(low) and j > start_idx + 2:
            break
        block.append(raw)
    return block

def _parse_candidate_line(line: str, ballot_types: list[str]) -> dict | None:
    """
    Parse a single candidate result line like:
      'Jane Doe (Democratic) Election Day 1,234 Early 567 Absentee 200 Total 2,001 54.3%'
      'John Smith REP 3,456 45.7%'
    Returns dict with Candidate, Party, per-group values, Total Vote, % Vote if detected.
    """
    if not line or sum(ch.isalpha() for ch in line) < 3:
        return None
    pct_match = _PCT_RE.search(line)
    pct_val = f"{pct_match.group('pct')}%" if pct_match else None
    nums = [m.group('num') for m in _NUM_RE.finditer(line)]
    if not nums:
        return None

    # Common aliases to canonical ballot groups
    alias = {
        "ed": "Election Day",
        "electionday": "Election Day",
        "inperson": "Election Day",
        "early": "Early Voting",
        "early voting": "Early Voting",
        "absentee": "Absentee",
        "mail": "Absentee",
        "by mail": "Absentee",
        "provisional": "Provisional",
        "advance": "Early Voting",
        "total": "Total Vote",
    }

    parts = line.split()
    first_num_idx = None
    for idx, p in enumerate(parts):
        if _NUM_RE.fullmatch(p.strip(",%")):
            first_num_idx = idx
            break

    party = None
    m_paren = re.search(r"\(([^)]+)\)", line)
    if m_paren:
        party = normalize_party_label(m_paren.group(1))
    else:
        trailing = parts[:first_num_idx] if first_num_idx else parts
        if trailing:
            tail = trailing[-1].strip(" ,;")
            if len(tail) <= 12 and any(x in tail.lower() for x in ("dem", "rep", "green", "ind", "wf", "conserv", "lib")):
                party = normalize_party_label(tail)

    name_region = " ".join(parts[:first_num_idx] or parts).strip()
    name_region = re.sub(r"\([^)]*\)", "", name_region).strip()
    name = re.sub(r"\s{2,}", " ", name_region).strip(" -:\t")

    row = {"Candidate": name}
    if party:
        row["Party"] = normalize_party_label(party)

    assigned = {}
    norm_line = _norm_txt(line)
    # Label-aware assignment: scan for each ballot type keyword near numbers
    for bt in (ballot_types or []):
        key = _norm_txt(bt).replace("_", " ")
        if key and key in norm_line:
            idx = norm_line.find(key)
            tail = norm_line[idx:]
            m = _NUM_RE.search(tail)
            if m:
                val = int(m.group("num").replace(",", ""))
                assigned[bt] = val
    # Also try alias mapping
    for k, v in alias.items():
        if k in norm_line and v not in assigned:
            idx = norm_line.find(k)
            tail = norm_line[idx:]
            m = _NUM_RE.search(tail)
            if m:
                val = int(m.group("num").replace(",", ""))
                assigned[v] = val

    total_val = None
    if not assigned:
        try:
            total_val = int(nums[-1].replace(",", ""))
        except Exception:
            total_val = None
    else:
        total_keys = [k for k in assigned.keys() if "total" in k.lower()]
        if total_keys:
            total_val = assigned.get(total_keys[0])
        else:
            total_val = sum(assigned.values()) if assigned else None

    for k, v in assigned.items():
        row[k] = v
    if total_val is not None:
        row["Total Vote"] = total_val
    if pct_val:
        row["% Vote"] = pct_val
    if "Total Vote" not in row and not assigned:
        return None
    return row

def extract_candidate_totals_from_lines(lines: list[str], selected_title: str) -> tuple[list[str], list[dict]]:
    """
    Given page lines and a selected contest title, extract a candidate totals table
    when no explicit table structure is present.
    """
    block = _extract_contest_block(lines, selected_title)
    if not block:
        return [], []
    # Prefer canonical ballot types from constants
    bt = list(BALLOT_TYPES) if BALLOT_TYPES else ["Election Day", "Early Voting", "Absentee", "Provisional"]
    rows = []
    present_cols = set(["Candidate", "Party"])
    for ln in block:
        r = _parse_candidate_line(ln, bt)
        if r:
            rows.append(r)
            present_cols.update(r.keys())
    if not rows:
        return [], []
    headers = ["Candidate"]
    if "Party" in present_cols:
        headers.append("Party")
    for g in bt:
        if g in present_cols:
            headers.append(g)
    if "Total Vote" in present_cols:
        headers.append("Total Vote")
    if "% Vote" in present_cols:
        headers.append("% Vote")
    norm_rows = [{h: rr.get(h, "") for h in headers} for rr in rows]
    return headers, norm_rows

def _split_ws_blocks(s: str) -> list[str]:
    # Split by 2+ spaces or tab/comma
    cells = re.split(r"\s{2,}|\t|,", s.strip())
    return [c.strip() for c in cells if c.strip()]

def _is_bad_header_line(line: str) -> bool:
    """
    Heuristics to reject lines as headers when they look like narrative/boilerplate or noise:
    - Very long segment(s) or too many cells
    - High digit density
    - Contains boilerplate tokens like 'Statement and Return', 'Page X of Y', 'Total Applicable Ballots'
    - Long write-in chains on one line
    """
    if not isinstance(line, str):
        return True
    s = line.strip()
    if not s:
        return True
    low = s.lower()
    # Obvious boilerplate/noise tokens
    bad_tokens = (
        "statement and return", "printed as of", "page", "of", "total applicable ballots",
        "public counter", "manually counted emergency", "absentee / military",
        "unrecorded", "affidavit", "less - inapplicable", "vote for", "page"
    )
    if any(bt in low for bt in bad_tokens):
        # Allow short contest-like headings that happen to have 'vote for'
        if "vote for" in low and len(s) < 40:
            pass
        else:
            return True
    # Too many cells
    cells = _split_ws_blocks(s)
    if len(cells) > 12:
        return True
    # Any cell extremely long (likely a whole paragraph)
    if any(len(c) > 80 for c in cells):
        return True
    # Digit density
    digits = sum(ch.isdigit() for ch in s)
    if digits and digits / max(1, len(s)) > 0.35:
        return True
    return False

def _table_looks_bad(headers: list[str], rows: list[dict]) -> bool:
    """
    Decide if the extracted table is low quality and should be replaced by semantic extraction.
    Triggers when:
    - Very few rows
    - Headers look narrative/long/numeric-heavy
    - Known boilerplate tokens present in headers
    """
    if not headers:
        return True
    if len(rows) <= 3:
        return True
    low = [h.lower() for h in headers if isinstance(h, str)]
    boiler = ("statement and return", "printed as of", "total applicable ballots", "page ")
    if any(any(b in h for b in boiler) for h in low):
        return True
    # Very long headers or many digits in headers
    if any(len(h) > 80 for h in headers if isinstance(h, str)):
        return True
    if any((sum(ch.isdigit() for ch in h) / max(1, len(h))) > 0.35 for h in headers if isinstance(h, str)):
        return True
    return False

def _find_header_line(lines: list[str], hints: set[str]) -> tuple[list[str], int]:
    """
    Return (headers, header_idx). Prefer lines containing multiple hint hits.
    Fallback to first line with 3+ cells split by whitespace blocks.
    Skips lines deemed noisy/narrative by _is_bad_header_line.
    """
    best = (-1, -1, [])
    for idx, line in enumerate(lines[:400]):
        if _is_bad_header_line(line):
            continue
        cells = _split_ws_blocks(line)
        # Skip overly wide candidates
        if len(cells) > 12:
            continue
        if len(cells) >= 2:
            score = sum(1 for h in hints if h in line.lower())
            if score > best[0]:
                best = (score, idx, cells)
    if best[1] >= 0:
        return best[2], best[1]
    # Fallback: first reasonable line that splits into 3+ cells and is not noisy
    for idx, line in enumerate(lines[:400]):
        if _is_bad_header_line(line):
            continue
        cells = _split_ws_blocks(line)
        if 3 <= len(cells) <= 12:
            return cells, idx
    return [], -1

def _extract_table_by_whitespace(lines: list[str], start_idx: int, headers: list[str]) -> list[dict]:
    """
    Parse rows beneath headers using whitespace block splitting; stop on blank or badly short rows.
    """
    data = []
    min_cols = max(2, len(headers))
    for raw in lines[start_idx+1:]:
        if not raw.strip():
            if len(data) > 0:
                break
            else:
                continue
        cells = _split_ws_blocks(raw)
        if len(cells) < min_cols:
            # allow one-cell tails if headers are 1 (raw lists)
            if len(headers) == 1 and len(cells) == 1:
                data.append({headers[0]: cells[0]})
            else:
                # likely end of table region
                if len(data) > 0:
                    break
                continue
        else:
            # If row has extra cells, pad/truncate to header count
            if len(cells) > len(headers):
                cells = cells[:len(headers)-1] + [" ".join(cells[len(headers)-1:])]
            row = dict(zip(headers, cells))
            data.append(row)
    return data

def _log_ocr_environment(session_id=None):
    try:
        info = {
            "platform": platform.platform(),
            "pytesseract": bool(pytesseract),
            "pdf2image": bool(pdf2image),
            "poppler_path_env": bool(CONFIG_POPPLER_PATH),
            "pdftoppm_in_path": bool(shutil.which("pdftoppm")),
            "tesseract_cmd_set": bool(CONFIG_TESSERACT_CMD),
            "ENABLE_OCR": bool(ENABLE_OCR),
            "ENABLE_OCR_FORCE": bool(ENABLE_OCR_FORCE),
        }
        logger.info({
            "level": "INFO",
            "type": "handler",
            "message": "[ENV] PDF/OCR capabilities detected",
            "session_id": session_id,
            "env": info
        })
    except Exception:
        pass

def _detect_poppler_path() -> str | None:
    """
    Cross-platform Poppler locator.
    - Windows: try config POPPLER_PATH, env POPPLER_PATH, common install dirs.
    - Linux/macOS: if pdftoppm is in PATH, return None (pdf2image will use PATH).
    """
    # Config or env
    if CONFIG_POPPLER_PATH and os.path.isdir(CONFIG_POPPLER_PATH):
        return CONFIG_POPPLER_PATH
    env_path = os.environ.get("POPPLER_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path

    system = platform.system().lower()
    if system.startswith("win"):
        candidates = [
            r"C:\Program Files\poppler\Library\bin",
            r"C:\Program Files\poppler-24.08.0\Library\bin",
            r"C:\Program Files\poppler-24.07.0\Library\bin",
            r"C:\Program Files\poppler-24.06.0\Library\bin",
            r"C:\Program Files\poppler\bin",
            r"C:\poppler\bin",
        ]
        for p in candidates:
            if os.path.isdir(p):
                return p
        return None
    else:
        # On Linux/macOS we rely on PATH
        if shutil.which("pdftoppm") or shutil.which("pdftocairo"):
            return None
        return None

# Build tolerant contest regex (same logic as JSON handler)
def _build_contest_regex(keywords) -> re.Pattern:
    parts = []
    for phrase in (keywords or []):
        if not isinstance(phrase, str) or not phrase.strip():
            continue
        toks = re.split(r"\s+", phrase.strip().lower())
        xtoks = []
        for t in toks:
            t = re.escape(t)
            t = t.replace(r"\.", r"\.?")
            t = t.replace(r"\-", r"[-\s]?")
            xtoks.append(t)
        pat = r"(?:[\s\-_\/]*?)".join(xtoks)
        pat = rf"(?<![A-Za-z0-9]){pat}(?![A-Za-z0-9])"
        parts.append(pat)
    return re.compile("|".join(parts), re.I) if parts else re.compile(r"(?!x)x", re.I)

_CONTEST_RX = _build_contest_regex(CONTEST_KEYWORDS)

def _detect_contest_titles_from_text(lines):
    """
    Heuristic detection of contest titles from plain PDF text using constants.
    - Keep lines containing contest/office keywords (regex tolerant)
    - Drop known skip phrases and very short/noisy lines
    """
    titles = []
    skip_set = {s.lower() for s in (CONTEST_TITLE_SKIP_PHRASES or set())}
    for line in lines:
        raw = (line or "").strip()
        low = raw.lower()
        if not raw or len(raw) < 6:
            continue
        if any(s in low for s in skip_set):
            continue
        # Use robust regex to detect contest references
        if _CONTEST_RX.search(low):
            titles.append(raw)
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for t in titles:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    return uniq[:50]

def _is_mostly_markup(text: str) -> bool:
    """
    Return True if the extracted 'text' is actually markup-wrappers (e.g., <img> tags) with little real text.
    """
    if not isinstance(text, str):
        return False
    s = text.strip().lower()
    if not s:
        return False
    # Heuristics: presence of HTML tags + low alphabetic character count
    has_tags = any(tok in s for tok in ("<img", "<div", "<span", "<html", "<svg", "<p", "<table", "data:image/"))
    if not has_tags:
        return False
    alpha = sum(1 for ch in s[:8000] if ch.isalpha())
    return alpha < 200

def _sanitize_extracted_text(text: str) -> str:
    """
    Convert raw extracted content (which may contain XHTML, <img src="data:image..."> etc.)
    into neat, readable lines for downstream steps.
    - Remove data:image/base64 payloads and HTML tags
    - Unescape entities
    - Collapse whitespace
    - Drop extremely noisy/empty lines
    """
    if not isinstance(text, str):
        return ""
    # Remove data:image base64 attributes entirely
    text = re.sub(r'src\s*=\s*"data:image/[^"]+"', 'src="[image]"', text, flags=re.IGNORECASE)
    # Remove long base64-like runs that may appear outside attributes
    text = re.sub(r'[A-Za-z0-9+/=]{200,}', ' ', text)
    # Strip all HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Unescape HTML entities
    try:
        text = html.unescape(text)
    except Exception:
        pass
    # Normalize whitespace but keep line structure
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        # Collapse internal whitespace
        s = re.sub(r'\s+', ' ', s)
        # Heuristics: keep lines that have some alphanum signal
        alnum = sum(ch.isalnum() for ch in s)
        if alnum < 2:
            continue
        # Drop bracket-only image placeholders
        if s in {"[image]", "[data]"}:
            continue
        # Avoid lines that are mostly punctuation
        punct = sum(not ch.isalnum() and not ch.isspace() for ch in s)
        if alnum and punct / max(1, len(s)) > 0.6:
            continue
        lines.append(s)
    # Deduplicate consecutive duplicates
    neat = []
    last = None
    for l in lines:
        if l != last:
            neat.append(l)
            last = l
    return "\n".join(neat)

def _pdf_to_images(pdf_path: str, session_id=None, dpi: int = 200, page_indices: list[int] | None = None, max_pages: int | None = None):
    """
    Convert PDF pages to PIL Images.
    - If page_indices or max_pages provided, render only that subset (via PyMuPDF to avoid full-doc raster).
    - Else, try pdf2image (Poppler) then fallback to PyMuPDF for all pages.
    """
    images = []
    # Try pdf2image first if available
    # If specific pages requested, use PyMuPDF directly (efficient random-page render)
    if page_indices is not None or max_pages is not None:
        try:
            doc = fitz.open(pdf_path)
            count = len(doc)
            if page_indices is None:
                # render first max_pages pages
                take = min(max_pages or count, count)
                idxs = list(range(take))
            else:
                idxs = sorted({i for i in page_indices if isinstance(i, int) and 0 <= i < count})
            for i in idxs:
                page = doc[i]
                pix = page.get_pixmap(dpi=dpi)
                mode = "RGBA" if pix.alpha else "RGB"
                pil_img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
                images.append(pil_img)
            doc.close()
            return images
        except Exception as e:
            logger.error({
                "level": "ERROR",
                "type": "handler",
                "message": f"[ERROR] Targeted page render failed: {e}",
                "session_id": session_id
            })
            images = []

    # Try full-document pdf2image first if available
    if pdf2image and (page_indices is None and max_pages is None):
        try:
            poppler_path = _detect_poppler_path()
            kwargs = {"dpi": dpi}
            if poppler_path and platform.system().lower().startswith("win"):
                kwargs["poppler_path"] = poppler_path
            images = pdf2image.convert_from_path(pdf_path, **kwargs)
            if images:
                return images
        except Exception as e:
            logger.error({
                "level": "ERROR",
                "type": "handler",
                "message": f"[ERROR] pdf2image conversion failed (Poppler missing or error). Falling back to PyMuPDF render. {e}",
                "session_id": session_id
            })
            images = []

    # Fallback: render via PyMuPDF (no Poppler needed)
    try:
        doc = fitz.open(pdf_path)
        total = len(doc)
        rng = range(total)
        if max_pages is not None:
            rng = range(min(max_pages, total))
        for i in rng:
            page = doc[i]
            # Render to pixmap at requested DPI for OCR
            pix = page.get_pixmap(dpi=dpi)
            mode = "RGBA" if pix.alpha else "RGB"
            pil_img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            images.append(pil_img)
        doc.close()
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "handler",
            "message": f"[ERROR] PyMuPDF render fallback failed: {e}",
            "session_id": session_id
        })
        images = []
    return images

def _prep_variants(images):
    """
    Yield (name, images_variant) for multiple preprocessing paths.
    """
    variants = []
    # identity
    variants.append(("none", images))
    # grayscale
    gray = [ImageOps.grayscale(img) for img in images]
    variants.append(("gray", gray))
    # adaptive threshold (simple)
    thresh = [ImageOps.autocontrast(ImageOps.grayscale(img)).point(lambda p: 255 if p > 180 else 0, mode='1') for img in images]
    variants.append(("thresh", thresh))
    # sharpen + contrast
    sharp = [ImageEnhance.Contrast(img.filter(ImageFilter.SHARPEN)).enhance(1.5) for img in gray]
    variants.append(("sharp_contrast", sharp))
    return variants

def _dedupe_contest_titles(titles: list[str]) -> list[str]:
    """
    Deduplicate contest-like titles by normalized token set and fuzzy containment.
    Keeps the first occurrence of near-duplicates.
    """
    def norm(s: str) -> str:
        s = (s or "").lower().strip()
        s = re.sub(r'[\s\-_\/]+', ' ', s)
        s = re.sub(r'[^a-z0-9 ]+', '', s)
        return re.sub(r'\s+', ' ', s).strip()
    seen = []
    out = []
    for t in (titles or []):
        nt = norm(t)
        if not nt:
            continue
        dup = False
        for st in seen:
            # treat as duplicate if token overlap is high or containment
            a = set(nt.split())
            b = set(st.split())
            inter = len(a & b)
            union = max(1, len(a | b))
            jacc = inter / union
            if jacc >= 0.85 or nt in st or st in nt:
                dup = True
                break
        if not dup:
            seen.append(nt)
            out.append(t)
    return out

def _ocr_images(images, tesseract_config: str, confidence_threshold=30):
    """
    Run pytesseract on a list of PIL images and return combined text and avg confidence.
    """
    if not pytesseract:
        return "", 0.0, []

    page_texts = []
    confs_all = []
    # Per-page confidences for debugging
    per_page = []

    for img in images:
        text = ""
        details = {}
        if hasattr(pytesseract, "Output"):
            try:
                details = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=tesseract_config)
            except Exception:
                # Fallback to plain text if data API fails
                try:
                    text = pytesseract.image_to_string(img, config=tesseract_config)
                except Exception:
                    text = ""
            else:
                # Prefer confidences from the data API when available
                words = details.get("text", []) or []
                confs = details.get("conf", []) or []
                for j in range(len(words)):
                    word = (words[j] or "").strip()
                    conf_raw = confs[j] if j < len(confs) else "-1"
                    if word:
                        try:
                            conf_val = float(conf_raw)
                        except Exception:
                            conf_val = -1.0
                        confs_all.append(conf_val)
                        if conf_val >= confidence_threshold:
                            text += word + " "
        else:
            try:
                text = pytesseract.image_to_string(img, config=tesseract_config)
            except Exception:
                text = ""

        page_texts.append(text)
        if details:
            vals = []
            for c in details.get("conf", []):
                try:
                    vals.append(float(c))
                except Exception:
                    pass
            per_page.append(sum(vals) / len(vals) if vals else 0.0)
        else:
            per_page.append(0.0)

    avg_conf = sum(confs_all) / len(confs_all) if confs_all else 0.0
    return "\n".join(page_texts), avg_conf, per_page

def adaptive_ocr_pipeline(pdf_path, session_id=None, target_conf=70.0, max_seconds=120, max_runs=20):
    """
    Adaptive OCR loop:
    - Try different DPIs, preprocessors, and Tesseract configs (psm/oem)
    - Keep the best result by avg confidence
    - Early stop on reaching target_conf or exceeding budgets
    Returns: best_text, best_conf, runs_summary(list of dict)
    """
    start = time.time()
    runs_summary = []
    best = {"text": "", "conf": 0.0, "params": {}}

    dpi_list = [200, 250, 300, 350]
    # Favor structured page mode first, then single-block variants
    psm_list = [6, 4, 3, 11, 12, 1, 13]
    # Try LSTM-only first, then default, then combo, then legacy-only
    # 1=LSTM only, 3=Default, 2=Legacy+LSTM, 0=Legacy only
    oem_list = [1, 3, 2, 0]
    conf_threshold_word = 30
    # Precompute sample page indices (first/middle/last up to 5 pages)
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
    except Exception:
        page_count = None
    if page_count and page_count > 0:
        pts = {0, max(0, page_count // 2), max(0, page_count - 1)}
        if page_count > 6:
            pts |= {page_count // 4, (3 * page_count) // 4}
        sample_indices = sorted({int(min(max(0, i), page_count - 1)) for i in pts})
    else:
        sample_indices = [0]
    # Caches to avoid rerendering
    cache_sample = {}  # dpi -> [PIL.Image]
    cache_full = {}    # dpi -> [PIL.Image]
    logger.info({
        "level": "INFO",
        "type": "handler",
        "message": f"[INFO] OCR param search on sample pages {sample_indices}; final pass on full document.",
        "session_id": session_id
    })

    for dpi in dpi_list:
        if time.time() - start > max_seconds or len(runs_summary) >= max_runs:
            break

        # Get sample images for trials (fast)
        if dpi not in cache_sample:
            cache_sample[dpi] = _pdf_to_images(pdf_path, session_id=session_id, dpi=dpi, page_indices=sample_indices)
        images = cache_sample[dpi]
        if not images:
            continue
        logger.debug({
            "level": "DEBUG",
            "type": "handler",
            "message": f"[DEBUG] OCR trials at dpi={dpi} on {len(images)} sample page(s).",
            "session_id": session_id
        })

        for prep_name, prep_imgs in _prep_variants(images):
            if time.time() - start > max_seconds or len(runs_summary) >= max_runs:
                break

            for oem in oem_list:
                for psm in psm_list:
                    if time.time() - start > max_seconds or len(runs_summary) >= max_runs:
                        break
                    config = f"--oem {oem} --psm {psm}"
                    text, avg_conf, per_page = _ocr_images(prep_imgs, config, confidence_threshold=conf_threshold_word)

                    # Record run
                    run = {
                        "dpi": dpi,
                        "prep": prep_name,
                        "oem": oem,
                        "psm": psm,
                        "avg_conf": round(avg_conf, 2),
                        "per_page": [round(c, 2) for c in per_page]
                    }
                    runs_summary.append(run)
                    if len(runs_summary) % 5 == 0:
                        logger.info({
                            "level": "INFO",
                            "type": "handler",
                            "message": f"[INFO] OCR trials progress: {len(runs_summary)} run(s), best={round(best['conf'],2)} conf",
                            "session_id": session_id
                        })
                    # Update best
                    if avg_conf > best["conf"]:
                        best = {"text": text, "conf": avg_conf, "params": {"dpi": dpi, "prep": prep_name, "oem": oem, "psm": psm}}

                    # Early stop when good enough
                    if avg_conf >= target_conf:
                        break
                else:
                    continue
                break
            else:
                continue
            break

    # Combine high-confidence lines across top runs to improve recall
    if runs_summary:
        # sort by confidence
        top = sorted(runs_summary, key=lambda r: r["avg_conf"], reverse=True)[:5]
        # Re-run OCR quickly for those top settings to collect lines
        line_sets = []
        for r in top:
            # Use sample images for quick combination
            if r["dpi"] not in cache_sample:
                cache_sample[r["dpi"]] = _pdf_to_images(pdf_path, session_id=session_id, dpi=r["dpi"], page_indices=sample_indices)
            imgs = cache_sample.get(r["dpi"]) or []
            if not imgs:
                continue
            # reapply preprocess
            prep_variants = dict(_prep_variants(imgs))
            imgs2 = prep_variants.get(r["prep"], imgs)
            txt, _, _ = _ocr_images(imgs2, f"--oem {r['oem']} --psm {r['psm']}", confidence_threshold=conf_threshold_word)
            line_sets.append(set((txt or "").splitlines()))
        if line_sets:
            combined = sorted(set.union(*line_sets))
            combined_text = "\n".join(combined)
            # Keep the better of combined vs. best raw
            if len(combined_text) > len(best["text"]):
                best["text"] = combined_text

    # Final assurance: run a full-document pass with the best params (covers all pages explicitly)
    try:
        params = best.get("params") or {}
        if params:
            dpi = params.get("dpi", 300)
            if dpi not in cache_full:
                cache_full[dpi] = _pdf_to_images(pdf_path, session_id=session_id, dpi=dpi)
            imgs = cache_full.get(dpi) or []
            prep_variants = dict(_prep_variants(imgs))
            imgs2 = prep_variants.get(params.get("prep", "none"), imgs)
            cfg = f"--oem {params.get('oem', 3)} --psm {params.get('psm', 6)}"
            # Use same confidence threshold variable as above
            text_full, _, _ = _ocr_images(imgs2, cfg, confidence_threshold=conf_threshold_word)
            # Prefer the longer of full pass vs. previously combined
            if text_full and len(text_full) > len(best["text"] or ""):
                best["text"] = text_full
            try:
                page_count = len(imgs2)
                logger.info({
                    "level": "INFO",
                    "type": "handler",
                    "message": f"[INFO] Final OCR full-document pass completed (pages={page_count}, dpi={params.get('dpi')}, prep={params.get('prep')}, oem={params.get('oem')}, psm={params.get('psm')}).",
                    "session_id": session_id
                })
            except Exception:
                pass
    except Exception:
        pass
    return best["text"], best["conf"], runs_summary, best.get("params", {})

def ocr_multi_pass(images, passes=3, confidence_threshold=30, session_id=None):
    ocr_runs = []
    pass_confidences = []

    def process_image_ocr(img):
        page_text = ""
        confidences = []
        if pytesseract:
            details = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT) if hasattr(pytesseract, "Output") else {}
            for j in range(len(details.get("text", []))):
                word = details["text"][j].strip()
                conf = details["conf"][j]
                if word:
                    try:
                        conf_val = float(conf)
                        confidences.append(conf_val)
                        if conf_val >= confidence_threshold:
                            page_text += word + " "
                    except ValueError:
                        continue
        return page_text, confidences

    for i in range(passes):
        logger.info({
            "level": "INFO",
            "type": "handler",
            "message": f"[INFO] OCR pass {i+1} of {passes}",
            "session_id": session_id
        })
        ocr_text = ""
        confidences = []
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_image_ocr, images))
        for text, conf_list in results:
            ocr_text += text + "\n"
            confidences.extend(conf_list)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        pass_confidences.append(avg_conf)
        ocr_runs.append(ocr_text)

    line_sets = [set(text.splitlines()) for text in ocr_runs]
    combined_lines = sorted(set.union(*line_sets))
    all_text = "\n".join(combined_lines)
    overall_avg = sum(pass_confidences) / len(pass_confidences) if pass_confidences else 0.0
    return all_text, overall_avg, ocr_runs

def _extract_text_multi(pdf_path, session_id=None):
    """
    Try multiple PyMuPDF extract modes and pick the longest.
    Only use modes that return strings.
    """
    try:
        doc = fitz.open(pdf_path)
        texts = {}
        # use string-returning modes only
        modes = ["text", "raw", "html", "xhtml"]
        for m in modes:
            buf = []
            for i in range(len(doc)):
                try:
                    t = doc[i].get_text(m)
                    if not isinstance(t, str):
                        t = ""
                    buf.append(t)
                except Exception:
                    continue
            texts[m] = "\n".join(buf)
        doc.close()
        best_mode = max(texts, key=lambda k: len(texts.get(k) or ""))
        return texts.get(best_mode) or "", best_mode
    except Exception as e:
        logger.warning({
            "level": "WARNING",
            "type": "handler",
            "message": f"[WARN] Multi-mode text extraction failed: {e}",
            "session_id": session_id
        })
        return "", "error"

def _save_ocr_debug_images(pdf_path, session_id=None, dpi=300, limit=2):
    try:
        # Render only first N pages instead of rasterizing the entire document
        idxs = list(range(max(0, limit)))
        imgs = _pdf_to_images(pdf_path, session_id=session_id, dpi=dpi, page_indices=idxs)
        saved = []
        base = safe_slug(os.path.basename(pdf_path))
        for idx, img in enumerate(imgs):
            out = os.path.join(
                OCR_DEBUG_DIR,
                f"{base}_p{idx+1}_{dpi}dpi.png"
            )
            try:
                img.save(out)
                saved.append(str(out))
            except Exception:
                pass
        if saved:
            logger.info({
                "level": "INFO",
                "type": "handler",
                "message": "[DEBUG] Saved OCR debug raster(s)",
                "session_id": session_id,
                "files": saved
            })
    except Exception:
        pass


def _write_debug_text(pdf_path: str, text: str | None, suffix: str, session_id=None) -> str | None:
    """Persist sanitized/diagnostic OCR text for manual inspection."""
    if not text:
        return None
    try:
        base = safe_slug(os.path.splitext(os.path.basename(pdf_path))[0])
        out_file = os.path.join(OCR_DEBUG_DIR, f"{base}__{suffix}.txt")
        with open(out_file, "w", encoding="utf-8") as handle:
            handle.write(text)
        logger.debug({
            "level": "DEBUG",
            "type": "handler",
            "message": f"[DEBUG] Wrote OCR debug text ({suffix}) to {out_file}",
            "session_id": session_id
        })
        return out_file
    except Exception:
        return None


_LAYOUT_HEADER_KEYWORDS = {
    "candidate", "party", "district", "ward", "precinct", "total", "votes", "vote",
    "absentee", "affidavit", "military", "emergency", "machine", "counter", "early",
    "mail", "provisional", "ballots", "election", "day", "overall", "write", "count"
}

_NUMERIC_VALUE_RE = re.compile(r"^-?\d[\d,]*(?:\.\d+)?%?$")


def _header_token_score(line_text: str) -> int:
    tokens = _token_set(line_text)
    return sum(1 for tok in tokens if tok in _LAYOUT_HEADER_KEYWORDS)


def _line_has_digits(words: list[dict]) -> bool:
    for w in words:
        if _NUM_RE.search(w.get("text", "")):
            return True
    return False


def _line_is_candidate_only(line_text: str) -> bool:
    if not line_text:
        return False
    digits = sum(ch.isdigit() for ch in line_text)
    letters = sum(ch.isalpha() for ch in line_text)
    return digits == 0 and letters >= 6


def _group_words_by_gaps(words: list[dict], gap_threshold: int = 28) -> list[list[dict]]:
    groups: list[list[dict]] = []
    current: list[dict] = []
    prev_right: float | None = None
    for word in sorted(words, key=lambda w: w.get("left", 0)):
        if prev_right is not None and word.get("left", 0) - prev_right > gap_threshold:
            if current:
                groups.append(current)
            current = []
        current.append(word)
        prev_right = word.get("right", word.get("left", 0))
    if current:
        groups.append(current)
    return groups


def _build_layout_columns(header_words: list[dict], image_width: int) -> list[dict]:
    grouped = _group_words_by_gaps(header_words)
    columns: list[dict] = []
    seen: set[str] = set()
    for idx, group in enumerate(grouped):
        raw_label = " ".join(w.get("text", "").strip() for w in group if w.get("text"))
        clean_label = re.sub(r"\s+", " ", raw_label).strip() or f"Column {idx + 1}"
        base_key = clean_label.lower()
        if base_key in seen:
            suffix = 2
            while f"{base_key}__{suffix}" in seen:
                suffix += 1
            clean_label = f"{clean_label} {suffix}"
            base_key = f"{base_key}__{suffix}"
        seen.add(base_key)
        left = min(w.get("left", 0) for w in group)
        right = max((w.get("right") or (w.get("left", 0) + w.get("width", 0))) for w in group)
        columns.append({
            "label": clean_label,
            "key": clean_label,
            "left": max(0, left - 4),
            "right": right + 6,
        })
    if not columns:
        return []
    columns[0]["left"] = 0
    columns[-1]["right"] = max(image_width, int(columns[-1]["right"]))
    return columns


def _assign_words_to_columns(columns: list[dict], words: list[dict]) -> dict:
    row: dict[str, str] = {col["label"]: "" for col in columns}
    if not columns:
        return row
    for word in sorted(words, key=lambda w: w.get("left", 0)):
        text = (word.get("text") or "").strip()
        if not text:
            continue
        center = word.get("center_x")
        if center is None:
            center = word.get("left", 0) + word.get("width", 0) / 2
        target = None
        for col in columns:
            if col["left"] - 2 <= center <= col["right"] + 2:
                target = col
                break
        if target is None:
            target = columns[0] if center < columns[0]["left"] else columns[-1]
        existing = row[target["label"]]
        row[target["label"]] = f"{existing} {text}".strip() if existing else text
    return row


def _clean_numeric_cell(value: str) -> str | int:
    if not isinstance(value, str):
        return value
    raw = value.strip()
    if not raw:
        return ""
    pct = False
    if raw.endswith("%"):
        pct = True
        raw = raw[:-1]
    cleaned = raw.replace(",", "").strip()
    if cleaned.isdigit():
        num = int(cleaned)
        return f"{num}%" if pct else num
    return value


def _split_key_value_line(text: str) -> tuple[str, str] | None:
    """Split a 'Label value' line emitted by statement-and-return style pages."""
    if not text:
        return None
    parts = text.rsplit(" ", 1)
    if len(parts) != 2:
        return None
    label, value = parts[0].strip().rstrip(":"), parts[1].strip()
    if not label or not value:
        return None
    if not _NUMERIC_VALUE_RE.match(value):
        return None
    return label, value


def _finalize_layout_table(columns: list[dict], rows: list[dict]) -> tuple[list[str], list[dict]]:
    headers = [col["label"] for col in columns]
    normalized_rows: list[dict] = []
    for row in rows:
        normalized = {}
        for col in columns:
            val = row.get(col["label"], "")
            if col is columns[0]:
                normalized[col["label"]] = re.sub(r"\s+", " ", val.strip())
                continue
            normalized[col["label"]] = _clean_numeric_cell(val)
        normalized_rows.append(normalized)
    return headers, normalized_rows


def _merge_layout_tables(tables: list[dict]) -> list[dict]:
    if not tables:
        return tables
    buckets: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for table in tables:
        key = tuple(h.lower() for h in table.get("headers", []))
        buckets[key].append(table)
    merged: list[dict] = []
    for key, group in buckets.items():
        if len(group) == 1:
            merged.append(group[0])
            continue
        rows: list[dict] = []
        for item in group:
            rows.extend(item.get("rows", []))
        merged.append({
            "headers": group[0].get("headers", []),
            "rows": rows,
            "pages": sorted({g.get("page") for g in group if g.get("page") is not None}),
            "score": sum(len(g.get("rows", [])) for g in group)
        })
    merged.sort(key=lambda t: (-len(t.get("rows", [])), -len(t.get("headers", []))))
    return merged


def _extract_tables_via_layout(pdf_path: str, session_id=None, ocr_params: dict | None = None, max_pages: int | None = None):
    if not pytesseract or not _PANDAS_AVAILABLE:
        return []
    try:
        tess_output = pytesseract.Output.DATAFRAME
    except Exception:
        return []

    dpi = int((ocr_params or {}).get("dpi", 300))
    oem = (ocr_params or {}).get("oem", 1)
    psm = (ocr_params or {}).get("psm", 6)
    config = f"--oem {oem} --psm {psm} -c preserve_interword_spaces=1"

    images = _pdf_to_images(pdf_path, session_id=session_id, dpi=dpi, max_pages=max_pages)
    layout_tables: list[dict] = []

    for page_index, image in enumerate(images):
        try:
            df = pytesseract.image_to_data(image, output_type=tess_output, config=config)
        except Exception as exc:
            logger.debug({
                "level": "DEBUG",
                "type": "handler",
                "message": f"[DEBUG] Tesseract DATAFRAME extraction failed on page {page_index}: {exc}",
                "session_id": session_id
            })
            continue

        if df is None or df.empty:
            continue
        try:
            df = df[df["conf"].fillna(-1) > -1]
            df["text"] = df["text"].fillna("").astype(str).str.strip()
            df = df[df["text"] != ""]
        except Exception:
            continue
        if df.empty:
            continue

        df["right"] = df["left"] + df["width"]
        df["center_x"] = df["left"] + (df["width"] / 2)
        df["center_y"] = df["top"] + (df["height"] / 2)

        line_records = []
        group_cols = ["page_num", "block_num", "par_num", "line_num"]
        try:
            grouped = df.groupby(group_cols)
        except Exception:
            continue

        for (_, _, _, line_num), group in grouped:
            if group.empty:
                continue
            words: list[dict] = []
            for _, row in group.sort_values("left").iterrows():
                text = str(row.get("text", "")).strip()
                if not text:
                    continue
                left = int(row.get("left", 0))
                width = int(row.get("width", 0))
                words.append({
                    "text": text,
                    "left": left,
                    "right": left + width,
                    "width": width,
                    "center_x": float(row.get("center_x", left + width / 2)),
                })
            if not words:
                continue
            text = " ".join(w["text"] for w in words)
            line_records.append({
                "words": words,
                "text": text,
                "page": page_index,
                "avg_top": float(group["top"].mean()),
                "image_width": image.width,
            })

        line_records.sort(key=lambda item: (item["page"], item["avg_top"]))

        current_table: dict | None = None
        pending_candidate_words: list[dict] | None = None
        blank_streak = 0

        for line in line_records:
            text = line["text"]
            words = line["words"]
            header_score = _header_token_score(text)
            digits_present = _line_has_digits(words)

            if header_score >= 2 and not _is_bad_header_line(text):
                if current_table and current_table.get("rows"):
                    headers, rows = _finalize_layout_table(current_table["columns"], current_table["rows"])
                    layout_tables.append({
                        "headers": headers,
                        "rows": rows,
                        "page": current_table.get("page"),
                        "score": len(rows)
                    })
                columns = _build_layout_columns(words, line["image_width"])
                if len(columns) < 2:
                    current_table = None
                    pending_candidate_words = None
                    continue
                current_table = {
                    "columns": columns,
                    "rows": [],
                    "page": page_index,
                }
                pending_candidate_words = None
                blank_streak = 0
                continue

            if not current_table:
                continue

            if not text.strip():
                blank_streak += 1
                if blank_streak >= 2 and current_table.get("rows"):
                    headers, rows = _finalize_layout_table(current_table["columns"], current_table["rows"])
                    layout_tables.append({
                        "headers": headers,
                        "rows": rows,
                        "page": current_table.get("page"),
                        "score": len(rows)
                    })
                    current_table = None
                    pending_candidate_words = None
                continue

            blank_streak = 0

            if digits_present:
                active_words = list(words)
                if pending_candidate_words:
                    active_words = pending_candidate_words + active_words
                row = _assign_words_to_columns(current_table["columns"], active_words)
                if any(row.get(col["label"]) for col in current_table["columns"]):
                    current_table["rows"].append(row)
                pending_candidate_words = None
            else:
                if _line_is_candidate_only(text):
                    pending_candidate_words = list(words)
                    continue
                if current_table["rows"]:
                    first_col = current_table["columns"][0]["label"]
                    existing = current_table["rows"][-1].get(first_col, "")
                    current_table["rows"][-1][first_col] = f"{existing} {text}".strip()

        if current_table and current_table.get("rows"):
            headers, rows = _finalize_layout_table(current_table["columns"], current_table["rows"])
            layout_tables.append({
                "headers": headers,
                "rows": rows,
                "page": current_table.get("page"),
                "score": len(rows)
            })

    return _merge_layout_tables(layout_tables)


_STATEMENT_AD_RE = re.compile(r"assembly\s+district\s+(\d+)", re.I)
_STATEMENT_ED_RE = re.compile(r"election\s+district\s+(\d+)", re.I)
_STATEMENT_PRECINCT_RE = re.compile(r"precinct\s+(\d+)", re.I)

_STATEMENT_SUMMARY_KEYWORDS = (
    "public counter",
    "manually counted",
    "absentee",
    "military",
    "affidavit",
    "total ballots",
    "total votes",
    "total vote",
    "total applicable",
    "less - inapplicable",
    "inapplicable",
    "unrecorded",
    "vote for",
    "scanner",
    "ballots cast",
    "ballots counted",
)

def _identify_statement_location_columns(headers: list[str]) -> list[str]:
    return collect_location_headers(headers or [], ensure_precinct=False)


def _statement_value_has_payload(value: object) -> bool:
    if value in (None, ""):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return False
        core = (
            stripped.replace(",", "")
            .replace("%", "")
            .replace("-", "")
            .replace("(", "")
            .replace(")", "")
        )
        if core.replace(".", "", 1).isdigit():
            return True
        return len(stripped) >= 3
    return True


def _coerce_statement_numeric(value: object) -> object:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except Exception:
            return value
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if cleaned.replace(".", "", 1).isdigit():
            try:
                return int(float(cleaned))
            except Exception:
                return value
    return value


def _remap_statement_summary_header(header: str) -> str:
    low = (header or "").strip().lower()
    if "public counter" in low:
        return "Precinct Public Counter"
    if "manually counted" in low and "emergency" in low:
        return "Precinct Emergency Ballots"
    if "absentee" in low or "military" in low:
        return "Precinct Absentee Military"
    if "affidavit" in low:
        return "Precinct Affidavit"
    if "total applicable" in low:
        return "Precinct Total Applicable Ballots"
    if "less - inapplicable" in low or "inapplicable" in low:
        return "Precinct Inapplicable Ballots"
    if low == "total vote" or "total votes" in low:
        return "Precinct Total Votes"
    if "total ballots" in low:
        return "Precinct Total Ballots"
    if "unrecorded" in low:
        return "Precinct Unrecorded"
    return header


def _is_statement_summary_header(header: str) -> bool:
    if is_strict_location_header(header):
        return False
    low = (header or "").strip().lower()
    if not low:
        return False
    return any(token in low for token in _STATEMENT_SUMMARY_KEYWORDS)


def _parse_statement_candidate_header(header: str) -> tuple[str, str | None, str | None]:
    text = re.sub(r"\s+", " ", (header or "").strip())
    candidate_type = "Write-In" if re.search(r"write[-\s]?in", text, re.I) else "Candidate"
    party = None

    without_write_in = re.sub(r"\s*\(write[-\s]?in\)\s*", "", text, flags=re.I)
    without_write_in = re.sub(r"/\s*write[-\s]?in", "", without_write_in, flags=re.I)

    party_match = re.search(r"\(([^)]+)\)$", without_write_in)
    if party_match and not re.search(r"write", party_match.group(1), re.I):
        party = party_match.group(1).strip()
        candidate_label = without_write_in[: party_match.start()].strip()
    else:
        candidate_label = without_write_in.strip()

    if candidate_label.isupper():
        candidate_label = candidate_label.title()

    return candidate_label or text, candidate_type, party


def _normalize_statement_candidate_results(
    headers: list[str],
    rows: list[dict],
) -> tuple[list[str], list[dict], dict]:
    if not headers or not rows:
        return [], [], {}

    location_cols = _identify_statement_location_columns(headers)
    if "Precinct" not in location_cols:
        location_cols = ["Precinct", *location_cols]
    summary_cols = [h for h in headers if _is_statement_summary_header(h)]
    base_cols = [h for h in location_cols if h in headers or h == "Precinct"]
    candidate_cols = [h for h in headers if h not in base_cols and h not in summary_cols]

    if not candidate_cols:
        return [], [], {}

    summary_targets: dict[str, str] = {}
    summary_order: list[str] = []
    for col in summary_cols:
        mapped = _remap_statement_summary_header(col)
        summary_targets[col] = mapped
        if mapped not in summary_order:
            summary_order.append(mapped)

    normalized_rows: list[dict] = []
    for rec in rows:
        base = {col: rec.get(col, "") for col in base_cols}
        for cand in candidate_cols:
            value = rec.get(cand, "")
            if not _statement_value_has_payload(value):
                continue
            candidate_label, candidate_type, candidate_party = _parse_statement_candidate_header(cand)
            normalized = dict(base)
            normalized["Candidate"] = candidate_label
            if candidate_type:
                normalized["Candidate Type"] = candidate_type
            if candidate_party:
                normalized["Party"] = candidate_party
            normalized["Votes"] = _coerce_statement_numeric(value)
            for src, dest in summary_targets.items():
                summary_val = rec.get(src, "")
                if not _statement_value_has_payload(summary_val):
                    continue
                normalized[dest] = _coerce_statement_numeric(summary_val)
            normalized_rows.append(normalized)

    if not normalized_rows:
        return [], [], {}

    header_candidates = []
    if "Precinct" in location_cols:
        header_candidates.append("Precinct")
    for loc in location_cols:
        if loc != "Precinct":
            header_candidates.append(loc)
    header_candidates.append("Candidate")
    if any(row.get("Candidate Type") for row in normalized_rows):
        header_candidates.append("Candidate Type")
    if any(row.get("Party") for row in normalized_rows):
        header_candidates.append("Party")
    header_candidates.append("Votes")
    header_candidates.extend(h for h in summary_order if h not in header_candidates)

    keep_headers: list[str] = []
    for header in header_candidates:
        if header in {"Precinct", "Assembly District", "Election District", "Candidate", "Votes"}:
            keep_headers.append(header)
            continue
        if header in {"Candidate Type", "Party"}:
            if any(row.get(header) for row in normalized_rows):
                keep_headers.append(header)
            continue
        if any(_statement_value_has_payload(row.get(header)) for row in normalized_rows):
            keep_headers.append(header)

    finalized_rows = [{h: row.get(h, "") for h in keep_headers} for row in normalized_rows]

    diagnostics = {
        "source_headers": len(headers),
        "candidate_columns": len(candidate_cols),
        "summary_columns": len(summary_cols),
        "normalized_rows": len(finalized_rows),
        "location_headers": location_cols,
    }

    return keep_headers, finalized_rows, diagnostics


def _extract_statement_return_blocks(pdf_path: str, session_id=None, ocr_params: dict | None = None, max_pages: int | None = None):
    """Parse statement & return style PDF pages into structured key/value rows."""
    if not pytesseract:
        return [], []

    dpi = int(max(360, (ocr_params or {}).get("dpi", 300)))
    oem = (ocr_params or {}).get("oem", 3)
    config = f"--oem {oem} --psm 4 -c preserve_interword_spaces=1"

    images = _pdf_to_images(pdf_path, session_id=session_id, dpi=dpi, max_pages=max_pages)
    if not images:
        return [], []

    records_map: dict[tuple[str, str, str], dict] = {}
    current_record: dict[str, object] | None = None
    current_ad: str | None = None
    current_ed: str | None = None

    def _ensure_record() -> bool:
        nonlocal current_record
        if current_record is None:
            if not (current_ad or current_ed):
                return False
            current_record = {"Assembly District": current_ad or ""}
            if current_ed:
                current_record["Election District"] = current_ed
        else:
            if current_ad and not current_record.get("Assembly District"):
                current_record["Assembly District"] = current_ad
            if current_ed and not current_record.get("Election District"):
                current_record["Election District"] = current_ed
        return True

    def commit_record():
        nonlocal current_record
        if not current_record:
            return
        ad = str(current_record.get("Assembly District") or "").strip()
        ed = str(current_record.get("Election District") or "").strip()
        precinct_label = str(current_record.get("Precinct") or "").strip()
        if not ad:
            current_record = None
            return
        has_payload = any(
            _statement_value_has_payload(value)
            for key, value in current_record.items()
            if key not in {"Assembly District", "Election District", "Precinct"}
        )
        if not has_payload and not ed and not precinct_label:
            current_record = None
            return
        key = (ad, ed, precinct_label)
        bucket = records_map.setdefault(key, {"Assembly District": ad})
        if ed:
            bucket["Election District"] = ed
        if precinct_label:
            bucket["Precinct"] = precinct_label
        for k, v in current_record.items():
            if isinstance(v, str) and not v.strip():
                continue
            if v in (None, ""):
                continue
            bucket[k] = v
        current_record = None

    for page_index, image in enumerate(images):
        try:
            df = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME, config=config)
        except Exception as exc:
            logger.debug({
                "level": "DEBUG",
                "type": "handler",
                "message": f"[DEBUG] Statement block OCR failed on page {page_index}: {exc}",
                "session_id": session_id
            })
            continue

        if df is None or df.empty:
            continue
        df = df[df["text"].notna()]
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"] != ""]
        if df.empty:
            continue

        df = df.assign(center_x=df["left"] + (df["width"] / 2))
        group_cols = ["page_num", "block_num", "par_num", "line_num"]
        try:
            grouped = df.groupby(group_cols)
        except Exception:
            continue

        # Sort by layout ordering
        lines = []
        for _, group in grouped:
            words = []
            for _, row in group.sort_values("left").iterrows():
                token = str(row.get("text", "")).strip()
                if not token:
                    continue
                words.append(token)
            if not words:
                continue
            text = re.sub(r"\s+", " ", " ".join(words)).strip()
            if text:
                top = float(group["top"].mean())
                lines.append((top, text))

        lines.sort(key=lambda item: item[0])

        for _, raw_text in lines:
            text = raw_text.strip()
            if not text:
                continue
            # Skip boilerplate page indicators
            low = text.lower()
            if low.startswith("page ") and " of " in low:
                continue

            ad_match = _STATEMENT_AD_RE.search(text)
            if ad_match:
                commit_record()
                current_ad = ad_match.group(1)
                current_ed = None
                current_record = {"Assembly District": current_ad}
                continue

            if current_ad is None and current_record is None:
                # Ignore until we know which assembly the block belongs to
                continue

            ed_match = _STATEMENT_ED_RE.search(text)
            if ed_match:
                commit_record()
                current_ed = ed_match.group(1)
                current_record = {"Assembly District": current_ad or "", "Election District": current_ed}
                continue

            prec_match = _STATEMENT_PRECINCT_RE.search(text)
            if prec_match:
                if _ensure_record():
                    current_record["Precinct"] = prec_match.group(1)
                continue

            key_val = _split_key_value_line(text)
            if not key_val:
                continue

            label, value_token = key_val
            parsed_value = _clean_numeric_cell(value_token)
            if _ensure_record():
                current_record[label] = parsed_value

        # Continue accumulating across pages; do not commit yet to allow multi-page sections

    commit_record()

    if not records_map:
        return [], []

    records = list(records_map.values())

    if not records:
        return [], []

    def _sort_key(rec: dict) -> tuple:
        ad_raw = str(rec.get("Assembly District", "")).strip()
        ed_raw = str(rec.get("Election District", "")).strip()
        def _coerce(val: str) -> tuple[int, str, str]:
            stripped = (val or "").strip()
            if stripped.isdigit():
                padded = stripped.zfill(6)
                return (0, padded, stripped)
            return (1, stripped.lower(), stripped)
        return (_coerce(ad_raw), _coerce(ed_raw))

    try:
        records.sort(key=_sort_key)
    except Exception:
        records.sort(key=lambda r: (str(r.get("Assembly District", "")), str(r.get("Election District", ""))))

    ordered_headers: list[str] = []
    priority = ["Precinct", "Assembly District", "Election District"]
    for key in priority:
        ordered_headers.append(key)
    seen = {h for h in ordered_headers}
    for rec in records:
        for key in rec.keys():
            if key not in seen:
                ordered_headers.append(key)
                seen.add(key)

    rows = []
    for rec in records:
        row = {h: rec.get(h, "") for h in ordered_headers}
        rows.append(row)

    return ordered_headers, rows


def _attach_statement_precinct(
    headers: list[str],
    rows: list[dict],
    *,
    location_headers: list[str] | None = None,
) -> tuple[list[str], list[dict], bool]:
    """Ensure statement rows expose a precinct-like label for downstream pivots."""
    if not rows:
        return headers, rows, False

    sanitized_extras = [
        header.strip()
        for header in (location_headers or [])
        if isinstance(header, str) and header.strip()
    ]

    detected = collect_location_headers(
        headers or [],
        ensure_precinct=True,
        extra_headers=sanitized_extras,
    )

    updated_headers, updated_rows, added_any = attach_precinct_column(
        list(headers or []),
        rows,
        location_headers=detected,
        column_name="Precinct",
    )

    return updated_headers, updated_rows, added_any


def _finalize_structured_table_output(
    pdf_path: str,
    headers: list[str],
    data: list[dict],
    selected_contest_title: str,
    state: str,
    county: str,
    year: int | None,
    contest_slug: str,
    metadata: dict,
    coordinator,
    session_id=None,
):
    statement_used = bool(metadata.get("statement_blocks_used"))
    precinct_attached = False
    if statement_used:
        diag_info = metadata.get("statement_blocks_normalized")
        location_headers = list(diag_info.get("location_headers") or []) if isinstance(diag_info, dict) else []
        headers, data, precinct_attached = _attach_statement_precinct(
            headers,
            data,
            location_headers=location_headers,
        )

    headers_adj, data_adj = harmonize_headers_and_data(
        headers,
        data,
        context={
            "contest": selected_contest_title,
            "state": state,
            "county": county,
        }
    )

    domain = safe_slug(os.path.basename(pdf_path))
    context = {
        "contest": selected_contest_title,
        "state": state,
        "county": county,
        "year": year,
        "session_id": session_id,
        "handler": "pdf_handler",
        "ocr_confidence_avg": metadata.get("ocr_confidence_avg"),
        "ocr_used": metadata.get("ocr_used"),
        "contest_slug": contest_slug,
        "source_slug": domain
    }

    if statement_used:
        context.setdefault("include_all_precincts_row", False)
        context["skip_pivot"] = True
        context["skip_row_noise_filter"] = True
        if precinct_attached:
            context.setdefault("precinct_sort", "natural")
        metadata["statement_blocks_precinct_attached"] = precinct_attached

    headers_exp, data_exp = expand_single_rawjson_row(headers_adj, data_adj, context=context)

    headers_final, data_final, _entity_info = build_table_noninteractive(
        domain=domain,
        headers=headers_exp,
        data=data_exp,
        coordinator=coordinator,
        context=context,
        pivot_to_wide=True,
        debug=False
    )

    export_context = _prepare_output_context(
        context,
        {
            "handler": "pdf_handler",
            "input_file": os.path.basename(pdf_path),
            "session_id": session_id,
            "ocr_confidence_avg": metadata.get("ocr_confidence_avg"),
            "ocr_used": metadata.get("ocr_used"),
        }
    )

    result = finalize_election_output(
        headers=headers_final,
        data=data_final,
        coordinator=coordinator,
        contest=selected_contest_title,
        state=state,
        county=county,
        context=export_context,
        enable_user_feedback=False,
        session_id=session_id
    )

    metadata.update({
        "output_file": os.path.basename(result.get("csv_path", "")),
        "headers": headers_final,
        "row_count": len(data_final),
        "csv_path": result.get("csv_path"),
        "metadata_path": result.get("metadata_path"),
    })

    logger.info({
        "level": "INFO",
        "type": "output",
        "message": f"[OUTPUT] Wrote {len(data_final)} rows to: {result.get('csv_path')}",
        "session_id": session_id
    })
    logger.info({
        "level": "INFO",
        "type": "output",
        "message": f"[OUTPUT] Metadata written to: {result.get('metadata_path')}",
        "session_id": session_id
    })

    return headers_final, data_final, result

def infer_headers_and_methods(lines, table_hints):
    hints = {h.lower() for h in (table_hints or []) if isinstance(h, str)}
    headers, header_idx = _find_header_line(lines, hints)
    return headers, (lines[header_idx] if header_idx >= 0 else "")

def _should_force_ocr(raw_text: str, clean_text: str) -> bool:
    """
    Force OCR if the sanitized signal is too small relative to raw text length,
    even if fitz returned non-empty content (common with markup/xhtml dumps).
    """
    try:
        raw_len = len(raw_text or "")
        clean_len = len(clean_text or "")
        if raw_len == 0:
            return False
        # Force when clean text is tiny compared to raw or trivially short
        ratio = clean_len / max(1, raw_len)
        return (clean_len < 500 and raw_len > 100_000) or ratio < 0.005
    except Exception:
        return False

def _should_auto_select(titles: list[str]) -> bool:
    """
    True if multiple detected titles are effectively near-duplicates
    (i.e., one contest repeated across the PDF).
    Uses token-set Jaccard similarity across all pairs.
    """
    def norm_tokens(s: str) -> set[str]:
        s = (s or "").lower()
        s = re.sub(r'[^a-z0-9 ]+', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return set(s.split())
    if not titles or len(titles) <= 1:
        return True
    toks = [norm_tokens(t) for t in titles if isinstance(t, str)]
    toks = [t for t in toks if t]
    if len(toks) <= 1:
        return True
    # Require high similarity across all pairs
    thresh = 0.80
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            a, b = toks[i], toks[j]
            inter = len(a & b)
            union = max(1, len(a | b))
            jacc = inter / union
            if jacc < thresh:
                return False
    return True

def _pick_representative_title(titles: list[str]) -> str:
    """
    Pick a stable representative from near-duplicates:
    prefer shortest meaningful title with max token overlap to others.
    """
    def norm_tokens(s: str) -> set[str]:
        s = (s or "").lower()
        s = re.sub(r'[^a-z0-9 ]+', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return set(s.split())
    if not titles:
        return ""
    toks = [norm_tokens(t) for t in titles]
    # Score: average token overlap with others, then length as tiebreaker
    best = (float("-inf"), "")
    for idx, t in enumerate(titles):
        a = toks[idx]
        if not a:
            continue
        score = 0.0
        for j, b in enumerate(toks):
            if j == idx or not b:
                continue
            inter = len(a & b)
            union = max(1, len(a | b))
            score += inter / union
        # Prefer shorter representative on tie
        key = (score, -len(t.strip()))
        if key > (best[0], -len(best[1].strip())):
            best = (score, t)
    return best[1] or titles[0]
    
def parse_pdf_election_results(pdf_path, session_id=None, coordinator=None) -> tuple[list[str], list[dict], str, dict]:
    """ Main PDF handler function."""
    _log_ocr_environment(session_id=session_id)
    all_text = ""
    metadata = {}

    global _FITZ_WARNING_LOGGED
    if _FITZ_IMPORT_WARNINGS and not _FITZ_WARNING_LOGGED:
        _FITZ_WARNING_LOGGED = True
        logger.warning({
            "level": "WARNING",
            "type": "handler",
            "message": "[WARN] PyMuPDF import emitted SWIG DeprecationWarnings; using safe import shim to avoid crashes under -W error.",
            "session_id": session_id,
            "warning_details": list(_FITZ_IMPORT_WARNINGS),
        })
    if _FITZ_IMPORT_WARNINGS:
        metadata["fitz_import_warnings"] = list(_FITZ_IMPORT_WARNINGS)
    if _FITZ_PATCHED_TYPES:
        metadata["fitz_patched_types"] = list(_FITZ_PATCHED_TYPES)
    if _FITZ_PATCH_FAILURES:
        metadata["fitz_patch_failures"] = list(_FITZ_PATCH_FAILURES)
    headers = []
    ocr_score = 0.0
    ocr_runs = []

    # Try standard text first
    try:
        doc = fitz.open(pdf_path)
        for i in range(len(doc)):
            all_text += doc[i].get_text()
        doc.close()
    except Exception as e:
        logger.warning({
            "level": "WARNING",
            "type": "handler",
            "message": f"[WARN] fitz text extraction failed: {e}",
            "session_id": session_id
        })
        all_text = ""

    # If empty or forced, try alternative extract modes
    if (not all_text.strip()) or ENABLE_OCR_FORCE:
        alt_text, mode_used = _extract_text_multi(pdf_path, session_id=session_id)
        if len(alt_text) > len(all_text):
            all_text = alt_text
            metadata["fitz_mode_used"] = mode_used

    # If the "text" is markup-only, treat as empty to force OCR
    if _is_mostly_markup(all_text):
        logger.info({
            "level": "INFO",
            "type": "handler",
            "message": "[INFO] Detected markup-only PDF text — switching to OCR.",
            "session_id": session_id
        })
        all_text = ""

    # OCR fallback (adaptive, cross‑platform)
    has_text = bool((all_text or "").strip())
    need_ocr = (not has_text) and bool(pytesseract) and ENABLE_OCR

    # If forcing OCR but pytesseract is unavailable, log and skip the loop
    if (not pytesseract) and ENABLE_OCR_FORCE:
        logger.warning({
            "level": "WARNING",
            "type": "handler",
            "message": "[WARN] ENABLE_OCR_FORCE is set but Tesseract is unavailable; skipping OCR fallback.",
            "session_id": session_id
        })

    if need_ocr or (ENABLE_OCR_FORCE and pytesseract):
        if not has_text:
            logger.info({
                "level": "INFO",
                "type": "handler",
                "message": "[INFO] Empty/forced OCR — attempting adaptive OCR fallback.",
                "session_id": session_id
            })
        _save_ocr_debug_images(pdf_path, session_id=session_id, dpi=300, limit=2)
        best_text, best_conf, runs_summary, ocr_params = adaptive_ocr_pipeline(
            pdf_path,
            session_id=session_id,
            target_conf=70.0,
            max_seconds=150,
            max_runs=28
        )
        candidate_ocr_params = dict(ocr_params or {})
        if candidate_ocr_params:
            metadata["ocr_params"] = candidate_ocr_params
        # Prefer OCR text if we had none, or if significantly better
        if (not has_text) or (len(best_text) > len(all_text) * 1.25):
            all_text = best_text or all_text
            ocr_score = best_conf or 0.0
            ocr_runs = runs_summary or []
            metadata["ocr_confidence_avg"] = round(ocr_score, 2)
            metadata["ocr_runs"] = ocr_runs
            metadata["ocr_used"] = True
            metadata["ocr_params"] = dict(ocr_params or {})
        else:
            metadata["ocr_used"] = False

    clean_text = _sanitize_extracted_text(all_text)
    try:
        logger.info({
            "level": "INFO",
            "type": "handler",
            "message": f"[INFO] Text lengths — raw={len(all_text or '')}, clean={len(clean_text or '')}",
            "session_id": session_id
        })
    except Exception:
        pass

    if clean_text:
        debug_path = _write_debug_text(pdf_path, clean_text, "clean", session_id=session_id)
        if debug_path:
            metadata["ocr_clean_text_path"] = debug_path
    if all_text and len(all_text) <= 2_500_000:
        raw_debug_path = _write_debug_text(pdf_path, all_text, "raw", session_id=session_id)
        if raw_debug_path:
            metadata["ocr_raw_text_path"] = raw_debug_path
    if not clean_text and all_text:
        # If sanitization nuked everything (e.g., fully-tagged), keep minimal fallback
        clean_text = os.path.splitext(os.path.basename(pdf_path))[0]

    # Force OCR when sanitized signal is very low vs raw fitz text (markup dump case)
    low_signal_force = False
    try_low_signal = _should_force_ocr(all_text, clean_text)
    ocr_already_used = bool(metadata.get("ocr_used"))
    if try_low_signal:
        low_signal_force = True
        metadata["ocr_low_signal"] = True
        if not pytesseract or not ENABLE_OCR:
            # Record why we can't run OCR in this scenario
            metadata["ocr_used"] = False
            metadata["ocr_reason"] = "low_signal_but_ocr_disabled"
            logger.warning({
                "level": "WARNING",
                "type": "handler",
                "message": "[WARN] Low-signal text detected but OCR is unavailable or disabled.",
                "session_id": session_id
            })
        elif not ocr_already_used:
            logger.info({
                "level": "INFO",
                "type": "handler",
                "message": "[INFO] Low-signal text detected from fitz (markup-heavy). Forcing OCR.",
                "session_id": session_id
            })
            _save_ocr_debug_images(pdf_path, session_id=session_id, dpi=300, limit=2)
            best_text, best_conf, runs_summary, ocr_params = adaptive_ocr_pipeline(
                pdf_path,
                session_id=session_id,
                target_conf=70.0,
                max_seconds=150,
                max_runs=28
            )
            candidate_ocr_params = dict(ocr_params or {})
            if candidate_ocr_params:
                metadata["ocr_params"] = candidate_ocr_params
            if best_text:
                all_text = best_text
                clean_text = _sanitize_extracted_text(all_text)
                ocr_score = best_conf or 0.0
                ocr_runs = runs_summary or []
                metadata["ocr_confidence_avg"] = round(ocr_score, 2)
                metadata["ocr_runs"] = ocr_runs
                metadata["ocr_used"] = True
                metadata["ocr_params"] = dict(ocr_params or {})
                metadata["ocr_reason"] = "low_signal_fitx_markup"
            else:
                metadata["ocr_used"] = False
                metadata["ocr_reason"] = "low_signal_ocr_no_text"
        else:
            # OCR already performed earlier; do not run twice
            logger.info({
                "level": "INFO",
                "type": "handler",
                "message": "[INFO] Low-signal detected, but OCR already completed earlier. Skipping second OCR.",
                "session_id": session_id
            })

    logger.debug({
        "level": "DEBUG",
        "type": "handler",
        "message": "[DEBUG] PDF extracted text preview (first 500 chars):" + (clean_text[:500] if isinstance(clean_text, str) else str(clean_text)[:500]),
        "session_id": session_id
    })

    table_hints = list(
        set(LOCATION_KEYWORDS) | set(CANDIDATE_KEYWORDS) | set(BALLOT_TYPES) |
        set(PARTY_KEYWORDS) | set(TOTAL_KEYWORDS) | set(MISC_FOOTER_KEYWORDS) | set(CONTEST_KEYWORDS)
    )
    # Use sanitized text from here on
    lines = clean_text.splitlines()
    camelot_tables = attempt_camelot_extraction(pdf_path, session_id=session_id)
    layout_tables = _extract_tables_via_layout(
        pdf_path,
        session_id=session_id,
        ocr_params=metadata.get("ocr_params"),
    )
    statement_headers, statement_rows = _extract_statement_return_blocks(
        pdf_path,
        session_id=session_id,
        ocr_params=metadata.get("ocr_params"),
    )

    statement_headers_raw = list(statement_headers or [])
    statement_rows_raw = [dict(row) for row in statement_rows] if statement_rows else []
    statement_headers_norm, statement_rows_norm, statement_norm_diag = _normalize_statement_candidate_results(
        statement_headers_raw,
        [dict(row) for row in statement_rows_raw],
    )

    statement_headers_use = statement_headers_raw
    statement_rows_use = statement_rows_raw
    if statement_rows_norm:
        statement_headers_use = statement_headers_norm
        statement_rows_use = statement_rows_norm
        diag = dict(statement_norm_diag or {})
        diag["raw_rows"] = len(statement_rows_raw)
        metadata["statement_blocks_normalized"] = diag

    statement_headers_copy = list(statement_headers_use)
    statement_rows_copy = [dict(row) for row in statement_rows_use]
    statement_headers = list(statement_headers_use)
    statement_rows = [dict(row) for row in statement_rows_use]
    if layout_tables:
        metadata["layout_tables_available"] = [
            {
                "headers": tbl.get("headers", [])[:8],
                "rows": len(tbl.get("rows", [])),
                "page": tbl.get("page"),
            }
            for tbl in layout_tables[:5]
        ]
    if statement_rows_copy:
        metadata["statement_blocks_available"] = {
            "rows": len(statement_rows_copy),
            "headers": statement_headers_copy[:10]
        }
        metadata["statement_blocks_diagnostic"] = {
            "normalized_rows": len(statement_rows_copy),
            "raw_rows": len(statement_rows_raw)
        }

    headers, header_candidate = infer_headers_and_methods(lines, table_hints)

    # Detect potential contests from text as hints
    detected_titles = _detect_contest_titles_from_text(lines)
    # Deduplicate aggressively – single-race PDFs often repeat the same heading
    detected_titles = _dedupe_contest_titles(detected_titles)
    if not detected_titles:
        detected_titles = [os.path.basename(pdf_path).replace(".pdf", "")]

    # Derive light context from filename for better selection (before prompting)
    fname = os.path.basename(pdf_path).lower()
    state = "Unknown"
    county = "Unknown"
    state_normalized = None
    county_normalized = None
    year = None
    for part in fname.replace(".pdf", "").split("_"):
        if "county" in part:
            county = part.replace("county", "").strip().title() + " County"
        if len(part) == 2 and part.isalpha():
            state = part.upper()
    m = re.search(r"(19|20)\d{2}", fname)
    if m:
        try:
            year = int(m.group(0))
        except Exception:
            year = None

    # Single contest fast-path or unified selector pass (no duplicate prompts)
    selector_context = {
        "selector_data": {
            "contests": [{"title": t} for t in detected_titles],
            "noisy_patterns": [s.lower() for s in (CONTEST_TITLE_SKIP_PHRASES or set())]
        },
        "input_file": os.path.basename(pdf_path)
    }
    auto_pick = select_contest_auto_first(
        coordinator=coordinator,
        context=selector_context,
        session_id=session_id,
        allow_multiple=False,
        force_interactive=False
    )
    if not auto_pick:
        logger.warning({
            "level": "WARNING",
            "type": "handler",
            "message": "[WARN] No contest selected. Using filename fallback.",
            "session_id": session_id
        })
        selected_contest_title = os.path.basename(pdf_path).replace(".pdf", "")
    else:
        selected_contest_title = safe_get(auto_pick[0], "title") or detected_titles[0]
    contest_slug = safe_slug(selected_contest_title, 80)
    
    # Update metadata using derived context
    metadata.update({
        "source_file": os.path.basename(pdf_path),
        "state": state,
        "county": county,
        "handler": "pdf_handler",
        "contest": selected_contest_title,
        "state_normalized": state_normalized,
        "county_normalized": county_normalized,
    })

    try:
        det_county, det_state, _handler_path, det_log = dynamic_state_county_detection(
            {"state": state, "county": county, "contest": selected_contest_title},
            clean_text,
            debug=False
        )
        if det_state:
            state_normalized = det_state
            formatted_state = format_state_label(det_state)
            if formatted_state:
                state = formatted_state
        if det_county:
            county_normalized = det_county
            formatted_county = format_county_label(det_county, det_state or state_normalized or state)
            if formatted_county:
                county = formatted_county
        if det_log:
            metadata["location_detection_log"] = det_log
    except Exception:
        pass

    contest_column = None
    if headers:
        # Auto-detect a contest-like column; avoid interactive re-prompt
        contest_header_keywords = CONTEST_HEADER_KEYWORDS
        candidates = [h for h in headers if any(kw in h.lower() for kw in contest_header_keywords)]

        if len(candidates) == 1:
            contest_column = candidates[0]
            logger.info({
                "level": "INFO",
                "type": "handler",
                "message": f"[INFO] Auto-selected contest column: {contest_column}",
                "session_id": session_id
            })
        elif len(candidates) > 1:
            # Rank by preference order from constants
            pref = CONTEST_HEADER_PREFERENCE
            def rank(h):
                low = h.lower()
                for i, kw in enumerate(pref):
                    if kw in low:
                        return i
                return len(pref)
            candidates.sort(key=rank)
            if rank(candidates[0]) < rank(candidates[1]):
                contest_column = candidates[0]
                logger.info({
                    "level": "INFO",
                    "type": "handler",
                    "message": f"[INFO] Auto-selected contest column (ranked): {contest_column}",
                    "session_id": session_id
                })
            else:
                logger.info({
                    "level": "INFO",
                    "type": "handler",
                    "message": "[INFO] Multiple possible contest columns detected; skipping auto-selection to avoid extra prompts.",
                    "session_id": session_id
                })

    data = []

    if headers:
        # Locate the header line index
        header_line_idx = -1
        if header_candidate:
            try:
                header_line_idx = lines.index(header_candidate)
            except ValueError:
                header_line_idx = -1
        if header_line_idx < 0:
            # heuristic search: first line whose split matches header cells
            for idx, line in enumerate(lines):
                if _split_ws_blocks(line) == headers:
                    header_line_idx = idx
                    break
            if header_line_idx < 0:
                header_line_idx = 0

        # First pass: strict width-split like before
        for line in lines[header_line_idx + 1:]:
            if not line.strip():
                continue
            row = _split_ws_blocks(line)
            if len(row) == len(headers):
                data.append(dict(zip(headers, row)))

        # Fallback pass: whitespace-table extractor if strict pass yielded no rows
        if not data:
            data = _extract_table_by_whitespace(lines, header_line_idx, headers)
        # Try semantic candidate totals extraction if still empty
        if not data:
            cand_headers, cand_rows = extract_candidate_totals_from_lines(lines, selected_contest_title)
            if cand_headers and cand_rows:
                export_context = _prepare_output_context(
                    None,
                    {
                        "handler": "pdf_handler",
                        "input_file": os.path.basename(pdf_path),
                        "session_id": session_id,
                        "semantic_extraction": True,
                    }
                )
                result = finalize_election_output(
                    headers=cand_headers,
                    data=cand_rows,
                    coordinator=coordinator,
                    contest=selected_contest_title,
                    state=state,
                    county=county,
                    context=export_context,
                    enable_user_feedback=False,
                    session_id=session_id
                )
                metadata.update({
                    "output_file": os.path.basename(result.get("csv_path", "")),
                    "headers": cand_headers,
                    "row_count": len(cand_rows),
                    "csv_path": result.get("csv_path"),
                    "metadata_path": result.get("metadata_path"),
                })
                logger.info({
                    "level": "INFO",
                    "type": "output",
                    "message": f"[OUTPUT] Wrote semantic candidate totals: {result.get('csv_path')}",
                    "session_id": session_id
                })
                return cand_headers, cand_rows, selected_contest_title, metadata
        if camelot_tables:
            top_c = camelot_tables[0]
            if metadata.get("ocr_used"):
                try:
                    hybrid_fill_camelot(top_c, lines)
                    metadata["camelot_hybrid_fill"] = True
                except Exception:
                    pass
            use_camelot = False
            if not data:
                use_camelot = True
            elif _table_looks_bad(headers, data) and top_c["score"] >= 0.8:
                use_camelot = True
            elif top_c["score"] >= 0.9 and len(top_c["rows"]) >= max(5, int(len(data) * 1.2)):
                use_camelot = True

            if use_camelot:
                headers = top_c["headers"]
                data = top_c["rows"]
                metadata["camelot_used"] = True
                metadata["camelot_flavor"] = top_c["flavor"]
                metadata["camelot_score"] = float(top_c["score"])
                metadata["camelot_rows"] = len(top_c["rows"])
            else:
                metadata["camelot_available"] = True
                metadata["camelot_top_score"] = float(top_c["score"])
                metadata["camelot_alt_count"] = len(camelot_tables)
                metadata["camelot_tables_summary"] = [
                    {
                        "score": round(t["score"], 3),
                        "rows": len(t["rows"]),
                        "cols": len(t["headers"]),
                        "flavor": t["flavor"]
                    } for t in camelot_tables[:5]
                ]
        # If we got some rows, but the table looks low-quality, switch to semantic extraction
        if data and _table_looks_bad(headers, data):
            logger.info({
                "level": "INFO",
                "type": "handler",
                "message": "[INFO] Detected low-quality header/rows; attempting semantic candidate extraction instead.",
                "session_id": session_id
            })
            cand_headers, cand_rows = extract_candidate_totals_from_lines(lines, selected_contest_title)
            if cand_headers and cand_rows and len(cand_rows) >= len(data):
                export_context = _prepare_output_context(
                    None,
                    {
                        "handler": "pdf_handler",
                        "input_file": os.path.basename(pdf_path),
                        "session_id": session_id,
                        "semantic_extraction": True,
                        "replaced_noisy_table": True,
                    }
                )
                result = finalize_election_output(
                    headers=cand_headers,
                    data=cand_rows,
                    coordinator=coordinator,
                    contest=selected_contest_title,
                    state=state,
                    county=county,
                    context=export_context,
                    enable_user_feedback=False,
                    session_id=session_id
                )
                metadata.update({
                    "output_file": os.path.basename(result.get("csv_path", "")),
                    "headers": cand_headers,
                    "row_count": len(cand_rows),
                    "csv_path": result.get("csv_path"),
                    "metadata_path": result.get("metadata_path"),
                })
                logger.info({
                    "level": "INFO",
                    "type": "output",
                    "message": f"[OUTPUT] Replaced noisy table with semantic candidate totals: {result.get('csv_path')}",
                    "session_id": session_id
                })
                return cand_headers, cand_rows, selected_contest_title, metadata
            else:
                logger.info({
                    "level": "INFO",
                    "type": "handler",
                    "message": "[INFO] Semantic extraction did not improve over noisy table; keeping extracted rows.",
                    "session_id": session_id
                })

        if layout_tables:
            best_layout = layout_tables[0]
            layout_headers = best_layout.get("headers") or []
            layout_rows = best_layout.get("rows") or []
            use_layout = False
            if not data and layout_headers and layout_rows:
                use_layout = True
            elif layout_headers and layout_rows:
                if _table_looks_bad(headers, data):
                    use_layout = True
                elif len(layout_rows) >= max(len(data) + 5, int(len(data) * 1.5)):
                    use_layout = True
            if use_layout:
                headers = layout_headers
                data = layout_rows
                contest_column = None
                metadata["layout_table_used"] = True
                metadata["layout_table_rows"] = len(layout_rows)
                metadata["layout_table_page"] = best_layout.get("page")
            elif layout_rows:
                metadata["layout_table_candidate_rows"] = len(layout_rows)

        if statement_rows_copy:
            use_statement = False
            pre_statement_rows = len(data) if isinstance(data, list) else 0
            if not data:
                use_statement = True
            elif len(statement_rows_copy) >= max(len(data), 5):
                use_statement = True
            elif len(data) <= 3 and len(statement_rows_copy) >= len(data) * 2:
                use_statement = True
            if use_statement:
                headers = list(statement_headers_copy)
                data = [dict(row) for row in statement_rows_copy]
                contest_column = None
                metadata["statement_blocks_used"] = True
                metadata["statement_blocks_rows"] = len(statement_rows_copy)
            metadata["statement_blocks_decision"] = {
                "pre_rows": pre_statement_rows,
                "use_statement": use_statement,
                "statement_rows": len(statement_rows_copy),
            }
            logger.debug({
                "level": "DEBUG",
                "type": "handler",
                "message": "[DEBUG] Statement-return heuristic",
                "session_id": session_id,
                "pre_rows": pre_statement_rows,
                "statement_rows": len(statement_rows_copy),
                "use_statement": use_statement,
            })

        # If we have a contest column, filter to the selected contest
        contest = selected_contest_title
        if contest_column:
            def _norm_title(s: str) -> str:
                s = (s or "").lower().strip()
                s = re.sub(r'[\s\-_/]+', ' ', s)
                s = re.sub(r'[^a-z0-9 ]+', '', s)
                return re.sub(r'\s+', ' ', s).strip()

            def _tokens(s: str) -> set[str]:
                return set(re.findall(r'[a-z0-9]+', (s or "").lower()))

            norm_selected = _norm_title(contest)
            present_values = sorted({(r.get(contest_column, "") or "").strip() for r in data if r.get(contest_column)})
            norm_map = {v: _norm_title(v) for v in present_values}

            # 1) exact normalized match
            exact = [v for v, nv in norm_map.items() if nv == norm_selected]
            chosen_value = exact[0] if exact else None

            # 2) token-overlap fallback if no exact
            if not chosen_value:
                sel_tok = _tokens(contest)
                scored = []
                for v in present_values:
                    vt = _tokens(v)
                    inter = len(sel_tok & vt)
                    union = len(sel_tok | vt) or 1
                    jacc = inter / union
                    # small boost for prefix/substring matches
                    if norm_selected and norm_map[v].startswith(norm_selected):
                        jacc += 0.15
                    if norm_selected and norm_selected in norm_map[v]:
                        jacc += 0.10
                    scored.append((jacc, v))
                scored.sort(reverse=True)
                if scored and scored[0][0] >= 0.45:
                    chosen_value = scored[0][1]

            if chosen_value:
                data = [r for r in data if _norm_title(r.get(contest_column, "")) == _norm_title(chosen_value)]
                logger.info({
                    "level": "INFO",
                    "type": "handler",
                    "message": f"[INFO] Filtered rows to contest '{chosen_value}' via column '{contest_column}'.",
                    "session_id": session_id
                })
            else:
                logger.warning({
                    "level": "WARNING",
                    "type": "handler",
                    "message": f"[WARN] Selected contest '{contest}' not found in column '{contest_column}'. Skipping row filter.",
                    "session_id": session_id,
                    "present": present_values[:25]
                })

        if data:
            metadata["pre_finalize_row_count"] = len(data)
            headers_final, data_final, _ = _finalize_structured_table_output(
                pdf_path,
                headers,
                data,
                selected_contest_title,
                state,
                county,
                year,
                contest_slug,
                metadata,
                coordinator,
                session_id=session_id,
            )
            if statement_rows_copy and statement_headers_copy:
                promote_statement = False
                if metadata.get("statement_blocks_used"):
                    promote_statement = True
                elif len(data_final) <= 3 and len(statement_rows_copy) >= max(len(data_final) * 2, 5):
                    promote_statement = True
                if promote_statement:
                    metadata["statement_blocks_used"] = True
                    metadata["statement_blocks_rows"] = len(statement_rows_copy)
                    metadata["statement_blocks_promoted"] = {
                        "rows_before": len(data_final),
                        "rows_statement": len(statement_rows_copy),
                    }
                    headers_final, data_final, _ = _finalize_structured_table_output(
                        pdf_path,
                        list(statement_headers_copy),
                        [dict(row) for row in statement_rows_copy],
                        selected_contest_title,
                        state,
                        county,
                        year,
                        contest_slug,
                        metadata,
                        coordinator,
                        session_id=session_id,
                    )
            return headers_final, data_final, selected_contest_title, metadata

        else:
            if statement_rows_copy and statement_headers_copy:
                metadata["statement_blocks_used"] = True
                metadata["statement_blocks_rows"] = len(statement_rows_copy)
                headers_final, data_final, _ = _finalize_structured_table_output(
                    pdf_path,
                    list(statement_headers_copy),
                    [dict(row) for row in statement_rows_copy],
                    selected_contest_title,
                    state,
                    county,
                    year,
                    contest_slug,
                    metadata,
                    coordinator,
                    session_id=session_id,
                )
                return headers_final, data_final, selected_contest_title, metadata

            unmatched_count = len(lines[header_line_idx + 1:])
            logger.warning({
                "level": "WARNING",
                "type": "output",
                "message": f"[WARN] No structured rows matched the inferred column count of {len(headers)}. Total lines scanned: {unmatched_count}",
                "session_id": session_id
            })
            fallback_rows = [{"raw_line": line} for line in lines[header_line_idx + 1:]]
            fallback_context = {
                "contest": selected_contest_title,
                "state": state,
                "county": county,
                "session_id": session_id,
                "handler": "pdf_handler",
            }
            export_context = _prepare_output_context(
                fallback_context,
                {
                    "handler": "pdf_handler",
                    "input_file": os.path.basename(pdf_path),
                    "session_id": session_id,
                    "fallback": True,
                }
            )
            result = finalize_election_output(
                headers=["raw_line"],
                data=fallback_rows,
                coordinator=coordinator,
                contest=selected_contest_title,
                state=state,
                county=county,
                context=export_context,
                enable_user_feedback=False,
                session_id=session_id
            )
            metadata.update({
                "output_file": os.path.basename(result.get("csv_path", "")),
                "headers": ["raw_line"],
                "row_count": len(fallback_rows),
                "csv_path": result.get("csv_path"),
                "metadata_path": result.get("metadata_path"),
            })
            logger.warning({
                "level": "WARNING",
                "type": "output",
                "message": f"[OUTPUT] Wrote fallback rows to: {result.get('csv_path')}",
                "session_id": session_id
            })
            return ["raw_line"], fallback_rows, selected_contest_title, metadata

    if layout_tables and not metadata.get("layout_table_used"):
        best_layout = layout_tables[0]
        layout_headers = best_layout.get("headers") or []
        layout_rows = best_layout.get("rows") or []
        if layout_headers and layout_rows:
            prefer_statement = False
            if statement_rows_copy and statement_headers_copy:
                if len(layout_rows) <= 3 and len(statement_rows_copy) >= max(len(layout_rows) * 2, 5):
                    prefer_statement = True
                elif len(statement_rows_copy) >= max(len(layout_rows) + 5, int(len(layout_rows) * 1.5)):
                    prefer_statement = True
            metadata["layout_table_used"] = True
            metadata["layout_table_rows"] = len(layout_rows)
            metadata["layout_table_page"] = best_layout.get("page")
            headers_final, data_final, _ = _finalize_structured_table_output(
                pdf_path,
                layout_headers,
                layout_rows,
                selected_contest_title,
                state,
                county,
                year,
                contest_slug,
                metadata,
                coordinator,
                session_id=session_id,
            )
            if (
                statement_rows_copy
                and statement_headers_copy
                and (
                    prefer_statement
                    or (
                        len(data_final) <= 3
                        and len(statement_rows_copy) >= max(len(data_final) * 2, 5)
                    )
                )
            ):
                metadata["layout_table_available_rows"] = len(layout_rows)
                metadata["layout_table_available_page"] = best_layout.get("page")
                metadata["layout_table_used"] = False
                metadata["statement_blocks_used"] = True
                metadata["statement_blocks_rows"] = len(statement_rows_copy)
                headers_final, data_final, _ = _finalize_structured_table_output(
                    pdf_path,
                    list(statement_headers_copy),
                    [dict(row) for row in statement_rows_copy],
                    selected_contest_title,
                    state,
                    county,
                    year,
                    contest_slug,
                    metadata,
                    coordinator,
                    session_id=session_id,
                )
            return headers_final, data_final, selected_contest_title, metadata

    if statement_rows and statement_headers:
        # Ensure contest metadata reflects statement extraction path
        metadata["statement_blocks_available"] = {
            "rows": len(statement_rows),
            "headers": statement_headers[:10]
        }
        metadata["statement_blocks_used"] = True
        metadata["statement_blocks_rows"] = len(statement_rows)
        headers_final, data_final, _ = _finalize_structured_table_output(
            pdf_path,
            list(statement_headers),
            [dict(row) for row in statement_rows],
            selected_contest_title,
            state,
            county,
            year,
            contest_slug,
            metadata,
            coordinator,
            session_id=session_id,
        )
        return headers_final, data_final, selected_contest_title, metadata

    # No headers at all: still try semantic candidate totals from entire text
    cand_headers, cand_rows = extract_candidate_totals_from_lines(lines, selected_contest_title)
    if cand_headers and cand_rows:
        export_context = _prepare_output_context(
            None,
            {
                "handler": "pdf_handler",
                "input_file": os.path.basename(pdf_path),
                "session_id": session_id,
                "semantic_extraction": True,
            }
        )
        result = finalize_election_output(
            headers=cand_headers,
            data=cand_rows,
            coordinator=coordinator,
            contest=selected_contest_title,
            state=state,
            county=county,
            context=export_context,
            enable_user_feedback=False,
            session_id=session_id
        )
        metadata.update({
            "output_file": os.path.basename(result.get("csv_path", "")),
            "headers": cand_headers,
            "row_count": len(cand_rows),
            "csv_path": result.get("csv_path"),
            "metadata_path": result.get("metadata_path"),
        })
        logger.info({
            "level": "INFO",
            "type": "output",
            "message": f"[OUTPUT] Wrote semantic candidate totals: {result.get('csv_path')}",
            "session_id": session_id
        })
        return cand_headers, cand_rows, selected_contest_title, metadata

    # Plain text fallback
    export_context = _prepare_output_context(
        None,
        {
            "handler": "pdf_handler",
            "input_file": os.path.basename(pdf_path),
            "session_id": session_id,
            "text_sanitized": True,
            "raw_text_len": len(all_text or ""),
            "clean_text_len": len(clean_text or ""),
        }
    )
    result = finalize_election_output(
        headers=["text"],
        data=[{"text": clean_text}],
        coordinator=coordinator,
        contest=selected_contest_title,
        state=state,
        county=county,
        context=export_context,
        enable_user_feedback=False,
        session_id=session_id
    )
    metadata.update({
        "output_file": os.path.basename(result.get("csv_path", "")),
        "headers": ["text"],
        "row_count": 1,
        "text_sanitized": True,
        "raw_text_len": len(all_text or ""),
        "clean_text_len": len(clean_text or ""),
        "csv_path": result.get("csv_path"),
        "metadata_path": result.get("metadata_path")
    })
    logger.warning({
        "level": "WARNING",
        "type": "output",
        "message": f"[OUTPUT] Wrote plain text to: {result.get('csv_path')}",
        "session_id": session_id
    })
    return ["text"], [{"text": clean_text}], selected_contest_title, metadata

def parse(page=None, coordinator=None, html_context=None, manual_file=None, session_id=None, **kwargs):
    """
    Universal pipeline entry: Accepts a PDF file path (manual_file) from the format router.
    Returns: headers, data, contest, metadata
    """
    html_context = html_context or {}
    # Parity guard: allow provided_tables + skip_pivot to bypass file requirement
    provided_tables = html_context.get("provided_tables")
    if isinstance(provided_tables, list) and provided_tables:
        ctx = dict(html_context)
        ctx.update({
            "session_id": session_id,
            "coordinator": coordinator,
        })
        merged_headers, merged_rows = robust_table_extraction(page=None, extraction_context=ctx)

        contest = html_context.get("contest") or "Provided Tables"
        state = html_context.get("state") or "Unknown"
        county = html_context.get("county") or "Unknown"
        year = html_context.get("year")
        domain = html_context.get("source_slug") or safe_slug(contest)

        headers_final, data_final, _entity_info = build_table_noninteractive(
            domain=domain,
            headers=merged_headers,
            data=merged_rows,
            coordinator=coordinator,
            context={
                **ctx,
                "contest": contest,
                "state": state,
                "county": county,
                "year": year,
                "handler": "pdf_handler",
            },
            pivot_to_wide=not bool(html_context.get("skip_pivot")),
            debug=False,
        )

        export_context = _prepare_output_context(
            ctx,
            {
                "handler": "pdf_handler",
                "session_id": session_id,
                "race": contest,
                "provided_tables": True,
                "skip_pivot": bool(html_context.get("skip_pivot")),
                "optional_deps": {
                    "camelot_available": bool(_CAMELOT_AVAILABLE),
                    "pytesseract_available": bool(pytesseract),
                    "pdf2image_available": bool(pdf2image),
                },
            }
        )

        result = finalize_election_output(
            headers=headers_final,
            data=data_final,
            coordinator=coordinator,
            contest=contest,
            state=state,
            county=county,
            context=export_context,
            enable_user_feedback=False,
            session_id=session_id,
        )

        metadata = {
            "race": contest,
            "input_file": html_context.get("input_file") or "<provided>",
            "output_file": os.path.basename(result.get("csv_path", "")),
            "headers": headers_final,
            "row_count": len(data_final),
            "handler": "pdf_handler",
            "state": state,
            "county": county,
            "year": year,
            "csv_path": result.get("csv_path"),
            "metadata_path": result.get("metadata_path"),
            "optional_deps": {
                "camelot_available": bool(_CAMELOT_AVAILABLE),
                "pytesseract_available": bool(pytesseract),
                "pdf2image_available": bool(pdf2image),
            },
        }
        return headers_final, data_final, contest, metadata
    if html_context.get("skip_format") or html_context.get("manual_skip"):
        logger.info({
            "level": "INFO",
            "type": "handler",
            "message": "[SKIP] PDF parsing intentionally skipped via context flag.",
            "session_id": session_id
        })
        return None, None, None, {"skipped": True}

    if not manual_file or not os.path.isfile(manual_file):
        logger.error({
            "level": "ERROR",
            "type": "handler",
            "message": "[ERROR] No PDF file provided to parse().",
            "session_id": session_id
        })
        return None, None, None, {"skipped": True}

    result = parse_pdf_election_results(manual_file, session_id=session_id, coordinator=coordinator)

    # Defensive: always return a 4-tuple, never a bool
    if not (isinstance(result, tuple) and len(result) == 4):
        logger.error({
            "level": "ERROR",
            "type": "handler",
            "message": "[ERROR] Invalid result from parse_pdf_election_results (expected 4-tuple).",
            "session_id": session_id,
            "got_type": type(result).__name__
        })
        return None, None, None, {"error": "Invalid parse result"}
    return result
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import os

# webapp/parser/utils/output_utils.py
# ---------------------------------------------------------------
# Output utilities for Smart Elections Parser Webapp
# ---------------------------------------------------------------
import re
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

import orjson
import pandas as pd

from ..config import BASE_DIR, LOG_DIR, OUTPUT_CACHE, OUTPUT_DIR
from .logger_singleton import logger
from .pivot import transform_wide_to_smart_standard
from .rawjson_utils import (
    extract_rawjson_enrichment_from_rows,
)
from .rawjson_utils import (
    offload_rawjson_to_ndjson as _shared_offload_rawjson_to_ndjson,
)
from .shared_logic import (
    is_path_safe,
    safe_filename,
    safe_get,
    safe_get_first,
    safe_items,
    safe_join_path,
    safe_lower,
    safe_resolve_path,
)

PERCENT_COL_REGEX = re.compile(r"(% Vote|Cumulative %|Percent Reported| - %)$", re.I)

def coerce_percent_strings(row: dict):
    for k, v in row.items():
        if isinstance(v, str) and PERCENT_COL_REGEX.search(k):
            sv = v.replace("%", "").strip()
            if sv.replace(".", "", 1).isdigit():
                row[k] = f"{float(sv):.2f}%"
    return row

def get_project_root() -> str:
    # Returns the parent directory of webapp (the project root)
    return os.path.dirname(BASE_DIR)

def get_output_root() -> str:
    # Output folder at the project root (validated)
    project_root = get_project_root()
    output_path = os.path.join(project_root, "output")
    
    # Validate and create if needed
    try:
        return str(safe_resolve_path(output_path, project_root))
    except ValueError:
        # Fallback to OUTPUT_DIR from config
        return OUTPUT_DIR

def safe_join(base: str, *paths: str) -> str:
    """
    Safely join paths and ensure the result is inside base.
    Prevents path traversal and path-injection.
    
    Args:
        base: Base directory path
        *paths: Path components to join
        
    Returns:
        Safe joined path
        
    Raises:
        ValueError: If resulting path escapes base directory
    """
    try:
        # Use the new safe_join_path utility
        result = safe_join_path(base, *paths)
        return str(result)
    except Exception as e:
        logger.error(f"[SECURITY] Path join validation failed: {e}")
        raise ValueError(f"Unsafe path detected during join: {e}")

def get_output_path(metadata, subfolder="parsed", coordinator=None, feedback_context=None) -> str:
    """
    Build output path using organized context metadata with strict path validation.
    If any key info is missing, use feedback loop (ML/NER/user prompt) to resolve.
    Safeguards all string operations and path parts.
    
    Raises:
        ValueError: If path validation fails or path traversal is detected
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = ContextCoordinator()
    # Use coordinator to try to fill missing info if available
    state = safe_get(metadata, "state", "") or (
        safe_get_first(getattr(coordinator, "get_states", lambda: [])(), "state", None, logger, default="") if coordinator and hasattr(coordinator, "get_states") and coordinator.get_states() else ""
    )
    county = safe_get(metadata, "county", "") or (
        safe_get_first(getattr(coordinator, "get_precincts", lambda: [])(), "county", None, logger, default="") if coordinator and hasattr(coordinator, "get_precincts") and coordinator.get_precincts() else ""
    )
    year = safe_get(metadata, "year", "")
    contests = safe_get(metadata, "contests", "")
    election_types = safe_get(metadata, "election_types", "")

    # Feedback loop for missing/unknown info (existing logic continues...)
    max_loops = 3
    for _ in range(max_loops):
        if not year or not str(year).isdigit() or len(str(year)) != 4:
            if coordinator and hasattr(coordinator, "get_years"):
                years = coordinator.get_years()
                if years and len(years) > 0:
                    year = safe_get_first(years, "year", None, logger)
            if not year and feedback_context:
                year = safe_get(feedback_context, "year", "")
        if not contests or safe_lower(contests) == "unknown":
            if coordinator and hasattr(coordinator, "get_contests"):
                contests_list = coordinator.get_contests()
                if contests_list and isinstance(contests_list, list) and len(contests_list) > 0:
                    first_contest = safe_get_first(contests_list, "contests", None, logger)
                    if isinstance(first_contest, dict):
                        contests = safe_get(first_contest, "title", "")
                    else:
                        contests = str(first_contest)
            if not contests and feedback_context:
                contests = safe_get(feedback_context, "contests", "")
        if year and contests:
            break

    if not year or not str(year).isdigit() or len(str(year)) != 4:
        logger.warning("[yellow][OUTPUT] Year could not be verified. Using 'Unknown'.[/yellow]")
        year = "Unknown"
    if not contests:
        logger.warning("[yellow][OUTPUT] contests could not be verified. Using 'unknown_contests'.[/yellow]")
        contests = "unknown_contests"

    # Centralized filtering/slugging with STRICT sanitization
    s_slug, c_slug, ct_slug = build_filename_triplet(state, county, contests)
    
    # SECURITY: Sanitize all path components with strict mode
    safe_components = []
    if ct_slug:
        safe_components.append(safe_filename(safe_lower(ct_slug), strict_mode=True))
    if s_slug:
        safe_components.append(safe_filename(safe_lower(s_slug), strict_mode=True))
    if c_slug:
        safe_components.append(safe_filename(safe_lower(c_slug), strict_mode=True))
    if year and str(year).isdigit() and len(str(year)) == 4:
        safe_components.append(safe_filename(str(year), strict_mode=True))
    else:
        safe_components.append("Unknown")
    if election_types:
        safe_components.append(safe_filename(safe_lower(safe_filename(election_types)), strict_mode=True))
    if contests:
        # Extra sanitization for contests
        safe_contests = safe_filename(str(contests), strict_mode=True)
        safe_components.append(safe_contests)
    else:
        safe_components.append("unknown_contests")
    if subfolder:
        safe_components.append(safe_filename(str(subfolder), strict_mode=True))

    # Always use output folder at project root with validation
    output_root = get_output_root()
    
    try:
        # Use safe_join_path to build the final path
        path = safe_join_path(output_root, *safe_components)
        
        # Validate it's within output_root
        if not is_path_safe(path, [output_root]):
            raise ValueError("Output path escapes output root directory")
        
        # Create directory
        path.mkdir(parents=True, exist_ok=True)
        
        return str(path)
        
    except Exception as e:
        logger.error(f"[SECURITY] Output path validation failed: {e}")
        # Safe fallback: return just the output root
        return output_root

def format_timestamp(fmt="%Y%m%d_%H%M%S") -> str:
    return datetime.now().strftime(fmt)

def update_output_cache(metadata, output_path, cache_file=OUTPUT_CACHE) -> None:
    """
    Append output metadata to a cache file for fast lookup and deduplication.
    Robustly handles orjson serialization and file writing.
    """
    cache_entry = {
        "timestamp": format_timestamp(),
        "output_path": output_path,
        "metadata": metadata,
    }
    try:
        serialized = orjson.dumps(cache_entry)
    except Exception as e:
        logger.error(f"[OUTPUT_UTILS] Failed to serialize cache entry: {e}")
        serialized = str(cache_entry).encode("utf-8")
    try:
        with open(cache_file, "ab") as f:
            f.write(serialized + b"\n")
    except Exception as e:
        logger.error(f"[OUTPUT_UTILS] Failed to write to cache file: {e}")

def check_existing_output(metadata, cache_file=OUTPUT_CACHE) -> Optional[dict]:
    """
    Check if output for this context already exists in the cache.
    Handles both JSONL (one JSON object per line) and JSON array formats.
    """
    if not os.path.exists(OUTPUT_CACHE):
        return None
    with open(cache_file, "rb") as f:
        content = f.read().strip()
        if not content:
            return None
        entries = []
        # Try JSON array first
        try:
            if content.startswith(b"["):
                arr = orjson.loads(content)
                if isinstance(arr, list):
                    entries = arr
            else:
                raise ValueError("Not a JSON array")
        except Exception:
            # Fallback: treat as JSONL
            entries = []
            for line in content.splitlines():
                if not line.strip():
                    continue
                try:
                    entries.append(orjson.loads(line))
                except Exception:
                    logger.debug(f"[DEBUG] Failed to parse line as JSON: {line!r}")
                    continue
        for entry in entries:
            meta = safe_get(entry, "metadata", {})
            if (
                safe_get(meta, "state", "Unknown") == safe_get(metadata, "state", "Unknown") and
                safe_get(meta, "county", "Unknown") == safe_get(metadata, "county", "Unknown") and
                safe_get(meta, "year", "Unknown") == safe_get(metadata, "year", "Unknown") and
                safe_get(meta, "contests", "Unknown") == safe_get(metadata, "contests", "Unknown")
            ):
                return entry
    return None

def convert_sets_to_lists(obj) -> dict:
    if isinstance(obj, dict):
        return {k: convert_sets_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, list):
        return [convert_sets_to_lists(i) for i in obj]
    else:
        return obj

def deep_merge_dicts(dest, src) -> dict:
    """
    Recursively merge src into dest.
    - If a key exists in both and both values are dicts, merge them recursively.
    - Otherwise, src overwrites dest.
    """
    for k, v in safe_items(src):
        if (
            k in dest
            and isinstance(dest[k], dict)
            and isinstance(v, dict)
        ):
            deep_merge_dicts(dest[k], v)
        else:
            dest[k] = v
    return dest

def _slug(value: Optional[str], max_len: int = 80) -> str:
    if not isinstance(value, str):
        return "na"
    stem = value.strip()
    stem = re.sub(r"[^\w\s-]+", "_", stem, flags=re.UNICODE)
    stem = re.sub(r"[\s_-]+", " ", stem).strip()
    stem = stem.replace(" ", "_")
    stem = re.sub(r"_+", "_", stem)
    return stem[:max_len] or "na"

def build_filename_triplet(state: Optional[str], county: Optional[str], contest: Optional[str]) -> tuple[str, str, str]:
    """
    Return (state_slug, county_slug, contest_slug) with Unknown filtered to empty.
    Slugs are safe for filenames.
    """
    s = _slug((state or "").strip())
    c = _slug((county or "").strip())
    ct = _slug((contest or "").strip(), max_len=120)
    if s.lower() == "unknown":
        s = ""
    if c.lower() == "unknown":
        c = ""
    return s, c, ct

def _ensure_dir(p: str) -> None:
    try:
        os.makedirs(p, exist_ok=True)
    except Exception:
        pass

def _coerce_headers(headers: List[str], rows: List[Dict[str, Any]]) -> List[str]:
    base = [h for h in (headers or []) if isinstance(h, str) and h.strip()]
    seen = set(base)
    # Append any additional keys discovered in data, stable order
    for row in rows or []:
        if isinstance(row, dict):
            for k in row.keys():
                if k not in seen:
                    base.append(k)
                    seen.add(k)
    return base

def apply_results_conditional_formatting(writer, sheet_name: str, df: pd.DataFrame):
    """
    Apply conditional formatting:
      - Green 3‑color scale on 'Percent Reported'
      - Data bars on each candidate 'Total Reported' column
      - Data bars on each candidate '%' column (strip % for evaluation)
    """
    try:
        worksheet = writer.sheets[sheet_name]
    except Exception:
        return
    # Percent Reported scale
    if "Percent Reported" in df.columns:
        col_idx = df.columns.get_loc("Percent Reported")
        # xlsxwriter style:
        worksheet.conditional_format(1, col_idx, len(df), col_idx, {
            "type": "3_color_scale",
            "min_color": "#fbe5e1",
            "mid_color": "#ffd965",
            "max_color": "#63be7b"
        })
    # Candidate total & percent columns
    for c in df.columns:
        if c.endswith("Total Reported"):
            idx = df.columns.get_loc(c)
            worksheet.conditional_format(1, idx, len(df), idx, {
                "type": "data_bar",
                "bar_color": "#4F81BD"
            })
        elif c.endswith(" %"):
            # strip % when writing? (Assumes already string with %; leave bars approximate)
            idx = df.columns.get_loc(c)
            worksheet.conditional_format(1, idx, len(df), idx, {
                "type": "data_bar",
                "bar_color": "#9BBB59"
            })

def export_dataframe_with_format(df, path: str, sheet_name: str = "Results"):
    """
    Write DataFrame to XLSX with election result formatting.
    """
    import pandas as pd
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        apply_results_conditional_formatting(writer, sheet_name, df)

def _compute_structure_hash(headers: List[str] | None, rows: List[Dict[str, Any]] | None) -> str:
    """
    Compute a light structure hash based on headers plus first row keys (stable).
    """
    try:
        parts = list(headers or [])
        if rows and isinstance(rows[0], dict):
            for k in rows[0].keys():
                parts.append(str(k))
        digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8", errors="ignore")).hexdigest()
        return digest[:16]
    except Exception:
        return "raw"

def finalize_election_output(
    *,
    headers: List[str],
    data: List[Dict[str, Any]],
    coordinator=None,
    contest: Optional[str] = None,
    state: Optional[str] = None,
    county: Optional[str] = None,
    context: Optional[dict] = None,
    enable_user_feedback: bool = False,
    session_id: Optional[str] = None
) -> Dict[str, str]:
    """
    Centralized writer for CSV + metadata.
    Returns: {"csv_path": ..., "metadata_path": ...}
    """
    context = context or {}
    if "fill_blanks_with_na" not in context:
        context["fill_blanks_with_na"] = True
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR or os.path.join(os.getcwd(), "outputs")
    _ensure_dir(out_dir)

    # Effective state/county/contest from context if provided; omit Unknown in filename
    state_eff = (safe_get(context, "state", None) or state or "").strip()
    county_eff = (safe_get(context, "county", None) or county or "").strip()
    contest_eff = (safe_get(context, "contest", None) or contest or "").strip()

    # Build filenames (skip Unknown parts)
    s_slug, c_slug, ct_slug = build_filename_triplet(state_eff, county_eff, contest_eff)
    parts_for_name = []
    if s_slug:
        parts_for_name.append(s_slug)
    if c_slug:
        parts_for_name.append(c_slug)
    parts_for_name.append(ct_slug or "contest")
    parts_for_name.append(ts)
    base_name = "__".join(parts_for_name)
    bundle_metadata_context = context.get("bundle_metadata")
    bundle_mode = context.get("bundle_mode")
    if not bundle_mode and isinstance(bundle_metadata_context, dict):
        bundle_mode = bundle_metadata_context.get("bundle_mode")

    output_folder = os.path.join(out_dir, base_name)
    _ensure_dir(output_folder)
    csv_path = os.path.join(output_folder, "results.csv")
    meta_path = os.path.join(output_folder, "results.metadata.json")
    context["output_folder"] = output_folder
    context.setdefault("output_base_name", base_name)
    if bundle_mode == "aggregate":
        context.setdefault("bundle_mode", "aggregate")

    # Establish structure hash early (stable NDJSON filename)
    context["structure_hash"] = context.get("structure_hash") or _compute_structure_hash(headers, data)

    # RawJSON enrichment (build from rows if not already provided)
    if not context.get("rawjson_enrichment"):
        try:
            built = extract_rawjson_enrichment_from_rows(data or [])
            if built:
                context["rawjson_enrichment"] = built["extended"]
                context["rawjson_enrichment_slim"] = built["slim"]
        except Exception:
            pass

    # Optional: offload RawJSON column to NDJSON and keep pointers in CSV
    try:
        if any(isinstance(r, dict) and "RawJSON" in r for r in (data or [])):
            structure_hash = context.get("structure_hash")
            data, ndjson_path = _shared_offload_rawjson_to_ndjson(data, os.path.dirname(csv_path), structure_hash)
            if ndjson_path:
                context["rawjson_offload_path"] = ndjson_path
    except Exception:
        pass

    transformed_headers, transformed_rows, smart_applied = transform_wide_to_smart_standard(headers, data, context)
    if smart_applied:
        headers = transformed_headers
        data = transformed_rows

    # Normalize headers and rows
    headers_final = _coerce_headers(headers or [], data or [])
    fill_with_na = bool(context.get("fill_blanks_with_na", False))
    safe_rows: List[Dict[str, Any]] = []
    for row in (data or []):
        if not isinstance(row, dict):
            value_cell = str(row)
            if fill_with_na and not value_cell.strip():
                value_cell = "NA"
            safe_rows.append({"value": value_cell})
            if "value" not in headers_final:
                headers_final = ["value"] + headers_final
            continue
        safe: Dict[str, Any] = {}
        for h in headers_final:
            val = row.get(h, "")
            if isinstance(val, bool):
                cell = "TRUE" if val else "FALSE"
            elif isinstance(val, (str, int, float)) or val is None:
                if val is None:
                    cell = "NA" if fill_with_na else ""
                elif isinstance(val, str):
                    cell = val if (val.strip() or not fill_with_na) else "NA"
                else:
                    cell = val
            else:
                try:
                    cell = orjson.dumps(val).decode("utf-8", errors="ignore")
                except Exception:
                    cell = str(val)
                if fill_with_na and (cell is None or not str(cell).strip()):
                    cell = "NA"
            safe[h] = cell
        # Coerce percent-like strings per row
        safe = coerce_percent_strings(safe)
        safe_rows.append(safe)

    # Write CSV
    def _write_csv(path: str, fieldnames: list[str], rows: list[dict]) -> bool:
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
            return True
        except Exception as e:
            logger.error({
                "level": "ERROR",
                "type": "output",
                "message": f"[ERROR] Failed to write CSV: {e}",
                "session_id": session_id,
                "path": path
            })
            return False
    if not _write_csv(csv_path, headers_final, safe_rows):
        return {"csv_path": "", "metadata_path": ""}

    # Build a simple row->byte-offset index for the CSV to support jump-to-row.
    def _build_csv_index(csv_path: str, max_rows: int = 200000) -> Optional[str]:
        try:
            idx_path = csv_path + '.idx'
            offsets = []
            with open(csv_path, 'rb') as fh:
                # read header
                _ = fh.readline()
                pos = fh.tell()
                count = 0
                while True:
                    line = fh.readline()
                    if not line:
                        break
                    offsets.append(pos)
                    count += 1
                    if count >= max_rows:
                        break
                    pos = fh.tell()
            with open(idx_path, 'wb') as fidx:
                for off in offsets:
                    fidx.write(f"{off}\n".encode('ascii'))
            return idx_path
        except Exception:
            return None

    try:
        idxp = _build_csv_index(csv_path)
        if idxp:
            context['csv_index_path'] = idxp
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Enrichment: build summary + hierarchical header export (if present)
    # ------------------------------------------------------------------
    try:
        enr = context.get("rawjson_enrichment")
        if enr:
            # Contest-level summary (derived consistently)
            group_totals = enr.get("group_totals") or {}
            total_votes_all = 0
            slim_candidates = []
            for c in (enr.get("candidates") or []):
                tv = c.get("total_votes_reported") or 0
                if isinstance(tv, (int, float)):
                    total_votes_all += tv
                pct = (tv / total_votes_all * 100.0) if total_votes_all else 0.0
                slim_candidates.append({
                    "label": c.get("label"),
                    "party": c.get("party"),
                    "total_votes": tv,
                    "pct_total": round(pct, 3) if total_votes_all else 0.0,
                    "groups": c.get("group_breakdown", {})
                })
            # Group percent distribution (fallback if not present)
            group_pct = context.get("summary", {}).get("group_percent_distribution") or {}
            if not group_pct and group_totals:
                grand_groups = sum(v for v in group_totals.values() if isinstance(v, (int, float)))
                if grand_groups:
                    group_pct = {g: round(v / grand_groups * 100.0, 3) for g, v in group_totals.items() if isinstance(v, (int, float))}
            context["summary"] = {
                "contest_id": enr.get("contest_id"),
                "contest_name": enr.get("contest_name"),
                "contest_type": enr.get("contest_type"),
                "vote_for": enr.get("vote_for"),
                "ballot_order": enr.get("ballot_order"),
                "precincts_participating": enr.get("precincts_participating"),
                "precincts_reporting": enr.get("precincts_reporting"),
                "contest_reporting_percent": enr.get("contest_reporting_percent"),
                "candidate_count": enr.get("candidate_count"),
                "group_totals": group_totals,
                "group_percent_distribution": group_pct,
                "total_candidate_votes": total_votes_all,
                "candidates": slim_candidates
            }
            # Ensure we keep a slim enrichment too
            context.setdefault("rawjson_enrichment_slim", {
                "contest_reporting_percent": enr.get("contest_reporting_percent"),
                "candidate_count": enr.get("candidate_count"),
                "groups_present": list(group_totals.keys())
            })
        # Hierarchical headers -> export (two rows) if present
        if "hierarchical_headers" in context and isinstance(context["hierarchical_headers"], dict):
            hh = context["hierarchical_headers"].get("rows")
            if hh and isinstance(hh, list) and all(isinstance(r, list) for r in hh):
                context["hierarchical_header_rows"] = hh
    except Exception as e:
        logger.warning(f"[OUTPUT_UTILS] Enrichment build failed: {e}")

    # structure_hash already set earlier and used for NDJSON offload

    # Build metadata
    meta = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,
        "principal": context.get("principal"),
        "principal_source": context.get("principal_source"),
        "handler": context.get("handler"),
        "input_file": context.get("input_file"),
        "contest": contest_eff,
        "state": state_eff,
        "county": county_eff,
        "row_count": len(safe_rows),
        "headers": headers_final,
        "csv_path": csv_path,
        "output_dir": os.path.dirname(csv_path),
        "output_folder": output_folder,
        "output_base_name": base_name,
        "context": context,
        "user_feedback_enabled": bool(enable_user_feedback),
        "hierarchical_header_rows": context.get("hierarchical_header_rows"),
        "rawjson_enrichment_extended": context.get("rawjson_enrichment"),
        "rawjson_summary": context.get("summary"),
        "rawjson_enrichment_slim": context.get("rawjson_enrichment_slim"),
        "structure_hash": context.get("structure_hash"),
    }
    if bundle_mode == "aggregate":
        meta["bundle_mode"] = "aggregate"
        if not isinstance(bundle_metadata_context, dict):
            bundle_metadata_context = context.get("bundle_metadata") if isinstance(context.get("bundle_metadata"), dict) else {}
        meta["bundle_size"] = context.get("bundle_size") or (
            bundle_metadata_context.get("bundle_size") if isinstance(bundle_metadata_context, dict) else None
        )
        if context.get("bundle_key"):
            meta["bundle_key"] = context.get("bundle_key")
        if context.get("bundle_metadata"):
            meta["bundle_metadata"] = context.get("bundle_metadata")
        if context.get("bundle_audit"):
            meta["bundle_audit"] = context.get("bundle_audit")
        if context.get("bundle_summary"):
            meta["bundle_summary"] = context.get("bundle_summary")
    try:
        with open(meta_path, "wb") as f:
            f.write(orjson.dumps(meta, option=orjson.OPT_INDENT_2))
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "output",
            "message": f"[ERROR] Failed to write metadata: {e}",
            "session_id": session_id,
            "path": meta_path
        })
        
    # Optional XLSX export with hierarchical headers
    try:
        if context.get("generate_xlsx", True):
            from .xlsx_exporter import export_candidate_group_pivot_xlsx
            xlsx_path = os.path.join(os.path.dirname(csv_path), "results.xlsx")
            export_candidate_group_pivot_xlsx(
                flat_headers=headers_final,
                rows=safe_rows,
                hierarchical_header_rows=context.get("hierarchical_header_rows"),
                xlsx_path=xlsx_path,
                context=context,
                format_numbers=context.get("xlsx_format_numbers", True),
                apply_color_scale=context.get("xlsx_color_scale", True)
            )
            meta["xlsx_path"] = xlsx_path
            # Rewrite metadata to include xlsx path
            try:
                with open(meta_path, "wb") as f:
                    f.write(orjson.dumps(meta, option=orjson.OPT_INDENT_2))
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"[OUTPUT_UTILS] XLSX export failed: {e}")

    # --- Audit export (NDJSON) + daily manifest for Data Framework UI ---
    try:
        exports_dir = LOG_DIR / "data_framework_exports"
        exports_dir.mkdir(parents=True, exist_ok=True)

        audit_entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "csv_path": csv_path,
            "metadata_path": meta_path,
            "state": state_eff,
            "county": county_eff,
            "contest": contest_eff,
            "structure_hash": context.get("structure_hash"),
            "principal": context.get("principal"),
            "principal_source": context.get("principal_source"),
            "session_id": session_id,
            "row_count": len(safe_rows),
            "extraction_confidence": (context.get("quality_metrics") or {}).get("extraction_confidence")
        }

        exports_file = exports_dir / "exports.jsonl"
        # append single-line JSONL
        try:
            with open(exports_file, "ab") as ef:
                ef.write(orjson.dumps(audit_entry) + b"\n")
        except Exception:
            # best-effort only
            pass

        # Build/update daily manifest with most recent N entries (keep last 500)
        try:
            manifest_path = exports_dir / f"exports-{datetime.now().strftime('%Y%m%d')}-manifest.json"
            # read last 500 lines from exports_file
            last_n = 500
            dq = deque(maxlen=last_n)
            if exports_file.exists():
                with open(exports_file, "rb") as ef:
                    for line in ef:
                        if not line.strip():
                            continue
                        try:
                            dq.append(orjson.loads(line))
                        except Exception:
                            continue
            manifest = list(dq)
            try:
                with open(manifest_path, "wb") as mf:
                    mf.write(orjson.dumps({"generated_at": datetime.now().isoformat(), "items": manifest}, option=orjson.OPT_INDENT_2))
            except Exception:
                pass
        except Exception:
            pass
    except Exception:
        pass

    return {"csv_path": csv_path, "metadata_path": meta_path}
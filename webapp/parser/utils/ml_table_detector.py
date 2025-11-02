from __future__ import annotations

# webapp/parser/utils/ml_table_detector.py
# ---------------------------------------------------------------
# Advanced ML-based Table Detection for HTML Table Extraction
# ---------------------------------------------------------------
"""
ml_table_detector.py

Advanced ML-based Table Detection for HTML Table Extraction

This module provides a robust, extensible interface for detecting and extracting tables from arbitrary HTML using
machine learning, heuristics, and hybrid approaches. It is designed to be used by table_core.py and similar utilities.

Features:
- Uses ML models (if available) to detect table regions in HTML, including non-standard and visually-styled tables.
- Optionally uses LLMs (e.g., OpenAI, local LLMs) for table region and header inference.
- Falls back to advanced heuristics and rule-based detection if ML/LLM is unavailable.
- Supports both standard <table> elements and "table-like" structures (div grids, repeated blocks, etc.).
- Optionally annotates detected tables with confidence scores, bounding boxes, and structure metadata.
- Can be extended to use external services, vision models, or LLMs for table detection.

Exports:
    - detect_tables_ml(html: str, options: dict = None) -> List[dict]
"""
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import orjson
from selectolax.parser import HTMLParser

from ..config import (
    LLM_API_KEY,
    LLM_EXTRA_INSTRUCTIONS,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_SYSTEM_PROMPT,
    TABLE_MODEL_PATH,
)
from .browser_utils import safe_attributes, safe_content
from .logger_singleton import logger
from .model_registry import TableDetectionModel

# Precompiled patterns to avoid recompilation hot spots
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]+?\}")
_SPLIT_COLS_RE = re.compile(r"\s{2,}|\t|\|")
# Truncation guard for LLM prompt
_LLM_HTML_TRUNCATE = 8000

# --- Optional LLM integration (OpenAI, local LLM, etc.) ---

def _llm_detect_tables(html: str, options: dict) -> List[Dict[str, Any]]:
    """
    Use an LLM to extract tables from HTML.
    Returns a list of {headers, data, meta}.
    """
    llm_provider = (options.get("llm_provider") or LLM_PROVIDER or "openai")
    llm_model = options.get("llm_model") or LLM_MODEL or "gpt-4-turbo"
    llm_api_key = options.get("llm_api_key") or LLM_API_KEY
    system_prompt = options.get("llm_system_prompt") or LLM_SYSTEM_PROMPT or "You are an expert at extracting tabular data from HTML."
    extra_instructions = options.get("llm_extra_instructions") or LLM_EXTRA_INSTRUCTIONS

    prompt = (
        system_prompt
        + " Given the following HTML, extract all tables (including non-standard, visually-styled, or grid-like tables). "
        + "For each table, return a JSON object with 'headers' (list of strings), 'data' (list of dicts), "
        + "and 'meta' (with any structure info you can infer). "
        + (f"Extra instructions: {extra_instructions}\n" if extra_instructions else "")
        + "HTML:\n" + html[:_LLM_HTML_TRUNCATE]
    )

    try:
        if llm_provider == "openai":
            import openai
            openai.api_key = llm_api_key
            response = openai.ChatCompletion.create(
                model=llm_model,
                messages=[{"role": "system", "content": prompt}],
                max_tokens=2048,
                temperature=0.0,
            )
            content = response["choices"][0]["message"]["content"]
        else:
            return []

        # Prefer fenced JSON blocks if present
        tables: List[Dict[str, Any]] = []

        def _try_parse(block: str) -> Optional[Dict[str, Any]]:
            try:
                obj = orjson.loads(block)
                if isinstance(obj, dict) and "headers" in obj and "data" in obj:
                    return obj
            except Exception:
                return None
            return None

        # ```json ... ```
        fenced = re.findall(r"```json\s+([\s\S]+?)```", content, flags=re.IGNORECASE)
        for blk in fenced:
            obj = _try_parse(blk)
            if obj:
                tables.append(obj)
        if tables:
            return tables

        # Fallback: any JSON-looking object
        for blk in _JSON_OBJECT_RE.findall(content):
            obj = _try_parse(blk)
            if obj:
                tables.append(obj)

        return tables
    except Exception as e:
        logger.error(f"[LLM TABLE DETECTION] Error ({llm_provider}): {e}")
        return []

def detect_tables_ml(html: str, options: Optional[dict] = None) -> List[Dict[str, Any]]:
    """
    Detect tables in HTML using ML, LLM, vision, and heuristics.
    Returns a list of dicts: {headers: [...], data: [...], meta: {...}}
    """
    options = options or {}
    use_ml = options.get("use_ml", True)
    use_llm = options.get("use_llm", False)
    use_vision = options.get("use_vision", False)
    use_heuristics = options.get("use_heuristics", True)
    use_regex = options.get("use_regex", True)

    tables: List[Dict[str, Any]] = []

    # 1) ML-based detection
    if use_ml:
        try:
            model_path = options.get("table_model_path") or TABLE_MODEL_PATH
            if model_path:
                table_model = TableDetectionModel.load_from_checkpoint(model_path)
                if table_model:
                    ml_results = table_model.predict_tables(html)
                    if ml_results:
                        tables.extend(ml_results)
        except Exception as e:
            logger.error(f"[ML TABLE DETECTION] Error loading/predicting TableDetectionModel: {e}")

    # 2) LLM-based detection
    if use_llm:
        llm_results = _llm_detect_tables(html, options)
        if llm_results:
            tables.extend(llm_results)

    # 3) Heuristics using selectolax
    html_tree = HTMLParser(html)
    if use_heuristics:
        # Standard <table> elements
        for table_node in html_tree.css("table"):
            headers, data, meta = _extract_table_from_selectolax(table_node)
            if headers and data:
                tables.append({"headers": headers, "data": data, "meta": meta})

        # Table-like structures
        for grid in html_tree.css("div,ul,ol"):
            if _looks_like_table_selectolax(grid):
                headers, data, meta = _extract_table_like_structure_selectolax(grid)
                if headers and data:
                    tables.append({"headers": headers, "data": data, "meta": meta})

    # 4) Optional: Vision-based detection
    if use_vision:
        vision_results = _vision_detect_tables(html, options)
        if vision_results:
            tables.extend(vision_results)

    # 5) Fallback regex
    if use_regex:
        regex_tables = _regex_table_detection(html)
        if regex_tables:
            tables.extend(regex_tables)

    # 6) Deduplicate by normalized header signature
    seen = set()
    unique_tables: List[Dict[str, Any]] = []
    for t in tables:
        headers = t.get("headers") or []
        sig = tuple(_normalize_header(h) for h in headers)
        if sig and sig not in seen:
            unique_tables.append(t)
            seen.add(sig)

    return unique_tables

def _ml_detect_tables(html: str, options: dict) -> List[Dict[str, Any]]:
    """
    Placeholder for ML-based table detection.
    Replace with actual model inference (vision transformer, LLM, etc.).
    """
    # Example: Use a vision model or LLM to predict table regions and extract cells
    # For now, just return empty (simulate no ML model)
    # If you have a model, run inference here and parse the output into headers/data/meta
    # Example:
    # model = YourTableDetectionModel.load_from_checkpoint(...)
    # tables = model.predict_tables(html)
    # return [{"headers": t.headers, "data": t.data, "meta": t.meta} for t in tables]
    # Optionally, use SentenceTransformer or spaCy NER for header/entity detection
    # Example (pseudo):
    # from sentence_transformers import SentenceTransformer
    # model = SentenceTransformer("fine_tuned_table_headers")
    # header_scores = model.encode(headers)
    return []

def _vision_detect_tables(html: str, options: dict) -> List[Dict[str, Any]]:
    """
    Optionally use a vision model (e.g., Donut, TableNet, PaddleOCR) to detect tables from rendered HTML screenshots.
    Returns a list of {headers, data, meta}.
    """
    # This is a placeholder. To use, render HTML to image (e.g., with Selenium, Playwright, or headless browser),
    # then run a vision model to detect table regions and extract cell text.
    # Example: Use PaddleOCR or Donut for table structure detection.
    # For now, return empty.
    return []

def _extract_table_from_selectolax(table_element) -> Tuple[List[str], List[Dict[str, str]], dict]:
    """
    Extract headers and data from a selectolax <table> element.
    Returns (headers, data, meta).
    """
    html = safe_content(table_element)
    html_tree = HTMLParser(html)
    # Defensive: check for selectolax Node API
    if not hasattr(html_tree, "css") or not callable(html_tree.css):
        return [], [], {}
    try:
        rows = html_tree.css("tr")
    except Exception:
        return [], [], {}
    if not rows:
        return [], [], {}
    # Try to find header row
    try:
        header_cells = rows[0].css("th")
        if not header_cells:
            header_cells = rows[0].css("td")
        headers = [cell.text(strip=True) for cell in header_cells]
    except Exception:
        headers = []
    data = []
    for row in rows[1:]:
        try:
            cells = row.css("td")
            if not cells:
                cells = row.css("th")
            row_data = {headers[i]: cells[i].text(strip=True) if i < len(cells) else "" for i in range(len(headers))}
            if any(v for v in row_data.values()):
                data.append(row_data)
        except Exception:
            continue
    meta = {
        "source": "selectolax_table",
        "n_rows": len(data),
        "n_cols": len(headers),
        "table_html": getattr(html_tree, "html", "")[:1000] if hasattr(html_tree, "html") else ""
    }
    return headers, data, meta

def _looks_like_table_selectolax(element) -> bool:
    """
    Heuristic: Does this selectolax element look like a table/grid? (e.g., repeated children, grid classes)
    """
    html = safe_content(element)
    html_tree = HTMLParser(html)
    tag = getattr(html_tree, "tag", "").lower() if hasattr(html_tree, "tag") else ""
    if tag == "div":
        attrs = safe_attributes(html_tree)
        classes = attrs.get("class", "")
        if any(x in classes for x in ["table", "row", "grid"]):
            return True
        children = html_tree.css("> *") if hasattr(html_tree, "css") and callable(html_tree.css) else []
        if len(children) >= 2 and all(
            len(child.css("> *")) == len(children[0].css("> *"))
            for child in children
            if hasattr(child, "css") and callable(child.css)
        ):
            return True
    if tag in ["ul", "ol"]:
        items = html_tree.css("li") if hasattr(html_tree, "css") and callable(html_tree.css) else []
        if len(items) >= 2:
            return True
    return False

def _extract_table_from_selectolax(table_node) -> Tuple[List[str], List[Dict[str, str]], dict]:
    """
    Extract headers and data from a selectolax <table> Node (no re-parse).
    """
    if not table_node or not hasattr(table_node, "css"):
        return [], [], {}

    try:
        rows = table_node.css("tr")
    except Exception:
        return [], [], {}
    if not rows:
        return [], [], {}

    # Header row
    try:
        header_cells = rows[0].css("th") or rows[0].css("td")
        headers = [c.text(strip=True) for c in header_cells] if header_cells else []
    except Exception:
        headers = []
    if not headers:
        return [], [], {}

    data: List[Dict[str, str]] = []
    for row in rows[1:]:
        try:
            cells = row.css("td") or row.css("th")
            row_map = {headers[i]: (cells[i].text(strip=True) if i < len(cells) else "") for i in range(len(headers))}
            if any(v for v in row_map.values()):
                data.append(row_map)
        except Exception:
            continue

    meta = {
        "source": "selectolax_table",
        "n_rows": len(data),
        "n_cols": len(headers),
        "table_html": getattr(table_node, "html", "")[:1000] if hasattr(table_node, "html") else ""
    }
    return headers, data, meta

def _looks_like_table_selectolax(node) -> bool:
    """
    Heuristic: Does this selectolax Node look like a table/grid?
    """
    if not node:
        return False
    tag = getattr(node, "tag", "")
    tag_lower = tag.lower() if isinstance(tag, str) else ""

    if tag_lower == "div":
        attrs = safe_attributes(node) if callable(safe_attributes) else {}
        classes = attrs.get("class", "")
        if any(x in classes for x in ["table", "row", "grid"]):
            return True
        children = node.css("> *") if hasattr(node, "css") else []
        if len(children) >= 2:
            try:
                first_cols = len(children[0].css("> *"))
                if first_cols > 0 and all(len(ch.css("> *")) == first_cols for ch in children):
                    return True
            except Exception:
                pass

    if tag_lower in ("ul", "ol"):
        items = node.css("li") if hasattr(node, "css") else []
        if len(items) >= 2:
            return True

    return False

def _extract_table_like_structure_selectolax(node) -> Tuple[List[str], List[Dict[str, str]], dict]:
    """
    Extract headers and data from a table-like structure (div grid, ul/ol) using the Node directly.
    """
    if not node or not hasattr(node, "css"):
        return [], [], {}

    rows: List[List[str]] = []
    tag = getattr(node, "tag", "")
    tag_lower = tag.lower() if isinstance(tag, str) else ""

    if tag_lower == "div":
        children = node.css("> *")
        for child in children:
            try:
                cells = child.css("> *")
                if cells:
                    rows.append([c.text(strip=True) for c in cells])
            except Exception:
                continue
    elif tag_lower in ("ul", "ol"):
        for li in node.css("li"):
            rows.append([li.text(strip=True)])

    if not rows or len(rows) < 2:
        return [], [], {}

    headers = rows[0]
    data: List[Dict[str, str]] = []
    for row in rows[1:]:
        row_map = {headers[i]: (row[i] if i < len(row) else "") for i in range(len(headers))}
        if any(v for v in row_map.values()):
            data.append(row_map)

    meta = {
        "source": "selectolax_table_like",
        "n_rows": len(data),
        "n_cols": len(headers),
        "tag": tag_lower,
        "table_html": getattr(node, "html", "")[:1000] if hasattr(node, "html") else ""
    }
    return headers, data, meta

def _regex_table_detection(html: str) -> List[Dict[str, Any]]:
    """
    Fallback: Use regex to find repeated row/column patterns in flat HTML.
    Returns list of {headers, data, meta}.
    """
    lines = [line.strip() for line in html.splitlines() if line.strip()]
    if not lines:
        return []

    # Column counts per line
    col_counts = [len(_SPLIT_COLS_RE.split(line)) for line in lines]
    if not col_counts:
        return []

    # Most common column count > 1
    count_freq = Counter(col_counts)
    common_col = max((c for c in count_freq if c > 1), key=lambda c: count_freq[c], default=None)
    if not common_col or count_freq[common_col] < 2:
        return []

    # Extract rows with the common col count
    rows = [
        _SPLIT_COLS_RE.split(line)
        for line, col_count in zip(lines, col_counts)
        if col_count == common_col
    ]
    if len(rows) < 2:
        return []

    headers = rows[0]
    data: List[Dict[str, str]] = []
    for row in rows[1:]:
        row_map = {headers[i]: (row[i] if i < len(row) else "") for i in range(len(headers))}
        if any(v for v in row_map.values()):
            data.append(row_map)

    meta = {"source": "regex_table", "n_rows": len(data), "n_cols": len(headers)}
    return [{"headers": headers, "data": data, "meta": meta}]

def _normalize_header(header: str) -> str:
    """
    Normalize header for deduplication.
    """
    return re.sub(r"\s+", " ", header.strip().lower())
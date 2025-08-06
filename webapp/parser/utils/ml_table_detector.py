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
from __future__ import annotations
import re
import orjson
from typing import List, Dict, Any, Optional, Tuple
from selectolax.parser import HTMLParser
from .shared_logger import SharedLogger
from .browser_utils import safe_content, safe_attributes
from .model_registry import TableDetectionModel

from ..config import (
    LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_SYSTEM_PROMPT, LLM_EXTRA_INSTRUCTIONS,
    TABLE_MODEL_PATH
)

logger = SharedLogger()
# --- Optional LLM integration (OpenAI, local LLM, etc.) ---
def _llm_detect_tables(html: str, options: dict) -> List[Dict[str, Any]]:
    """
    Use OpenAI LLM to extract tables from HTML.
    Returns a list of {headers, data, meta}.
    """
    # Prefer explicit options, then config.py, then sensible defaults
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
        + "HTML:\n" + html[:8000]  # Truncate for token safety
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
        # Try to extract JSON from the response
        json_blocks = re.findall(r"\{[\s\S]+?\}", content)
        tables = []
        for block in json_blocks:
            try:
                obj = orjson.loads(block)
                if "headers" in obj and "data" in obj:
                    tables.append(obj)
            except Exception:
                continue
        return tables
    except Exception as e:
        logger.error(f"[LLM TABLE DETECTION] Error ({llm_provider}): {e}")
        return []

def detect_tables_ml(html: str, options: Optional[dict] = None) -> List[Dict[str, Any]]:
    """
    Detects tables in HTML using ML, LLM, vision, and advanced heuristics.
    Returns a list of dicts: {headers: [...], data: [...], meta: {...}}
    Uses selectolax for all HTML parsing.
    """
    tables = []

    # 1. Try ML-based detection (vision or transformer model)
    if options and options.get("use_ml", True):
        try:
            model_path = options.get("table_model_path") or TABLE_MODEL_PATH
            table_model = TableDetectionModel.load_from_checkpoint(model_path)
            if table_model:
                ml_results = table_model.predict_tables(html)
                if ml_results:
                    tables.extend(ml_results)
        except Exception as e:
            logger.error(f"[ML TABLE DETECTION] Error loading TableDetectionModel: {e}")

    # 2. Optionally try LLM-based detection (OpenAI, local, etc.)
    if options and options.get("use_llm", False):
        llm_results = _llm_detect_tables(html, options)
        if llm_results:
            tables.extend(llm_results)

    # 3. Heuristic: Standard <table> extraction (with header/data detection) using selectolax
    html_tree = HTMLParser(html)
    for table in html_tree.css("table"):
        headers, data, meta = _extract_table_from_selectolax(table)
        if headers and data:
            tables.append({"headers": headers, "data": data, "meta": meta})

    # 4. Heuristic: Table-like div/ul/ol grids (repeated structures) using selectolax
    for grid in html_tree.css("div,ul,ol"):
        if _looks_like_table_selectolax(grid):
            headers, data, meta = _extract_table_like_structure_selectolax(grid)
            if headers and data:
                tables.append({"headers": headers, "data": data, "meta": meta})

    # 5. Vision-based table detection (optional, e.g., Donut, TableNet, PaddleOCR, etc.)
    if options and options.get("use_vision", False):
        vision_results = _vision_detect_tables(html, options)
        if vision_results:
            tables.extend(vision_results)

    # 6. Fallback: Regex-based row/column detection for "flat" HTML
    regex_tables = _regex_table_detection(html)
    tables.extend(regex_tables)

    # 7. Deduplicate by header signature
    seen = set()
    unique_tables = []
    for t in tables:
        sig = tuple(_normalize_header(h) for h in t["headers"])
        if sig not in seen:
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

def _extract_table_like_structure_selectolax(element) -> Tuple[List[str], List[Dict[str, str]], dict]:
    """
    Extract headers and data from a table-like structure (div grid, ul/ol) using selectolax.
    """
    html = safe_content(element)
    rows = []
    html_tree = HTMLParser(html)
    tag = getattr(html_tree, "tag", "").lower() if hasattr(html_tree, "tag") else ""
    if tag == "div":
        children = html_tree.css("> *") if hasattr(html_tree, "css") and callable(html_tree.css) else []
        for child in children:
            cell_texts = [c.text(strip=True) for c in child.css("> *")] if hasattr(child, "css") and callable(child.css) else []
            if cell_texts:
                rows.append(cell_texts)
    elif tag in ["ul", "ol"]:
        lis = html_tree.css("li") if hasattr(html_tree, "css") and callable(html_tree.css) else []
        for li in lis:
            cell_texts = [li.text(strip=True)]
            rows.append(cell_texts)
    if not rows or len(rows) < 2:
        return [], [], {}
    headers = rows[0]
    data = []
    for row in rows[1:]:
        row_data = {headers[i]: row[i] if i < len(row) else "" for i in range(len(headers))}
        if any(v for v in row_data.values()):
            data.append(row_data)
    meta = {
        "source": "selectolax_table_like",
        "n_rows": len(data),
        "n_cols": len(headers),
        "tag": tag,
        "table_html": getattr(element, "html", "")[:1000] if hasattr(element, "html") else ""
    }
    return headers, data, meta

def _regex_table_detection(html: str) -> List[Dict[str, Any]]:
    """
    Fallback: Use regex to find repeated row/column patterns in flat HTML.
    Returns list of {headers, data, meta}.
    """
    tables = []
    # Simple heuristic: look for repeated lines with similar number of columns
    lines = [l.strip() for l in html.splitlines() if l.strip()]
    col_counts = [len(re.split(r"\s{2,}|\t|\|", l)) for l in lines]
    if not col_counts:
        return []
    # Find most common col count (excluding 1)
    from collections import Counter
    count_freq = Counter(col_counts)
    common_col = max((c for c in count_freq if c > 1), key=lambda c: count_freq[c], default=None)
    if not common_col or count_freq[common_col] < 2:
        return []
    # Extract rows with this col count
    rows = [re.split(r"\s{2,}|\t|\|", l) for l, c in zip(lines, col_counts) if c == common_col]
    if len(rows) < 2:
        return []
    headers = rows[0]
    data = []
    for row in rows[1:]:
        row_data = {headers[i]: row[i] if i < len(row) else "" for i in range(len(headers))}
        if any(v for v in row_data.values()):
            data.append(row_data)
    meta = {
        "source": "regex_table",
        "n_rows": len(data),
        "n_cols": len(headers)
    }
    tables.append({"headers": headers, "data": data, "meta": meta})
    return tables

def _normalize_header(header: str) -> str:
    """
    Normalize header for deduplication.
    """
    return re.sub(r"\s+", " ", header.strip().lower())
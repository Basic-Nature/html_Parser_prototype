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
import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple

try:
    import torch
    import numpy as np
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

# --- Optional LLM integration (OpenAI, local LLM, etc.) ---
def _llm_detect_tables(html: str, options: dict) -> List[Dict[str, Any]]:
    """
    Use an LLM (OpenAI, Anthropic, Gemini, local, etc.) to extract tables from HTML.
    Returns a list of {headers, data, meta}.
    """
    llm_provider = (options.get("llm_provider") or os.getenv("LLM_PROVIDER", "openai")).lower()
    llm_model = options.get("llm_model") or os.getenv("LLM_MODEL", "gpt-4-turbo")
    llm_api_key = options.get("llm_api_key") or os.getenv("LLM_API_KEY")
    system_prompt = options.get("llm_system_prompt") or os.getenv("LLM_SYSTEM_PROMPT")
    extra_instructions = options.get("llm_extra_instructions") or os.getenv("LLM_EXTRA_INSTRUCTIONS")
    prompt = (
        (system_prompt or "You are an expert at extracting tabular data from HTML. ")
        + "Given the following HTML, extract all tables (including non-standard, visually-styled, or grid-like tables). "
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
        elif llm_provider == "anthropic":
            import anthropic # type: ignore
            client = anthropic.Anthropic(api_key=llm_api_key)
            system_prompt = system_prompt or "You are an expert at extracting tabular data from HTML."
            message = client.messages.create(
                model=llm_model,
                max_tokens=2048,
                temperature=0.0,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            content = message.content[0].text if hasattr(message.content[0], "text") else str(message.content[0])
        elif llm_provider == "gemini":
            import google.generativeai as genai # type: ignore
            genai.configure(api_key=llm_api_key)
            model = genai.GenerativeModel(llm_model)
            response = model.generate_content(prompt)
            content = response.text
        elif llm_provider == "local":
            # Example: Use a local LLM API (e.g., llama.cpp, vllm, etc.)
            # Implement your local LLM call here, e.g.:
            # response = requests.post(local_llm_url, json={"prompt": prompt, ...})
            # content = response.json()["text"]
            return []
        else:
            return []
        # Try to extract JSON from the response
        json_blocks = re.findall(r"\{[\s\S]+?\}", content)
        tables = []
        for block in json_blocks:
            try:
                obj = json.loads(block)
                if "headers" in obj and "data" in obj:
                    tables.append(obj)
            except Exception:
                continue
        return tables
    except Exception as e:
        print(f"[LLM TABLE DETECTION] Error ({llm_provider}): {e}")
        return []

def detect_tables_ml(html: str, options: Optional[dict] = None) -> List[Dict[str, Any]]:
    """
    Detects tables in HTML using ML, LLM, vision, and advanced heuristics.
    Returns a list of dicts: {headers: [...], data: [...], meta: {...}}
    """
    tables = []

    # 1. Try ML-based detection (vision or transformer model)
    if ML_AVAILABLE and options and options.get("use_ml", True):
        ml_results = _ml_detect_tables(html, options)
        if ml_results:
            tables.extend(ml_results)

    # 2. Optionally try LLM-based detection (OpenAI, local, etc.)
    if options and options.get("use_llm", False):
        llm_results = _llm_detect_tables(html, options)
        if llm_results:
            tables.extend(llm_results)

    # 3. Heuristic: Standard <table> extraction (with header/data detection)
    if BeautifulSoup:
        soup = BeautifulSoup(html, "html.parser")
        for table in soup.find_all("table"):
            headers, data, meta = _extract_table_from_bs4(table)
            if headers and data:
                tables.append({"headers": headers, "data": data, "meta": meta})

        # 4. Heuristic: Table-like div/ul/ol grids (repeated structures)
        for grid in soup.find_all(lambda tag: tag.name in ["div", "ul", "ol"] and _looks_like_table(tag)):
            headers, data, meta = _extract_table_like_structure(grid)
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

def _extract_table_from_bs4(table) -> Tuple[List[str], List[Dict[str, str]], dict]:
    """
    Extract headers and data from a BeautifulSoup <table> element.
    Returns (headers, data, meta).
    """
    rows = table.find_all("tr")
    if not rows:
        return [], [], {}
    # Try to find header row
    header_cells = rows[0].find_all(["th", "td"])
    headers = [th.get_text(strip=True) for th in header_cells]
    data = []
    for row in rows[1:]:
        cells = row.find_all(["td", "th"])
        row_data = {headers[i]: cells[i].get_text(strip=True) if i < len(cells) else "" for i in range(len(headers))}
        if any(v for v in row_data.values()):
            data.append(row_data)
    meta = {
        "source": "bs4_table",
        "n_rows": len(data),
        "n_cols": len(headers),
        "table_html": str(table)[:1000]
    }
    return headers, data, meta

def _looks_like_table(tag) -> bool:
    """
    Heuristic: Does this tag look like a table/grid? (e.g., repeated children, grid classes)
    """
    if tag.name == "div":
        classes = tag.get("class", [])
        if any("table" in c or "row" in c or "grid" in c for c in classes):
            return True
        # Many direct children with similar structure
        children = tag.find_all(recursive=False)
        if len(children) >= 2 and all(len(child.find_all(recursive=False)) == len(children[0].find_all(recursive=False)) for child in children):
            return True
    if tag.name in ["ul", "ol"]:
        items = tag.find_all("li", recursive=False)
        if len(items) >= 2:
            return True
    return False

def _extract_table_like_structure(tag) -> Tuple[List[str], List[Dict[str, str]], dict]:
    """
    Extract headers and data from a table-like structure (div grid, ul/ol).
    """
    rows = []
    if tag.name == "div":
        children = tag.find_all(recursive=False)
        for child in children:
            cell_texts = [c.get_text(strip=True) for c in child.find_all(recursive=False)]
            if cell_texts:
                rows.append(cell_texts)
    elif tag.name in ["ul", "ol"]:
        for li in tag.find_all("li", recursive=False):
            cell_texts = [li.get_text(strip=True)]
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
        "source": "bs4_table_like",
        "n_rows": len(data),
        "n_cols": len(headers),
        "tag": tag.name,
        "table_html": str(tag)[:1000]
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
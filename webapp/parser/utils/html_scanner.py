import os
import re
import time
from collections import defaultdict
from typing import Dict, Any, List
from ..utils.logger_instance import logger


# Tags to extract (add more as needed)
HTML_TAGS = [
    "html", "head", "title", "body", "h1", "h2", "h3", "h4", "h5", "h6",
    "b", "i", "center", "ul", "li", "br", "p", "hr", "img", "a", "span", "div", "button", "input", "form", "table"
]

# Regex patterns for tag extraction (captures start tag, attributes, and content)
TAG_PATTERN = re.compile(
    r"<({tags})(\s[^>]*)?>.*?</\1\s*>|<({tags})(\s[^>]*)?/?>".format(
        tags="|".join(HTML_TAGS)
    ),
    re.IGNORECASE | re.DOTALL
)

# Regex for extracting attributes from a tag
ATTR_PATTERN = re.compile(r'(\w+)\s*=\s*["\']([^"\']+)["\']')

def extract_tagged_segments_with_attrs(html: str) -> List[Dict[str, Any]]:
    """
    Extracts all segments from the HTML that match the specified tags.
    Returns a list of dicts: {tag, attrs, html, classes, id, is_button, is_clickable}
    """
    segments = []
    for match in TAG_PATTERN.finditer(html):
        segment = match.group(0)
        tag = match.group(1) or match.group(3)
        attrs_str = match.group(2) or match.group(4) or ""
        attrs = dict(ATTR_PATTERN.findall(attrs_str))
        classes = attrs.get("class", "").split() if "class" in attrs else []
        id_ = attrs.get("id", "")
        # Button/clickable detection
        is_button = tag.lower() == "button" or (tag.lower() == "input" and attrs.get("type", "").lower() in ["button", "submit"])
        is_clickable = is_button or tag.lower() == "a" or "onclick" in attrs or "btn" in classes or "button" in classes
        segments.append({
            "tag": tag,
            "attrs": attrs,
            "classes": classes,
            "id": id_,
            "html": segment,
            "is_button": is_button,
            "is_clickable": is_clickable
        })
    return segments

def scan_html_for_context(page, debug=False) -> Dict[str, Any]:
    """
    Scans the HTML for all relevant tags and downloadable formats.
    Returns a context dict ready for organize_context.
    """
    context_result = {
        "raw_html": "",
        "tagged_segments": [],
        "tagged_segments_with_attrs": [],
        "available_formats": [],
        "metadata": {},
        "selector_log": [],
        "error": None,
    }

    try:
        # Wait for page content to load
        SCAN_WAIT_SECONDS = 7
        logger.info(f"[SCAN] Waiting {SCAN_WAIT_SECONDS} seconds for page content to load...")
        time.sleep(SCAN_WAIT_SECONDS)
        html = page.content()
        context_result["raw_html"] = html

        # Extract all tagged segments as strings and with attributes
        segments = []
        segments_with_attrs = extract_tagged_segments_with_attrs(html)
        context_result["tagged_segments_with_attrs"] = segments_with_attrs
        context_result["tagged_segments"] = [seg["html"] for seg in segments_with_attrs]

        # Collect all unique selectors/classes/ids for logging and downstream use
        selector_log = set()
        for seg in segments_with_attrs:
            if seg["id"]:
                selector_log.add(f'#{seg["id"]}')
            for cls in seg["classes"]:
                selector_log.add(f'.{cls}')
            selector_log.add(seg["tag"].lower())
        context_result["selector_log"] = sorted(selector_log)

        # Detect downloadable formats (CSV, JSON, PDF, etc.)
        supported_formats = [".csv", ".json", ".pdf", ".xlsx", ".xls"]
        for a in page.query_selector_all("a[href]"):
            try:
                href = a.get_attribute("href")
                if not href:
                    continue
                for ext in supported_formats:
                    if href.lower().endswith(ext):
                        context_result["available_formats"].append((ext.lstrip('.'), page.urljoin(href)))
                        break
            except Exception:
                continue

        # Metadata for context organizer
        context_result["metadata"] = {
            "source_url": page.url,
            "scrape_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        if debug:
            print("\n[DEBUG] Extracted HTML segments with attrs:")
            for seg in segments_with_attrs:
                print(f"{seg['tag']} {seg['attrs']} {seg['html'][:80]}{'...' if len(seg['html']) > 80 else ''}")

    except Exception as e:
        err_msg = f"[SCAN ERROR] HTML parsing failed: {e}"
        logger.error(err_msg)
        context_result["error"] = err_msg
    return context_result
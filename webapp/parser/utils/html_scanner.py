import json
import os
import re
import time

from collections import defaultdict
from typing import Dict, Any, List
from ..config import CONTEXT_LIBRARY_PATH
from ..utils.download_utils import download_file
from ..utils.format_router import prompt_user_for_format, route_format_handler 
from ..utils.logger_instance import logger, logging
from rich import print as rprint
from ..utils.user_prompt import prompt_user_input, PromptCancelled

if os.path.exists(CONTEXT_LIBRARY_PATH):
    with open(CONTEXT_LIBRARY_PATH, "r", encoding="utf-8") as f:
        CONTEXT_LIBRARY = json.load(f)
    supported_formats = CONTEXT_LIBRARY.get("supported_formats", {})
    supported_links = [link for link in CONTEXT_LIBRARY.get("download_links", []) if link["format"] in supported_formats]
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

def extract_download_links_from_html(html, exts=(".csv", ".json", ".pdf")):
    """
    Extracts download links from HTML for given file extensions.
    Returns a list of dicts: {"href": ..., "format": ...}
    """
    pattern = re.compile(r'<a[^>]+href=["\']([^"\']+\.(?:csv|json|pdf))["\']', re.IGNORECASE)
    links = []
    for match in pattern.finditer(html):
        href = match.group(1)
        for ext in exts:
            if href.lower().endswith(ext):
                links.append({"href": href, "format": ext})
    return links

def scan_html_for_context(target_url, page, debug=False) -> Dict[str, Any]:
    """
    Scans the HTML for all relevant tags and downloadable formats.
    Returns a context dict ready for organize_context.
    Ensures the URL is always included for downstream context organization.
    """
    # Load context library and supported formats/links
    if os.path.exists(CONTEXT_LIBRARY_PATH):
        with open(CONTEXT_LIBRARY_PATH, "r", encoding="utf-8") as f:
            CONTEXT_LIBRARY = json.load(f)
        supported_formats = CONTEXT_LIBRARY.get("supported_formats", {})
        # Only keep links whose format is supported
        supported_links = [link for link in CONTEXT_LIBRARY.get("download_links", []) if link["format"] in supported_formats]
    else:
        supported_formats = {}
        supported_links = []

    context_result = {
        "raw_html": "",
        "tagged_segments": [],
        "tagged_segments_with_attrs": [],
        "available_formats": list(supported_formats) if isinstance(supported_formats, list) else list(supported_formats.keys()),
        "metadata": {},
        "selector_log": [],
        "error": None,
        "url": page.url,  # Always include the URL at the top level
    }

    try:
        page_url = target_url or page.url
        SCAN_WAIT_SECONDS = 3
        logger.info(f"[SCAN] Waiting {SCAN_WAIT_SECONDS} to scan page content...")
        time.sleep(SCAN_WAIT_SECONDS)
        html = page.content()
        context_result["raw_html"] = html

        # --- Download link extraction and merging ---
        # Extract download links from the HTML (e.g., <a href="...json">)
        dynamic_links = extract_download_links_from_html(html)
        # Merge links from context library and those found dynamically, deduplicating by (href, format)
        all_links = { (l["href"], l["format"]): l for l in (supported_links + dynamic_links) }
        supported_links = list(all_links.values())
        # Store all found download links in metadata for downstream use
        context_result["metadata"]["download_links"] = supported_links       

        # --- Downloadable file prompt logic ---
        if supported_links:
            # Show user what formats are available for download
            available_files = [f"{os.path.basename(link['href'])} ({link['format']})" for link in supported_links]
            rprint(f"[cyan]Downloadable file(s) found: {', '.join(available_files)}.[/cyan]")
            rprint("[magenta]Would you like to download one now? (y/n) (type 'cancel' to abort)[/magenta]")
            user_input = prompt_user_input("> ")
            if user_input and user_input.strip().lower().startswith("y"):
                # If multiple formats, ask user which one to download
                if len(supported_links) > 1:
                    rprint("[bold cyan]Which format do you want to download?[/bold cyan] " + ", ".join(available_files))
                    chosen_fmt = prompt_user_input("> ").strip().lower()
                    # Find the chosen link by format
                    chosen_link = next((l for l in supported_links if l["format"].lower() == chosen_fmt.lower()), None)
                else:
                    chosen_link = supported_links[0]
                if chosen_link:
                    # Download the file (or skip if already downloaded)
                    from ..html_election_parser import mark_url_processed
                    local_file = download_file(page.url, chosen_link["href"])
                    # Always route the file (downloaded or already present) to the format handler
                    if local_file:
                        fmt = chosen_link["format"]
                        format_handler = route_format_handler(fmt)
                        if format_handler and hasattr(format_handler, "parse"):
                            # Parse the file using the appropriate handler
                            result = format_handler.parse(None, {"manual_file": local_file, "source_url": target_url})
                            if result and all(result):
                                # If parsing succeeded, mark URL as processed with metadata
                                *_, metadata = result
                                mark_url_processed(target_url, status="success", **metadata)
                            else:
                                # If parsing failed, mark as fail
                                mark_url_processed(target_url, status="fail")
                            # Return early since download/parse is handled
                            return context_result
                if not chosen_link:
                    # User entered a format that wasn't found
                    rprint(f"[red]No download link found for format: {chosen_fmt}[/red]")
            else:
                # User declined download; log URLs for later use in metadata
                context_result["metadata"]["download_links"] = [
                    {"format": link["format"], "url": link["href"]} for link in supported_links
                ]

        # --- HTML tag extraction for context organization ---
        # Extract all tagged segments as strings and with attributes
        segments_with_attrs = extract_tagged_segments_with_attrs(html)
        for seg in segments_with_attrs:
            if not isinstance(seg, dict):
                continue
            if not all(k in seg for k in ("tag", "attrs", "html")):
                continue
            print(
                f"<{seg['tag']}"
                + (f" id='{seg['id']}'" if seg.get('id') else "")
                + (f" class='{', '.join(seg['classes'])}'" if seg.get('classes') else "")
                + f"> {seg['html'][:60].replace(chr(10), ' ')}{'...' if len(seg['html']) > 60 else ''}"
            )
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

        # Add metadata for context organizer
        context_result["metadata"].update({
            "source_url": page.url,
            "scrape_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        # --- Debug output for development ---
        if debug:
            rprint("\n[orange][DEBUG] Extracted HTML segments with attrs:[/orange]")
            for seg in segments_with_attrs:
                rprint(f"{seg['tag']} {seg['attrs']} {seg['html'][:80]}{'...' if len(seg['html']) > 80 else ''}")
            if supported_links:
                rprint("\n[orange][DEBUG] Detected download links:[/orange]")
                for link in supported_links:
                    # Print the actual file name (not just extension)
                    file_name = os.path.basename(link["href"])
                    rprint(f"[green]  - {file_name} ({link['format']})[/green]")
    except Exception as e:
        # Log and store any error that occurs during scanning
        rprint(f"[SCAN ERROR] HTML parsing failed: {e}")
        logger.error(f"[SCAN ERROR] HTML parsing failed: {e}")
        context_result["error"] = f"[SCAN ERROR] HTML parsing failed: {e}"

    logger.debug(f"Available formats detected: {context_result['available_formats']}")
    return context_result
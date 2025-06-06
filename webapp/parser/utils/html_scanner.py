import hashlib
import json
import os
import re
import time

from typing import Dict, Any, List, Optional, Callable, Set
from ..config import CONTEXT_LIBRARY_PATH
from ..utils.download_utils import download_file
from ..utils.format_router import route_format_handler 
from ..utils.logger_instance import logger
from rich import print as rprint
from ..utils.user_prompt import prompt_user_input
from selectolax.parser import HTMLParser
from sentence_transformers import SentenceTransformer
from ..bots.librarian import (
    HTML_TAGS, PANEL_TAGS, HEADING_TAGS, CUSTOM_ATTR_PATTERNS, LOCATION_KEYWORDS, CANDIDATE_KEYWORDS, BALLOT_TYPES,
    extend_panel_tags, extend_heading_tags, extend_html_tags, extend_custom_attr_patterns,
    log_unknown_tag, log_unknown_attr
)
import numpy as np

from bs4 import BeautifulSoup, Tag

# --- No longer need local UNKNOWN_TAGS_LOG/UNKNOWN_ATTRS_LOG ---

# Example: dynamically extend from learning/feedback
extend_panel_tags(["custom-panel"])
extend_custom_attr_patterns([r"^x-data-"])
extend_heading_tags(["custom-heading", "special-h2"])
extend_html_tags(["custom-element", "widget"])

def load_additional_tags_from_context_library():
    tags = set()
    if os.path.exists(CONTEXT_LIBRARY_PATH):
        with open(CONTEXT_LIBRARY_PATH, "r", encoding="utf-8") as f:
            context_lib = json.load(f)
            for key in ["panel_tags", "table_tags", "section_keywords"]:
                if key in context_lib and isinstance(context_lib[key], list):
                    tags.update([t.lower() for t in context_lib[key] if isinstance(t, str)])
    return tags
HTML_TAGS |= load_additional_tags_from_context_library()

def safe_log_path(filename: str, log_dir: str = "log") -> str:
    from ..config import BASE_DIR
    filename = _sanitize_log_filename(filename)
    parent_dir = os.path.dirname(BASE_DIR)
    log_folder = os.path.join(parent_dir, log_dir)
    os.makedirs(log_folder, exist_ok=True)
    full_path = os.path.join(log_folder, filename)
    if not os.path.abspath(full_path).startswith(os.path.abspath(log_folder)):
        raise ValueError("Unsafe log path detected!")
    return full_path

def _sanitize_log_filename(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)

def extract_attrs_bs4(bs4_tag: Tag) -> Dict[str, Any]:
    """Extract attributes from a BeautifulSoup Tag object, including data-* attributes."""
    attrs = {}
    for k, v in bs4_tag.attrs.items():
        if isinstance(v, list):
            attrs[k] = " ".join(v)
        elif v is None:
            attrs[k] = True
        else:
            attrs[k] = v
        log_unknown_attr(k)
    # Include data-* attributes
    for k, v in bs4_tag.attrs.items():
        if k.startswith("data-"):
            attrs[k] = v
    return attrs

def extract_custom_attrs(attrs: Dict[str, Any], include_data: bool = True) -> Dict[str, Any]:
    """Extract custom attributes (data-*, aria-*, role, etc.) based on dynamic patterns."""
    custom = {}
    for k, v in attrs.items():
        for pat in CUSTOM_ATTR_PATTERNS:
            if pat.match(k):
                custom[k] = v
                break
        else:
            log_unknown_attr(k)
    return custom

def extract_tagged_segments_with_attrs(
    html: str,
    include_data_attrs: bool = True,
    fallback_on_error: bool = True
) -> List[Dict[str, Any]]:
    start_time = time.time()
    segments: List[Dict[str, Any]] = []
    heading_tags = HEADING_TAGS
    panel_tags = PANEL_TAGS

    try:
        tree = HTMLParser(html)
        def walk(node, parent_idx=None, heading_idx=None, panel_idx=None):
            tag = node.tag
            if not tag or tag.lower() not in HTML_TAGS:
                log_unknown_tag(tag)
                for child in node.iter(include_text=True):
                    walk(child, parent_idx, heading_idx, panel_idx)
                return
            attrs = dict(node.attributes)
            if include_data_attrs:
                attrs.update({k: v for k, v in node.attributes.items() if k.startswith("data-")})
            for k in attrs:
                log_unknown_attr(k)
            classes = attrs.get("class", "").split() if "class" in attrs else []
            id_ = attrs.get("id", "")
            is_button = tag == "button" or (tag == "input" and attrs.get("type", "").lower() in ["button", "submit"])
            is_clickable = is_button or tag == "a" or "onclick" in attrs or "btn" in classes or "button" in classes

            this_heading_idx = heading_idx
            if tag.lower() in heading_tags:
                this_heading_idx = len(segments)

            this_panel_idx = panel_idx
            if tag.lower() in panel_tags:
                this_panel_idx = len(segments)

            seg = {
                "tag": tag.lower(),
                "attrs": attrs,
                "classes": classes,
                "id": id_,
                "html": "",
                "is_button": is_button,
                "is_clickable": is_clickable,
                "parent_idx": parent_idx,
                "children": [],
                "start": getattr(node, "start", None),
                "end": getattr(node, "end", None),
                "_idx": len(segments),
                "context_heading": None,
                "panel_ancestor_idx": this_panel_idx,
                "panel_ancestor_heading": None,
            }
            if hasattr(node, "start") and hasattr(node, "end") and node.start is not None and node.end is not None:
                html_bytes = html.encode("utf-8")
                try:
                    seg["html"] = html_bytes[node.start:node.end].decode("utf-8", errors="replace")
                except Exception:
                    seg["html"] = html[node.start:node.end]
            else:
                seg["html"] = ""
            segments.append(seg)
            this_idx = seg["_idx"]
            for child in node.iter(include_text=True):
                child_idx = walk(child, this_idx, this_heading_idx, this_panel_idx)
                if child_idx is not None:
                    seg["children"].append(child_idx)
            return this_idx

        root = tree.body or tree.html or tree.root
        walk(root)

        # Second pass: assign context_heading and panel_ancestor_heading
        for seg in segments:
            if seg["tag"] in panel_tags or seg["tag"] == "table":
                parent_idx = seg["parent_idx"]
                heading_html = None
                while parent_idx is not None:
                    parent = segments[parent_idx]
                    if parent["tag"] in heading_tags:
                        heading_html = parent["html"]
                        break
                    parent_idx = parent["parent_idx"]
                seg["context_heading"] = heading_html

            if seg["tag"] == "table" and seg["panel_ancestor_idx"] is not None:
                panel_node = segments[seg["panel_ancestor_idx"]]
                seg["panel_ancestor_heading"] = panel_node.get("context_heading")

        logger.info(f"[PERF] DOM extraction (selectolax) took {time.time() - start_time:.2f} seconds, {len(segments)} segments.")
        return segments
    except Exception as e:
        logger.error(f"[FALLBACK] selectolax failed: {e}")
        if not fallback_on_error:
            raise
        soup = BeautifulSoup(html, "html.parser")
        def walk_bs4(node, parent_idx=None, heading_idx=None, start_search=0):
            if not isinstance(node, Tag):
                return start_search
            tag = node.name.lower()
            if tag not in HTML_TAGS:
                log_unknown_tag(tag)
                for child in node.children:
                    start_search = walk_bs4(child, parent_idx, heading_idx, start_search)
                return start_search
            tag_html = str(node)
            start, end = html.find(tag_html, start_search), -1
            if start != -1:
                end = start + len(tag_html)
            attrs = extract_attrs_bs4(node)
            for k in attrs:
                log_unknown_attr(k)
            classes = attrs.get("class", "").split() if "class" in attrs else []
            id_ = attrs.get("id", "")
            is_button = tag == "button" or (tag == "input" and attrs.get("type", "").lower() in ["button", "submit"])
            is_clickable = is_button or tag == "a" or "onclick" in attrs or "btn" in classes or "button" in classes

            this_heading_idx = heading_idx
            if tag in heading_tags:
                this_heading_idx = len(segments)

            seg = {
                "tag": tag,
                "attrs": attrs,
                "classes": classes,
                "id": id_,
                "html": tag_html,
                "is_button": is_button,
                "is_clickable": is_clickable,
                "parent_idx": parent_idx,
                "children": [],
                "start": start,
                "end": end,
                "_idx": len(segments),
                "context_heading": None
            }
            segments.append(seg)
            this_idx = seg["_idx"]
            for child in node.children:
                start_search = walk_bs4(child, this_idx, this_heading_idx, start_search)
            return end if end > 0 else start_search

        root = soup.find("html") or soup.find("body") or soup
        walk_bs4(root)

        for seg in segments:
            if seg["tag"] in panel_tags or seg["tag"] == "table":
                parent_idx = seg["parent_idx"]
                heading_html = None
                while parent_idx is not None:
                    parent = segments[parent_idx]
                    if parent["tag"] in heading_tags:
                        heading_html = parent["html"]
                        break
                    parent_idx = parent["parent_idx"]
                seg["context_heading"] = heading_html

        logger.info(f"[PERF] DOM extraction (BeautifulSoup fallback) took {time.time() - start_time:.2f} seconds, {len(segments)} segments.")
        return []

def extract_panel_table_hierarchy(segments):
    from bs4 import BeautifulSoup
    panel_tags = PANEL_TAGS
    panels = []
    found_panel = False

    def is_panel(seg):
        tag = seg["tag"]
        classes = [c.lower() for c in seg.get("classes", [])]
        id_ = seg.get("id", "").lower()
        # Panel if tag in PANEL_TAGS or class/id contains panel-like keywords
        panel_like = ["panel", "card", "container", "box", "section-panel"]
        return (
            tag in PANEL_TAGS
            or any(cls in panel_like for cls in classes)
            or any(p in id_ for p in panel_like)
        )

    def extract_heading_text(heading_html):
        if not heading_html:
            return None
        soup = BeautifulSoup(heading_html, "html.parser")
        span = soup.find("span")
        if span and span.get_text(strip=True):
            return span.get_text(strip=True)
        return soup.get_text(strip=True)

    for seg in segments:
        if is_panel(seg):
            found_panel = True
            panel_heading_html = seg.get("context_heading")
            panel_heading = extract_heading_text(panel_heading_html)
            panel = {
                "panel_idx": seg["_idx"],
                "panel_tag": seg["tag"],
                "panel_heading": panel_heading,
                "panel_html": seg["html"],
                "fully_reported": extract_fully_reported_from_panel(seg["html"]),
                "tables": []
            }
            stack = list(seg["children"])
            while stack:
                child_idx = stack.pop()
                child = segments[child_idx]
                if child["tag"] == "table":
                    context_heading_html = child.get("context_heading")
                    context_heading = extract_heading_text(context_heading_html)
                    panel_ancestor_heading_html = child.get("panel_ancestor_heading")
                    panel_ancestor_heading = extract_heading_text(panel_ancestor_heading_html)
                    panel["tables"].append({
                        "table_idx": child["_idx"],
                        "table_html": child["html"],
                        "context_heading": context_heading,
                        "panel_ancestor_heading": panel_ancestor_heading,
                    })
                stack.extend(child["children"])
            if panel["tables"]:
                panels.append(panel)
    if not found_panel:
        for seg in segments:
            if seg["tag"] == "table":
                context_heading_html = seg.get("context_heading")
                context_heading = extract_heading_text(context_heading_html)
                panel_ancestor_heading_html = seg.get("panel_ancestor_heading")
                panel_ancestor_heading = extract_heading_text(panel_ancestor_heading_html)
                panels.append({
                    "panel_idx": seg["_idx"],
                    "panel_tag": "table",
                    "panel_heading": context_heading,
                    "panel_html": seg["html"],
                    "tables": [{
                        "table_idx": seg["_idx"],
                        "table_html": seg["html"],
                        "context_heading": context_heading,
                        "panel_ancestor_heading": panel_ancestor_heading,
                    }]
                })
    return panels

def extract_fully_reported_from_panel(panel_html):
    soup = BeautifulSoup(panel_html, "html.parser")
    for span in soup.find_all("span", class_="fw-bold"):
        txt = span.get_text(strip=True)
        if "Reported" in txt:
            return txt
    txt = soup.get_text(" ", strip=True)
    for part in txt.splitlines():
        if "Reported" in part:
            return part.strip()
    return ""

TAG_PATTERN = re.compile(
    r"<({tags})(\s[^>]*)?>.*?</\1\s*>|<({tags})(\s[^>]*)?/?>".format(
        tags="|".join(HTML_TAGS)
    ),
    re.IGNORECASE | re.DOTALL
)
ATTR_PATTERN = re.compile(
    r'([a-zA-Z_:][a-zA-Z0-9_\-.:]*)'
    r'(?:\s*=\s*'
    r'(?:'
    r'"([^"]*)"'
    r"|"
    r"'([^']*)'"
    r"|"
    r'([^\s"\'=<>`]+)'
    r'))?',
    re.UNICODE
)

def extract_attrs(attr_str):
    attrs = {}
    for match in ATTR_PATTERN.finditer(attr_str):
        name = match.group(1)
        value = match.group(2) if match.group(2) is not None else (
            match.group(3) if match.group(3) is not None else (
                match.group(4) if match.group(4) is not None else None
            )
        )
        if value is None:
            attrs[name] = True
        else:
            attrs[name] = value
        log_unknown_attr(name)
    return attrs
# Load or initialize the DOM pattern knowledge base
def load_pattern_kb():
    kb = []
    path = safe_log_path("dom_pattern_kb.jsonl")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    kb.append(json.loads(line))
                except Exception:
                    continue
    return kb

def save_pattern_kb(kb):
    path = safe_log_path("dom_pattern_kb.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for entry in kb:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
def append_pattern_kb(entry):
    path = safe_log_path("dom_pattern_kb.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
def append_feedback_log(entry):
    path = safe_log_path("segment_feedback_log.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def get_page_hash(page):
    content = page.content()
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def extract_download_links_from_html(html, exts=(".csv", ".json", ".pdf")):
    pattern = re.compile(r'<a[^>]+href=["\']([^"\']+\.(?:csv|json|pdf))["\']', re.IGNORECASE)
    links = []
    for match in pattern.finditer(html):
        href = match.group(1)
        for ext in exts:
            if href.lower().endswith(ext):
                links.append({"href": href, "format": ext})
    return links

# --- ML/Embedding/Clustering helpers ---

def auto_label_segment(segment):
    tag = segment.get("tag", "")
    classes = [c.lower() for c in segment.get("classes", [])]
    attrs = segment.get("attrs", {})
    html = segment.get("html", "").lower()
    id_ = segment.get("id", "").lower()

    # 1. Table
    if tag == "table":
        return "results_table"

    # 2. Ballot toggle/button
    if segment.get("is_button") or "btn" in classes or "button" in classes or "toggle" in classes or "toggle" in id_:
        return "ballot_toggle"

    # 3. Heading
    if tag in HEADING_TAGS or any(cls in ["heading", "header", "title"] for cls in classes):
        return "heading"

    # 4. Panel
    if tag in PANEL_TAGS or any(cls in ["panel", "card", "container", "box", "section-panel"] for cls in classes):
        return "panel"

    # 5. Download link
    if tag == "a" and "href" in attrs:
        href = str(attrs["href"]).lower()
        if any(href.endswith(ext) for ext in [".csv", ".json", ".pdf", ".xlsx"]):
            return "download_link"

    # 6. Location/candidate panel
    if any(kw in html for kw in LOCATION_KEYWORDS):
        return "location_panel"
    if any(kw in html for kw in CANDIDATE_KEYWORDS):
        return "candidate_panel"

    # 7. Ballot type
    if any(bt in html for bt in BALLOT_TYPES):
        return "ballot_type"

    # 8. Clickable
    if segment.get("is_clickable"):
        return "clickable"

    # 9. Ignore empty or decorative
    if tag in {"br", "hr", "i", "b", "u", "span"} and not html.strip():
        return "ignore"

    # 10. Fallback
    return "ignore"

def get_segment_embedding(model, segment):
    # Use text and tag/attrs for embedding
    text = segment.get("html", "")
    tag = segment.get("tag", "")
    attrs = " ".join([f"{k}={v}" for k, v in segment.get("attrs", {}).items()])
    full_text = f"{tag} {attrs} {text}"   
    return model.encode(full_text, convert_to_numpy=True)


def cosine_sim(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def ml_classify_segment(segment, model, pattern_kb, threshold=0.85):
    """
    Classify a segment by comparing its embedding to known clusters in the KB.
    Returns (label, confidence, matched_pattern_id)
    """
    emb = get_segment_embedding(model, segment)
    best_label = "unknown"
    best_conf = 0.0
    best_pattern_id = None
    for entry in pattern_kb:
        kb_emb = np.array(entry.get("embedding", []))
        if kb_emb.shape != emb.shape:
            continue
        sim = cosine_sim(emb, kb_emb)
        if sim > best_conf:
            best_conf = sim
            best_label = entry.get("label", "unknown")
            best_pattern_id = entry.get("pattern_id")
    # If no match above threshold, label as unknown
    if best_conf < threshold:
        return "unknown", best_conf, None
    return best_label, best_conf, best_pattern_id


def prompt_for_segment_label(segment):
    # Try to auto-label first
    auto = auto_label_segment(segment)
    if auto != "ignore":
        return auto
    # Fallback to user prompt if ambiguous
    html_preview = segment.get("html", "")
    if not html_preview:
        html_preview = f"[No HTML] tag={segment.get('tag')} attrs={segment.get('attrs')}"
    rprint(f"\n[bold yellow]Segment needs review:[/bold yellow]\n{html_preview[:200]}{'...' if len(html_preview) > 200 else ''}")
    rprint("[cyan]What is the semantic role of this segment? (e.g., results_table, ballot_toggle, heading, ignore, etc.)[/cyan]")
    label = prompt_user_input("> ").strip()
    return label

def scan_html_for_context(
    target_url, 
    page, 
    debug=False, 
    context_cache=None, 
    rejected_downloads: Optional[set] = None  # <-- add this
) -> Dict[str, Any]:
    """
    Advanced HTML scanner with ML-driven DOM pattern clustering, active learning, dynamic tagging,
    confidence-driven processing, and persistent knowledge base.
    """
    if rejected_downloads is None:
        rejected_downloads = set()
    # --- 1. Caching ---
    page_hash = get_page_hash(page)
    if context_cache is not None and page_hash in context_cache:
        logger.info(f"[SCAN] Using cached context for {target_url}")
        return context_cache[page_hash]

    # --- 2. Load context library and supported formats/links ---
    if os.path.exists(CONTEXT_LIBRARY_PATH):
        with open(CONTEXT_LIBRARY_PATH, "r", encoding="utf-8") as f:
            CONTEXT_LIBRARY = json.load(f)
        supported_formats = CONTEXT_LIBRARY.get("supported_formats", {})
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
        "url": page.url,
        "pattern_kb_matches": [],
        "segments_needing_review": [],
    }

    try:
        page_url = target_url or page.url
        SCAN_WAIT_SECONDS = 3
        logger.info(f"[SCAN] Waiting {SCAN_WAIT_SECONDS} to scan page content...")
        time.sleep(SCAN_WAIT_SECONDS)
        html = page.content()
        context_result["raw_html"] = html

        # --- 3. Download link extraction and merging ---
        dynamic_links = extract_download_links_from_html(html)
        all_links = { (l["href"], l["format"]): l for l in (supported_links + dynamic_links) }
        supported_links = list(all_links.values())
        context_result["metadata"]["download_links"] = supported_links       

        # --- 4. Downloadable file prompt logic (with ML-driven format clustering) ---
        # Cluster available formats for review and ML learning
        format_kb = load_pattern_kb()
        for link in supported_links:
            fmt = link["format"]
            # Use ML to cluster/categorize format extensions
            # (Stub: just log for now, but could embed file metadata if downloaded)
            append_pattern_kb({
                "pattern_id": f"format_{fmt}_{os.path.basename(link['href'])}",
                "label": "download_format",
                "format": fmt,
                "href": link["href"],
                "source_url": page.url,
                "timestamp": time.time(),
                "embedding": [],  # Could add file content embedding if downloaded
            })

        # --- Filter out rejected files ---
        new_links = [link for link in supported_links if link["href"] not in rejected_downloads]

        if new_links:
            available_files = [f"{os.path.basename(link['href'])} ({link['format']})" for link in new_links]
            rprint(f"[cyan]Downloadable file(s) found: {', '.join(available_files)}.[/cyan]")
            rprint("[magenta]Would you like to download one now? (y/n) (type 'cancel' to abort)[/magenta]")
            user_input = prompt_user_input("> ")
            if user_input and user_input.strip().lower().startswith("y"):
                if len(new_links) > 1:
                    rprint("[bold cyan]Which format do you want to download?[/bold cyan] " + ", ".join(available_files))
                    chosen_fmt = prompt_user_input("> ").strip().lower()
                    chosen_link = next((l for l in new_links if l["format"].lower() == chosen_fmt.lower()), None)
                else:
                    chosen_link = new_links[0]
                if chosen_link:
                    from ..html_election_parser import mark_url_processed
                    local_file = download_file(page.url, chosen_link["href"])
                    if local_file:
                        fmt = chosen_link["format"]
                        format_handler = route_format_handler(fmt)
                        if format_handler and hasattr(format_handler, "parse"):
                            result = format_handler.parse(None, {"manual_file": local_file, "source_url": target_url})
                            if result and all(result):
                                *_, metadata = result
                                mark_url_processed(target_url, status="success", **metadata)
                            else:
                                mark_url_processed(target_url, status="fail")
                            return context_result
                    pass
                if not chosen_link:
                    rprint(f"[red]No download link found for format: {chosen_fmt}[/red]")
            else:
                # User rejected all new files, add to rejected_downloads
                for link in new_links:
                    rejected_downloads.add(link["href"])
                context_result["metadata"]["download_links"] = [
                    {"format": link["format"], "url": link["href"]} for link in supported_links
                ]

        # --- 5. HTML tag extraction for context organization ---
        segments_with_attrs = extract_tagged_segments_with_attrs(html)
        context_result["tagged_segments_with_attrs"] = segments_with_attrs
        context_result["tagged_segments"] = [seg["html"] for seg in segments_with_attrs]

        # --- 6. ML-driven DOM pattern clustering and tagging ---
        pattern_kb = load_pattern_kb()
        model = SentenceTransformer("all-MiniLM-L6-v2")
        pattern_matches = []
        segments_needing_review = []

        for seg in segments_with_attrs:
            label, confidence, pattern_id = ml_classify_segment(seg, model, pattern_kb)
            seg["ml_label"] = label
            seg["ml_confidence"] = confidence
            seg["pattern_id"] = pattern_id
            if confidence < 0.7 or label == "unknown":
                # Active learning: prompt user for feedback
                user_label = prompt_for_segment_label(seg)
                seg["ml_label"] = user_label
                seg["ml_confidence"] = 1.0
                seg["pattern_id"] = f"pattern_{hashlib.sha256(seg['html'].encode('utf-8')).hexdigest()[:10]}"
                # Save to KB and feedback log
                emb = get_segment_embedding(model, seg).tolist()
                kb_entry = {
                    "pattern_id": seg["pattern_id"],
                    "label": user_label,
                    "embedding": emb,
                    "example_html": seg["html"][:500],
                    "source_url": page.url,
                    "timestamp": time.time(),
                }
                append_pattern_kb(kb_entry)
                append_feedback_log({
                    "pattern_id": seg["pattern_id"],
                    "label": user_label,
                    "html": seg["html"][:500],
                    "source_url": page.url,
                    "timestamp": time.time(),
                })
                segments_needing_review.append(seg)
            else:
                pattern_matches.append({
                    "pattern_id": pattern_id,
                    "label": label,
                    "confidence": confidence,
                    "segment_html": seg["html"][:200],
                })

        context_result["pattern_kb_matches"] = pattern_matches
        context_result["segments_needing_review"] = segments_needing_review

        # --- 7. Dynamic tagging and context enrichment ---
        selector_log = set()
        for seg in segments_with_attrs:
            if seg["id"]:
                selector_log.add(f'#{seg["id"]}')
            for cls in seg["classes"]:
                selector_log.add(f'.{cls}')
            selector_log.add(seg["tag"].lower())
            # Add semantic tags for downstream use
            if "semantic_tags" not in seg:
                seg["semantic_tags"] = []
            if seg["ml_label"] not in ("unknown", "ignore"):
                seg["semantic_tags"].append(seg["ml_label"])
        context_result["selector_log"] = sorted(selector_log)

        # Add metadata for context organizer
        context_result["metadata"].update({
            "source_url": page.url,
            "scrape_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pattern_kb_size": len(pattern_kb),
        })

        # --- 8. Debug output for development ---
        if debug:
            rprint("\n[orange][DEBUG] Extracted HTML segments with ML labels:[/orange]")
            for seg in segments_with_attrs:
                rprint(f"{seg['tag']} {seg['attrs']} [label={seg['ml_label']}, conf={seg['ml_confidence']:.2f}] {seg['html'][:80]}{'...' if len(seg['html']) > 80 else ''}")
            if supported_links:
                rprint("\n[orange][DEBUG] Detected download links:[/orange]")
                for link in supported_links:
                    file_name = os.path.basename(link["href"])
                    rprint(f"[green]  - {file_name} ({link['format']})[/green]")
            if segments_needing_review:
                rprint(f"\n[red][DEBUG] {len(segments_needing_review)} segments flagged for review.[/red]")

    except Exception as e:
        rprint(f"[SCAN ERROR] HTML parsing failed: {e}")
        logger.error(f"[SCAN ERROR] HTML parsing failed: {e}")
        context_result["error"] = f"[SCAN ERROR] HTML parsing failed: {e}"
        pass

    logger.debug(f"Available formats detected: {context_result['available_formats']}")
    if context_cache is not None:
        context_cache[page_hash] = context_result
    return context_result

from __future__ import annotations

# webapp/parser/utils/spacy_utils.py
# -----------------------------------------------------------------------------------
# Advanced spaCy NLP utilities for election data integrity, context validation, and interference mitigation.
# -----------------------------------------------------------------------------------
"""
spacy_utils.py

Advanced spaCy NLP utilities for election data integrity, context validation, and interference mitigation.
"""
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

import orjson
spacy = None  # lazy import to avoid thinc->torch chain at module import

from ..Context_Integration.Context_Library.constants import KNOWN_STATE_TO_COUNTY_MAP
from .logger_singleton import logger
from .shared_logic import safe_get, safe_lower

def _get_nlp():
    """
    Lazy initializer for spaCy NLP model. Returns None if unavailable.
    Avoids importing torch via thinc in environments without DLLs.
    """
    global spacy
    try:
        if spacy is None:
            import spacy as _spacy
            spacy = _spacy
        # Attempt to load lightweight English model; fail gracefully
        return spacy.load("en_core_web_sm")
    except Exception as e:
        logger.warning(f"spaCy unavailable or model load failed: {e}")
        return None

# --- Core NLP Utilities ---

def extract_entities(text: str) -> List[Tuple[str, str]]:
    """
    Extract named entities from text using spaCy, with error handling.
    Returns a list of (entity_text, entity_label) tuples.
    """
    if not isinstance(text, str) or not text.strip():
        logger.error(f"[extract_entities] Invalid or empty text input: {repr(text)}")
        return []
    nlp = _get_nlp()
    if nlp is None:
        return []
    try:
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
    except Exception as e:
        logger.error(f"[extract_entities] spaCy failed on input: {repr(text)[:80]} | Error: {e}")
        return []

def get_sentences(text: str) -> List[str]:
    nlp = _get_nlp()
    if nlp is None:
        return []
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def clean_text(text: str) -> str:
    return " ".join(text.lower().strip().split())

def extract_entities_from_list(texts: List[str]) -> List[List[Tuple[str, str]]]:
    return [extract_entities(t) for t in texts]

def extract_entity_labels(text: str) -> Set[str]:
    nlp = _get_nlp()
    if nlp is None:
        return set()
    doc = nlp(text)
    return set(ent.label_ for ent in doc.ents)

def is_location_entity(ent_label: str) -> bool:
    return ent_label in {"GPE", "LOC", "FAC"}

def extract_locations(text: str) -> List[str]:
    nlp = _get_nlp()
    if nlp is None:
        return []
    doc = nlp(text)
    return [ent.text for ent in doc.ents if is_location_entity(ent.label_)]

def extract_dates(text: str) -> List[str]:
    nlp = _get_nlp()
    if nlp is None:
        return []
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "DATE"]

def filter_entities_by_type(text: str, types: List[str]) -> List[str]:
    nlp = _get_nlp()
    if nlp is None:
        return []
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in types]

def entity_frequency(texts: List[str], entity_type: List[str] = None, top_n: int = 10) -> Dict[str, int]:
    counter = Counter()
    for text in texts:
        nlp = _get_nlp()
        if nlp is None:
            continue
        doc = nlp(text)
        for ent in doc.ents:
            if entity_type is None or ent.label_ in entity_type:
                counter[ent.text] += 1
    return dict(counter.most_common(top_n))

def get_entity_context(text: str, entity: str, window: int = 30) -> List[str]:
    contexts = []
    idx = text.lower().find(entity.lower())
    while idx != -1:
        start = max(0, idx - window)
        end = min(len(text), idx + len(entity) + window)
        contexts.append(text[start:end])
        idx = text.lower().find(entity.lower(), idx + 1)
    return contexts

def similarity_score(text1: str, text2: str) -> float:
    nlp = _get_nlp()
    if nlp is None:
        return 0.0
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    if doc1.vector_norm and doc2.vector_norm:
        return doc1.similarity(doc2)
    return 0.0

def extract_persons(text: str) -> List[str]:
    nlp = _get_nlp()
    if nlp is None:
        return []
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

def extract_organizations(text: str) -> List[str]:
    nlp = _get_nlp()
    if nlp is None:
        return []
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "ORG"]

def extract_money(text: str) -> List[str]:
    nlp = _get_nlp()
    if nlp is None:
        return []
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "MONEY"]

def extract_emails(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)

def extract_urls(text: str) -> List[str]:
    url_pattern = r"https?://[^\s]+"
    return re.findall(url_pattern, text)

# --- Election-Specific Integrity & Validation Utilities ---

def load_known_states_counties() -> Tuple[Set[str], Set[str]]:
    """
    Loads known states and counties from the canonical mapping in constants.py.
    Returns (states_set, counties_set).
    """
    states = set(KNOWN_STATE_TO_COUNTY_MAP.keys())
    counties = set()
    for county_list in KNOWN_STATE_TO_COUNTY_MAP.values():
        counties.update(c.lower() for c in county_list)
    return set(s.lower() for s in states), counties

def normalize_location(name: str) -> str:
    """
    Normalize state/county names for comparison (lowercase, strip, remove 'county').
    """
    name = name.lower().replace("county", "").strip()
    name = re.sub(r"\s+", " ", name)
    return name

def is_known_state(state: str, known_states: Set[str]) -> bool:
    return normalize_location(state) in known_states

def is_known_county(county: str, known_counties: Set[str]) -> bool:
    return normalize_location(county) in known_counties

def detect_noisy_or_ambiguous_entities(text: str, noisy_patterns: List[str] = None) -> List[str]:
    """
    Detects entities that match known noisy or ambiguous patterns.
    Returns a list of suspicious entity strings.
    """
    if noisy_patterns is None:
        noisy_patterns = [
            r"test", r"sample", r"unknown", r"n/a", r"tbd", r"lorem", r"ipsum"
        ]
    nlp = _get_nlp()
    if nlp is None:
        return []
    doc = nlp(text)
    noisy = []
    for ent in doc.ents:
        for pat in noisy_patterns:
            if re.search(pat, ent.text, re.IGNORECASE):
                noisy.append(ent.text)
    return noisy

def canonicalize_entity(entity: str) -> str:
    """
    Canonicalize entity names (e.g., remove extra whitespace, standardize case).
    """
    return re.sub(r"\s+", " ", entity.strip().title())

def validate_contest(title: str, known_states: Set[str], known_counties: Set[str]) -> Dict[str, Any]:
    """
    Validates a contest title for integrity:
    - Checks for known state/county presence
    - Flags noisy/ambiguous entities
    - Returns a dict with flags and extracted info
    """
    entities = extract_entities(title)
    locations = extract_locations(title)
    dates = extract_dates(title)
    persons = extract_persons(title)
    orgs = extract_organizations(title)
    noisy = detect_noisy_or_ambiguous_entities(title)
    state_found = any(is_known_state(loc, known_states) for loc in locations)
    county_found = any(is_known_county(loc, known_counties) for loc in locations)
    return {
        "entities": entities,
        "locations": locations,
        "dates": dates,
        "persons": persons,
        "organizations": orgs,
        "noisy_entities": noisy,
        "state_found": state_found,
        "county_found": county_found,
        "valid": state_found and county_found and not noisy
    }

def flag_suspicious_contests(contests, context_library_path=None):
    """
    Flags contests with suspicious or ambiguous titles/entities.
    Optionally uses a context library if context_library_path is provided.
    Returns a list of flagged contest dicts with reasons.
    """
    context_library = None
    if context_library_path:
        # Load and use the context library as needed
        if os.path.exists(context_library_path):
            try:
                with open(context_library_path, "rb") as f:
                    context_library = orjson.loads(f.read())
            except Exception as e:
                logger.error(f"[flag_suspicious_contests] Failed to load context library: {e}")
                context_library = None

    known_states, known_counties = load_known_states_counties()
    flagged = []
    for c in contests:
        title = safe_get(c, "title", "")
        # Use context library for additional validation if available
        context_info = {}
        if context_library and isinstance(context_library, dict):
            # Try to match contest title to context library entries (case-insensitive)
            match_key = next(
                (k for k in context_library.keys() if safe_lower(k) == safe_lower(title)),
                None
            )
            if match_key:
                context_info = context_library.get(match_key, {})
        result = validate_contest(title, known_states, known_counties)
        # Enhance result with context library info if available
        if context_info:
            result["context_info"] = context_info
            # Optionally, flag if context library marks as suspicious or ambiguous
            if safe_get(context_info, "suspicious", False):
                result["valid"] = False
                result.setdefault("noisy_entities", []).append("context_library_flagged")
        if not result["valid"]:
            flagged.append({
                "title": title,
                "reasons": {
                    "no_state": not result.get("state_found", False),
                    "no_county": not result.get("county_found", False),
                    "noisy_entities": result.get("noisy_entities", [])
                },
                "entities": result.get("entities", []),
                "locations": result.get("locations", []),
                "context_info": result.get("context_info", {})
            })
    return flagged

def demo_analysis(text: str):
    logger.info("Entities:", extract_entities(text))
    logger.info("Sentences:", get_sentences(text))
    logger.info("Locations:", extract_locations(text))
    logger.info("Dates:", extract_dates(text))
    logger.info("Persons:", extract_persons(text))
    logger.info("Organizations:", extract_organizations(text))
    logger.info("Money:", extract_money(text))
    logger.info("Emails:", extract_emails(text))
    logger.info("URLs:", extract_urls(text))
    logger.info("Entity frequency:", entity_frequency([text]))
    logger.info("Similarity (sample vs itself):", similarity_score(text, text))
    # Election integrity check example
    known_states, known_counties = load_known_states_counties()
    logger.info("Contest validation:", validate_contest(text, known_states, known_counties))

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        sample = sys.argv[1]
        demo_analysis(sample)
    else:
        logger.info("Usage: python spacy_utils.py 'your sample text here'")
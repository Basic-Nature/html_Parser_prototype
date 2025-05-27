"""
spacy_utils.py

Advanced spaCy NLP utilities for election data integrity, context validation, and interference mitigation.
"""

import spacy
from collections import Counter
from typing import List, Tuple, Dict, Any, Set, Optional
import re
import os
import json

# Load spaCy model globally for efficiency, auto-download if missing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

# --- Core NLP Utilities ---

def extract_entities(text: str) -> List[Tuple[str, str]]:
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def get_sentences(text: str) -> List[str]:
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def clean_text(text: str) -> str:
    return " ".join(text.lower().strip().split())

def extract_entities_from_list(texts: List[str]) -> List[List[Tuple[str, str]]]:
    return [extract_entities(t) for t in texts]

def extract_entity_labels(text: str) -> Set[str]:
    doc = nlp(text)
    return set(ent.label_ for ent in doc.ents)

def is_location_entity(ent_label: str) -> bool:
    return ent_label in {"GPE", "LOC", "FAC"}

def extract_locations(text: str) -> List[str]:
    doc = nlp(text)
    return [ent.text for ent in doc.ents if is_location_entity(ent.label_)]

def extract_dates(text: str) -> List[str]:
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "DATE"]

def filter_entities_by_type(text: str, types: List[str]) -> List[str]:
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in types]

def entity_frequency(texts: List[str], entity_types: List[str] = None, top_n: int = 10) -> Dict[str, int]:
    counter = Counter()
    for text in texts:
        doc = nlp(text)
        for ent in doc.ents:
            if entity_types is None or ent.label_ in entity_types:
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
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    if doc1.vector_norm and doc2.vector_norm:
        return doc1.similarity(doc2)
    return 0.0

def extract_persons(text: str) -> List[str]:
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

def extract_organizations(text: str) -> List[str]:
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "ORG"]

def extract_money(text: str) -> List[str]:
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "MONEY"]

def extract_emails(text: str) -> List[str]:
    import re
    return re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)

def extract_urls(text: str) -> List[str]:
    import re
    url_pattern = r"https?://[^\s]+"
    return re.findall(url_pattern, text)

# --- Election-Specific Integrity & Validation Utilities ---

def load_known_states_counties(context_library_path: Optional[str] = None) -> Tuple[Set[str], Set[str]]:
    """
    Loads known states and counties from context_library.json if available.
    Returns (states_set, counties_set).
    """
    if context_library_path is None:
        context_library_path = os.path.join(
            os.path.dirname(__file__), "context_library.json"
        )
    states, counties = set(), set()
    if os.path.exists(context_library_path):
        with open(context_library_path, "r", encoding="utf-8") as f:
            lib = json.load(f)
        states = set(s.lower() for s in lib.get("known_states", []))
        counties = set(c.lower() for c in lib.get("known_counties", []))
    return states, counties

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

def validate_contest_title(title: str, known_states: Set[str], known_counties: Set[str]) -> Dict[str, Any]:
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
    Returns a list of flagged contest dicts with reasons.
    """
    from spacy_utils import load_known_states_counties, validate_contest_title

    known_states, known_counties = load_known_states_counties(context_library_path)
    flagged = []
    for c in contests:
        title = c.get("title", "")
        result = validate_contest_title(title, known_states, known_counties)
        if not result["valid"]:
            flagged.append({
                "title": title,
                "reasons": {
                    "no_state": not result["state_found"],
                    "no_county": not result["county_found"],
                    "noisy_entities": result["noisy_entities"]
                },
                "entities": result["entities"],
                "locations": result["locations"]
            })
    return flagged

def demo_analysis(text: str):
    print("Entities:", extract_entities(text))
    print("Sentences:", get_sentences(text))
    print("Locations:", extract_locations(text))
    print("Dates:", extract_dates(text))
    print("Persons:", extract_persons(text))
    print("Organizations:", extract_organizations(text))
    print("Money:", extract_money(text))
    print("Emails:", extract_emails(text))
    print("URLs:", extract_urls(text))
    print("Entity frequency:", entity_frequency([text]))
    print("Similarity (sample vs itself):", similarity_score(text, text))
    # Election integrity check example
    known_states, known_counties = load_known_states_counties()
    print("Contest validation:", validate_contest_title(text, known_states, known_counties))

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        sample = sys.argv[1]
        demo_analysis(sample)
    else:
        print("Usage: python spacy_utils.py 'your sample text here'")
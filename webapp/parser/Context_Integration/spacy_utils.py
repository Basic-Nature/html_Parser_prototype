"""
spacy_utils.py

Utility functions for spaCy NLP tasks, including NER, sentence segmentation, text cleaning,
entity type filtering, entity frequency analysis, and similarity scoring.
"""

import spacy
from collections import Counter
from typing import List, Tuple, Dict, Any, Set

# Load spaCy model globally for efficiency, auto-download if missing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

def extract_entities(text: str) -> List[Tuple[str, str]]:
    """
    Extract named entities from text using spaCy.
    Returns a list of (entity_text, entity_label) tuples.
    """
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def get_sentences(text: str) -> List[str]:
    """
    Split text into sentences using spaCy.
    Returns a list of sentence strings.
    """
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def clean_text(text: str) -> str:
    """
    Basic text cleaning: lowercasing, stripping, removing extra whitespace.
    """
    return " ".join(text.lower().strip().split())

def extract_entities_from_list(texts: List[str]) -> List[List[Tuple[str, str]]]:
    """
    Extract entities from a list of texts.
    Returns a list of lists of (entity_text, entity_label) tuples.
    """
    return [extract_entities(t) for t in texts]

def extract_entity_labels(text: str) -> Set[str]:
    """
    Extract only the entity labels from text.
    Returns a set of entity labels found in the text.
    """
    doc = nlp(text)
    return set(ent.label_ for ent in doc.ents)

def is_location_entity(ent_label: str) -> bool:
    """
    Check if an entity label is a location type (GPE, LOC, FAC).
    """
    return ent_label in {"GPE", "LOC", "FAC"}

def extract_locations(text: str) -> List[str]:
    """
    Extract location entities from text.
    Returns a list of location strings.
    """
    doc = nlp(text)
    return [ent.text for ent in doc.ents if is_location_entity(ent.label_)]

def extract_dates(text: str) -> List[str]:
    """
    Extract date entities from text.
    Returns a list of date strings.
    """
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "DATE"]

def filter_entities_by_type(text: str, types: List[str]) -> List[str]:
    """
    Extract entities of specific types from text.
    Returns a list of entity strings.
    """
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in types]

def entity_frequency(texts: List[str], entity_types: List[str] = None, top_n: int = 10) -> Dict[str, int]:
    """
    Count the frequency of entities (optionally filtered by type) across a list of texts.
    Returns a dict of entity_text -> count.
    """
    counter = Counter()
    for text in texts:
        doc = nlp(text)
        for ent in doc.ents:
            if entity_types is None or ent.label_ in entity_types:
                counter[ent.text] += 1
    return dict(counter.most_common(top_n))

def get_entity_context(text: str, entity: str, window: int = 30) -> List[str]:
    """
    Get the context (window of characters) around each occurrence of an entity in the text.
    Returns a list of context strings.
    """
    contexts = []
    idx = text.lower().find(entity.lower())
    while idx != -1:
        start = max(0, idx - window)
        end = min(len(text), idx + len(entity) + window)
        contexts.append(text[start:end])
        idx = text.lower().find(entity.lower(), idx + 1)
    return contexts

def similarity_score(text1: str, text2: str) -> float:
    """
    Compute the cosine similarity between two texts using spaCy embeddings.
    Returns a float between 0 and 1.
    """
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    # spaCy vectors may be zero if not present in small models
    if doc1.vector_norm and doc2.vector_norm:
        return doc1.similarity(doc2)
    return 0.0

def extract_persons(text: str) -> List[str]:
    """
    Extract person entities from text.
    Returns a list of person names.
    """
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

def extract_organizations(text: str) -> List[str]:
    """
    Extract organization entities from text.
    Returns a list of organization names.
    """
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "ORG"]

def extract_money(text: str) -> List[str]:
    """
    Extract monetary values from text.
    Returns a list of money strings.
    """
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "MONEY"]

def extract_emails(text: str) -> List[str]:
    """
    Extract email addresses using spaCy's matcher.
    Returns a list of email strings.
    """
    import re
    return re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)

def extract_urls(text: str) -> List[str]:
    """
    Extract URLs using regex.
    Returns a list of URL strings.
    """
    import re
    url_pattern = r"https?://[^\s]+"
    return re.findall(url_pattern, text)

# Example usage (for testing)
if __name__ == "__main__":
    sample = "The 2024 election in New York was held on November 5th. Contact info: john.doe@example.com. Visit https://nyvotes.gov for details. Joe Biden and the DNC raised $1,000,000."
    print("Entities:", extract_entities(sample))
    print("Sentences:", get_sentences(sample))
    print("Locations:", extract_locations(sample))
    print("Dates:", extract_dates(sample))
    print("Persons:", extract_persons(sample))
    print("Organizations:", extract_organizations(sample))
    print("Money:", extract_money(sample))
    print("Emails:", extract_emails(sample))
    print("URLs:", extract_urls(sample))
    print("Entity frequency:", entity_frequency([sample]))
    print("Similarity (sample vs itself):", similarity_score(sample, sample))
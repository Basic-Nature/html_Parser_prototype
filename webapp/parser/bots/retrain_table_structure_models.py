import os
import orjson
import re
import datetime
import hashlib
import subprocess
import time
import shutil
import gc
import sys
import random
from typing import List, Dict, Any, Optional, Set, Tuple
from ..utils.model_registry import ModelRegistry
from collections import Counter
from sentence_transformers import InputExample, losses
from torch.utils.data import DataLoader
from ..bots.librarian import load_context_library
from ..utils.db_utils import _safe_db_path, get_session, create_engine
from ..utils.shared_logger import log_info, log_error, log_warning, log_debug, RichConsoleProxy
from ..config import CONTEXT_DB_PATH, MODEL_DIR, PROJECT_ROOT, POSTGRES_URL, LOG_DIR
import numpy as np
import spacy
from spacy.training import Example, offsets_to_biluo_tags
from spacy.lookups import Lookups
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import tqdm
import torch
from sqlalchemy import select, inspect
from ..utils.models import TableStructure, Base

console = RichConsoleProxy()

# --- Logging Setup ---

# --- Advanced Entity Models (see models.py for full implementation) ---
# See previous answer for SQLAlchemy models: Party, State, County, District, Office, Candidate, Contest, Result, etc.

# --- Utility: get_or_create for advanced schema ---
def get_or_create(session, model, defaults=None, **kwargs):
    instance = session.query(model).filter_by(**kwargs).first()
    if instance:
        return instance
    else:
        params = dict((k, v) for k, v in kwargs.items())
        params.update(defaults or {})
        instance = model(**params)
        session.add(instance)
        session.commit()
        return instance

# --- Entity Extraction and Normalization ---
def normalize_entity(value: str) -> str:
    if not value or not isinstance(value, str):
        return ""
    return value.strip().title()

def normalize_entity_list(entity_list: List[str]) -> List[str]:
    return sorted(set(normalize_entity(e) for e in entity_list if e and isinstance(e, str)))

# --- Advanced DB Update Function ---
def update_advanced_entities(parsed_data: List[Dict[str, Any]], db_path: str):
    """
    parsed_data: list of dicts with keys: candidate, party, contest, office, votes, percent, etc.
    """
    from ..utils.models import Party, State, County, District, Office, Candidate, Contest, Result
    with get_session() as session:
        for row in parsed_data:
            try:
                party = get_or_create(session, Party, name=normalize_entity(row.get("party", "")))
                state = get_or_create(session, State, name=normalize_entity(row.get("state", "")))
                county = get_or_create(session, County, name=normalize_entity(row.get("county", "")), state=state)
                office = get_or_create(session, Office, name=normalize_entity(row.get("office", "")))
                district = get_or_create(session, District, name=normalize_entity(row.get("district", "")), state=state)
                candidate = get_or_create(session, Candidate, name=normalize_entity(row.get("candidate", "")), party=party, district=district, office=office)
                contest = get_or_create(session, Contest, title=normalize_entity(row.get("contest", "")), year=row.get("year"), state=state, county=county, district=district, office=office)
                result = get_or_create(
                    session, Result,
                    candidate=candidate, contest=contest,
                    votes=row.get("votes"), percent=row.get("percent"),
                    is_winner=row.get("is_winner", False), is_incumbent=row.get("is_incumbent", False),
                    vote_method=row.get("vote_method")
                )
                console.panel(f"Upserted result for candidate {candidate.name} in contest {contest.title}")
            except Exception as e:
                console.table(f"Failed to upsert entity row: {row} ({e})")
        session.commit()
    console.log("Advanced entity DB update complete.")

ELECTION_ENTITY_LABELS = [
    "CONTEST", "CANDIDATE", "PARTY", "COUNTY", "STATE", "DISTRICT", "VOTE_METHOD",
    "BALLOT_TYPES", "PRECINCT", "TOTAL", "PERCENT", "YEAR", "ELECTION_TYPES", "OFFICE", "MISC",
    "BALLOT_MEASURE", "LOCATION", "DATE", "INCUMBENT", "WINNER", "LOSER", "WRITE_IN", "UNOPPOSED", "PROPOSITION", 
    "AMENDMENT", "DISTRICT_TYPES", "JURISDICTION", "ELECTION_OFFICIAL", "RESULTS", "VOTE_COUNT", "AFFIDAVIT", "OTHER"   
]

ENTITY_PATTERNS = [
    (r"\b(19|20)\d{2}\b", "YEAR"),
    (r"\b(?:president|senate|governor|mayor|school board|proposition|referendum|assembly|council|trustee|justice|clerk)\b", "OFFICE"),
    (r"\b(democratic|republican|libertarian|green|independent|conservative|working families|write-in|other)\b", "PARTY"),
    (r"\b(absentee|early voting|mail|provisional|affidavit|other|void)\b", "VOTE_METHOD"),
    (r"\b(precinct|ward|district|area|city|municipal|location)\b", "PRECINCT"),
    (r"\btotal|sum|votes|overall|all\b", "TOTAL"),
    (r"\bpercent\b|\b% precincts reporting\b|\b% reporting\b|\bpercent reporting\b", "PERCENT"),
    (r"\bcounty\b", "COUNTY"),
    (r"\bstate\b", "STATE"),
    (r"\bgeneral|primary|special\b", "ELECTION_TYPES"),
    (r"\b(overvote|undervote|scattering|write-in|blank|spoiled)\b", "MISC"),
    (r"\b(proposition|amendment|measure|referendum|initiative)\b", "BALLOT_MEASURE"),
    (r"\b(city|town|village|borough|municipality|community|district)\b", "LOCATION"),
    (r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b \d{1,2}, \d{4}", "DATE"),
    (r"\bincumbent\b", "INCUMBENT"),
    (r"\bwinner\b", "WINNER"),
    (r"\bloser\b", "LOSER"),
    (r"\bwrite[- ]?in\b", "WRITE_IN"),
    (r"\bunopposed\b", "UNOPPOSED"),  
    (r"\bproposition \d+\b", "PROPOSITION"),
    (r"\bamendment \d+\b", "AMENDMENT"),
    (r"\b(jurisdiction|authority|agency|department)\b", "JURISDICTION"),
    (r"\belection official\b", "ELECTION_OFFICIAL"),
    (r"\b(results|outcome|tally|count)\b", "RESULTS"),
    (r"\b(vote count|vote total|vote tally)\b", "VOTE_COUNT"),
    (r"\b(?:election|vote|poll|referendum|plebiscite)\b", "ELECTION_TYPES"),  
    (r"\b(?:candidate|nominee|aspirant|hopeful)\b", "CANDIDATE"),
    (r"\b(?:election official|poll worker|election judge|inspector)\b", "ELECTION_OFFICIAL"),
    
    # Add more as needed
]

MISALIGNED_PATTERNS = [
    r"^Totals - ",  # Exclude any header starting with 'Totals - '
    # Add more patterns as needed
]

def is_misaligned_text(text):
    for pat in MISALIGNED_PATTERNS:
        if re.match(pat, text):
            return True
    return False

def clean_misaligned_ner_jsonl(jsonl_path, extra_patterns=None):
    """
    Remove misaligned NER examples from a JSONL file based on patterns and alignment check.
    Keeps only valid, aligned examples.
    """
    nlp = spacy.blank("en")
    patterns = MISALIGNED_PATTERNS.copy()
    if extra_patterns:
        patterns.extend(extra_patterns)
    def is_misaligned(text):
        for pat in patterns:
            if re.match(pat, text):
                return True
        return False

    cleaned = []
    misaligned = []
    if not os.path.exists(jsonl_path):
        return
    with open(jsonl_path, "rb") as f:
        for line in f:
            obj = orjson.loads(line)
            text = obj.get("text", "")
            entities = obj.get("entities", [])
            # Pattern-based skip
            if is_misaligned(text):
                misaligned.append(obj)
                continue
            # Alignment check
            try:
                tags = offsets_to_biluo_tags(nlp.make_doc(text), entities)
                if "-" in tags:
                    misaligned.append(obj)
                    continue
            except Exception:
                misaligned.append(obj)
                continue
            cleaned.append(obj)
    with open(jsonl_path, "wb") as f:
        for obj in cleaned:
            f.write(orjson.dumps(obj, option=orjson.OPT_APPEND_NEWLINE))
    if misaligned:
        misaligned_path = jsonl_path.replace(".jsonl", "_misaligned.jsonl")
        with open(misaligned_path, "wb") as f:
            for obj in misaligned:
                f.write(orjson.dumps(obj, option=orjson.OPT_APPEND_NEWLINE))
        log_info(f"[CLEAN] Removed {len(misaligned)} misaligned NER examples. Saved to {misaligned_path}")
    log_info(f"[CLEAN] Cleaned NER training data saved to {jsonl_path}. Remaining: {len(cleaned)}")

def safe_model_save(model, model_save_path, retries=3):

    for attempt in range(1, retries+1):
        try:
            gc.collect()
            model.save(model_save_path)
            log_info(f"[INFO] Model saved successfully on attempt {attempt}.")
            return
        except Exception as e:
            log_warning(f"[WARN] Model save failed (attempt {attempt}): {e}")
            time.sleep(2 * attempt)
            gc.collect()
    # Try saving to a temp dir and moving
    tmp_path = model_save_path + "_tmp"
    try:
        gc.collect()
        model.save(tmp_path)
        shutil.rmtree(model_save_path, ignore_errors=True)
        shutil.move(tmp_path, model_save_path)
        log_info(f"[INFO] Model saved via temp path workaround.")
    except Exception as e:
        log_error(f"[ERROR] Final model save failed: {e}\nIf you see repeated save failures, close any file explorers or editors viewing the model directory.")

def append_training_data(new_data, path="spacy_ner_train_data.jsonl"):
    """
    Appends new training data to a JSONL file in the log directory, deduplicating by text/entities,
    and adds a timestamp to each entry.
    """
    log_dir = LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    safe_path = os.path.abspath(os.path.join(log_dir, path))
    if not safe_path.startswith(log_dir):
        raise ValueError("Unsafe path detected for training data output!")
    existing = set()
    if os.path.exists(safe_path):
        with open(safe_path, "rb") as f:
            for line in f:
                existing.add(line.strip())
    with open(safe_path, "ab") as f:
        for text, annots in new_data:
            entry = {
                "text": text,
                "entities": annots["entities"],
                "timestamp": datetime.datetime.now().isoformat()
            }
            line = orjson.dumps(entry, option=orjson.OPT_APPEND_NEWLINE)
            if line.strip() not in existing:
                f.write(line)

def save_training_data_jsonl(train_data, path="spacy_ner_train_data.jsonl"):
    log_dir = LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    filename = os.path.basename(path)
    filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', filename)
    safe_path = os.path.join(log_dir, filename)
    if not os.path.abspath(safe_path).startswith(log_dir):
        raise ValueError("Unsafe path detected for training data output!")
    with open(safe_path, "wb") as f:
        for text, annots in train_data:
            f.write(orjson.dumps({"text": text, "entities": annots["entities"]}, option=orjson.OPT_APPEND_NEWLINE))
    log_info(f"Saved spaCy NER training data to {safe_path}")

def cluster_container_patterns(log_dir=None, n_clusters=5):
    """
    Cluster container HTML snippets and metadata for ML/NLP training.
    Prints cluster assignments and common selectors/classes/headings.
    """

    if log_dir is None:
        log_dir = LOG_DIR

    htmls = []
    meta = []
    for path in glob.glob(os.path.join(log_dir, "failed_container_*.json")):
        with open(path, "rb") as f:
            entry = orjson.loads(f.read())
            htmls.append(entry.get("html", ""))
            meta.append(entry)
    if not htmls:
        log_info("No failed containers to cluster.")
        return

    vectorizer = TfidfVectorizer(max_features=200, stop_words="english")
    X = vectorizer.fit_transform(htmls)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(X)
    clusters = [[] for _ in range(n_clusters)]
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(meta[i])

    for idx, group in enumerate(clusters):
        log_info(f"\n=== Cluster {idx+1} ({len(group)} containers) ===")
        selectors = [g.get("selector") for g in group]
        parent_classes = [g.get("parent_class") for g in group]
        headings = [g.get("heading") for g in group]
        log_info("  Common selectors:", Counter(selectors).most_common(3))
        log_info("  Common parent classes:", Counter(parent_classes).most_common(3))
        log_info("  Common headings:", Counter(headings).most_common(3))
        log_info("  Example HTML snippet:", group[0].get("html", "")[:200].replace("\n", " ") if group else "")

def auto_label_header(header: str, context: dict = None):
    labels = []
    for pattern, label in ENTITY_PATTERNS:
        for match in re.finditer(pattern, header, re.IGNORECASE):
            start, end = match.span()
            labels.append((start, end, label))
    if context:
        for label_type, values in [
            ("COUNTY", context.get("known_counties", [])),
            ("LOCATION", context.get("known_cities", [])),
            ("STATE", context.get("known_states", [])),
            ("CANDIDATE", context.get("known_candidates", [])),
            ("DISTRICT", context.get("known_districts", [])),
            
        ]:
            for val in values:
                for match in re.finditer(re.escape(val), header, re.IGNORECASE):
                    start, end = match.span()
                    labels.append((start, end, label_type))
    labels = sorted(set(labels), key=lambda x: (x[0], x[1]))
    return labels

def extract_candidates_from_context(context):
    candidates = set(context.get("known_candidates", []))
    for title in context.get("contests", []):
        if isinstance(title, dict):
            t = title.get("title", "")
        else:
            t = title
        for match in re.findall(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", t):
            candidates.add(match)
    return list(candidates)

def entity_frequency_analysis(train_data):
    counter = Counter()
    for _, annots in train_data:
        for _, _, label in annots["entities"]:
            counter[label] += 1
    log_info("Entity frequency:", counter)

def update_db_with_new_entities(new_entities, db_path):
    """
    Update the Entity table in PostgreSQL with new entities using SQLAlchemy.
    """
    from ..utils.models import Entity
    with get_session() as session:
        for entity_type, values in new_entities.items():
            for value in values:
                exists = session.execute(
                    select(Entity).where(Entity.entity_type == entity_type, Entity.value == value)
                ).scalar_one_or_none()
                if not exists:
                    session.add(Entity(entity_type=entity_type, value=value))
        session.commit()
    log_info(f"Updated DB with new entities: {{ { {k: len(v) for k,v in new_entities.items()} } }}")

def load_spacy_ner_examples(jsonl_path):
    """
    Loads extra NER training examples from a JSONL file.
    Each line should be: {"text": ..., "entities": [[start, end, label], ...]}
    """
    examples = []
    if not os.path.exists(jsonl_path):
        return examples
    with open(jsonl_path, "rb") as f:
        for line in f:
            obj = orjson.loads(line)
            text = obj["text"]
            entities = obj["entities"]
            examples.append((text, {"entities": entities}))
    return examples

# Label priority for deduplication: higher in the list = higher priority
LABEL_PRIORITY = [
    "CANDIDATE", "PARTY", "VOTE_METHOD", "PRECINCT", "DISTRICT", "COUNTY", "STATE", "TOTAL", "ELECTION_TYPES", "OFFICE", "WRITE_IN", "MISC", "BALLOT_MEASURE", "LOCATION", "DATE", "INCUMBENT", "WINNER", "LOSER", "UNOPPOSED", "PROPOSITION", "AMENDMENT", "DISTRICT_TYPES", "JURISDICTION", "ELECTION_OFFICIAL", "RESULTS", "VOTE_COUNT", "AFFIDAVIT", "OTHER"
]

def remove_overlapping_entities(entities):
    """
    Remove overlapping and duplicate-span entities from a list of (start, end, label) tuples.
    Keeps the longest span first, then next non-overlapping, and only one label per span (by priority).
    """
    def label_rank(label):
        try:
            return LABEL_PRIORITY.index(label)
        except ValueError:
            return len(LABEL_PRIORITY)
    # Sort by start, then by longest span (descending), then by label priority
    entities = sorted(entities, key=lambda x: (x[0], -(x[1] - x[0]), label_rank(x[2])))
    result = []
    last_end = -1
    seen_spans = set()
    for start, end, label in entities:
        if start >= last_end and (start, end) not in seen_spans:
            result.append((start, end, label))
            last_end = end
            seen_spans.add((start, end))
        # else: skip this entity because it overlaps or is a duplicate span
    return result

def validate_training_data(train_data, nlp, logged=None):
    """
    Validate and skip misaligned spaCy NER training examples to avoid [W030] warnings.
    Pre-check alignment before creating Example.
    """
    valid_data = []
    for text, annots in train_data:
        try:
            tags = offsets_to_biluo_tags(nlp.make_doc(text), annots["entities"])
            if "-" in tags:
                if logged:
                    log_warning(f"Skipping misaligned entity in: {text}")
                continue
            valid_data.append((text, annots))
        except Exception as e:
            if logged:
                log_warning(f"Error validating entity alignment: {e}")
    return valid_data

def retrain_spacy_ner_advanced(
    confirmed_structures, 
    context_library=None, 
    model_save_path="fine_tuned_spacy_ner",
    max_epochs=None,
    patience=3,
    min_delta=0.01,
    batch_size=32
):
    import importlib

    nlp = spacy.blank("en")
    # Try to use GPU if available
    if spacy.prefer_gpu():
        log_info("[INFO] spaCy using GPU for training.")
    else:
        log_info("[INFO] spaCy using CPU for training.")

    # --- Robust lexeme normalization loading ---
    try:
        lookups_mod = importlib.util.find_spec("spacy.lookups")
        if lookups_mod and hasattr(Lookups(), "add_table"):
            lookups = Lookups()
            if hasattr(spacy.lookups, "load_lookups_data"):
                lookups.add_table("lexeme_norm", spacy.lookups.load_lookups_data("en", tables=["lexeme_norm"]).get_table("lexeme_norm"))
                nlp.vocab.lookups = lookups
    except Exception as e:
        log_warning("[spaCy] Could not load lexeme normalization table. You may ignore this for English. Error:", e)

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    for label in ELECTION_ENTITY_LABELS:
        ner.add_label(label)

    # --- Data Preparation ---
    known_context = context_library or {}
    train_data = []
    all_candidates = set()
    all_parties = set()
    all_counties = set()
    all_states = set()
    all_districts = set()
    all_locations = set()
    
    # --- Load extra examples from JSONL file ---
    extra_examples = load_spacy_ner_examples(
        os.path.join(LOG_DIR, "spacy_ner_train_data.jsonl")
    )
    if extra_examples:
        log_info(f"Loaded {len(extra_examples)} extra NER examples from log/spacy_ner_train_data.jsonl")
    train_data.extend(extra_examples)

    # ...existing code for auto-labeling confirmed_structures...
    for struct in confirmed_structures:
        headers = struct["headers"]
        context = struct.get("context", {})
        context.update({
            "known_counties": known_context.get("known_counties", []),
            "known_cities": known_context.get("known_cities", []),
            "known_states": known_context.get("known_states", []),
            "known_candidates": known_context.get("known_candidates", []),
            "known_districts": known_context.get("known_districts", []),
        })
        context_candidates = extract_candidates_from_context(context)
        context["known_candidates"] = list(set(context.get("known_candidates", []) + context_candidates))
        all_candidates.update(context["known_candidates"])
        all_parties.update([p for p in re.findall(r"\\b(?:Democratic|Republican|Libertarian|Green|Independent|Conservative|Working Families|Write-in|Other)\\b", " ".join(headers), re.IGNORECASE)])
        all_counties.update(context.get("known_counties", []))
        all_states.update(context.get("known_states", []))
        all_districts.update(context.get("known_districts", []))
        all_locations.update(context.get("known_cities", []))
        
        for header in headers:
            if is_misaligned_text(header):
                continue  # Skip known misaligned patterns
            entities = auto_label_header(header, context)
            if entities:
                entities = remove_overlapping_entities(entities)
                train_data.append((header, {"entities": entities}))

    # Validate and skip misaligned entities, log skipped
    misaligned_count = 0
    misaligned_examples = []
    valid_data = []
    from spacy.training import offsets_to_biluo_tags
    for text, annots in train_data:
        try:
            tags = offsets_to_biluo_tags(nlp.make_doc(text), annots["entities"])
            if "-" in tags:
                misaligned_count += 1
                misaligned_examples.append({"text": text, "entities": annots["entities"]})
                continue
            valid_data.append((text, annots))
        except Exception as e:
            misaligned_count += 1
            misaligned_examples.append({"text": text, "entities": annots["entities"], "error": str(e)})
    if misaligned_examples:
        misaligned_path = os.path.join(LOG_DIR, "spacy_ner_misaligned.jsonl")
        with open(misaligned_path, "wb") as f:
            for ex in misaligned_examples:
                f.write(orjson.dumps(ex, option=orjson.OPT_APPEND_NEWLINE))
        log_warning(f"[NER] Skipped {misaligned_count} misaligned examples. Saved to {misaligned_path}")
    train_data = valid_data
    save_training_data_jsonl(train_data)
    entity_frequency_analysis(train_data)
    log_info(f"[NER] Used {len(train_data)} valid examples, skipped {misaligned_count} misaligned.")

    # --- Training ---
    examples = []
    for text, annots in train_data:
        doc = nlp.make_doc(text)
        annots["entities"] = remove_overlapping_entities(annots["entities"])
        example = Example.from_dict(doc, annots)
        examples.append(example)
    if not examples:
        log_warning("No NER training examples found. Skipping spaCy NER retraining.")
        return

    optimizer = nlp.begin_training()
    optimizer.learn_rate = 0.001

    # --- Dynamic Early Stopping and Adaptive min_delta ---
    epochs = max_epochs or int(os.getenv("SPACY_NER_EPOCHS", 10))
    patience = int(os.getenv("SPACY_NER_PATIENCE", 3))
    min_delta = float(os.getenv("SPACY_NER_MIN_DELTA", 0.01))
    batch_size = int(os.getenv("SPACY_NER_BATCH_SIZE", 32))
    no_improve = 0
    best_loss = float("inf")
    best_model_path = model_save_path + "_best"
    best_epoch = 0
    loss_history = []

    log_info(f"[INFO] Starting spaCy NER training for up to {epochs} epochs, batch size {batch_size}...")
    for i in range(epochs):
        losses = {}
        random.shuffle(examples)
        for batch in tqdm.tqdm([examples[j:j+batch_size] for j in range(0, len(examples), batch_size)], desc=f"spaCy NER epoch {i+1}"):
            nlp.update(batch, sgd=optimizer, drop=0.2, losses=losses)
        epoch_loss = losses.get("ner", 0)
        loss_history.append(epoch_loss)
        log_info(f"spaCy NER retraining epoch {i+1}, loss: {epoch_loss:.4f}")

        # --- Dynamic min_delta: scale with loss magnitude ---
        if i == 0 and min_delta < 1:
            # If user left min_delta at default, auto-scale for large loss
            min_delta = max(0.01, epoch_loss * 0.01)
            log_info(f"[AUTO] Adjusted min_delta to {min_delta:.2f} based on initial loss.")

        # --- Loss smoothing: use moving average over last 3 epochs ---
        if len(loss_history) > 3:
            smoothed_loss = np.mean(loss_history[-3:])
        else:
            smoothed_loss = epoch_loss

        # --- Save best model ---
        if smoothed_loss < best_loss - min_delta:
            best_loss = smoothed_loss
            no_improve = 0
            best_epoch = i + 1
            nlp.to_disk(best_model_path)
            log_info(f"[INFO] New best model saved at epoch {i+1} with smoothed loss {smoothed_loss:.2f}")
        else:
            no_improve += 1

        # --- Dynamic patience: extend if still improving fast ---
        if no_improve >= patience:
            log_info(f"[INFO] Early stopping at epoch {i+1} (no improvement for {patience} epochs).")
            break
        if i > 2 and smoothed_loss < 0.5 * loss_history[0] and patience < 8:
            patience += 1
            log_info(f"[AUTO] Increased patience to {patience} due to rapid improvement.")

    # Restore best model
    if os.path.exists(best_model_path):
        log_info(f"[INFO] Restoring best model from epoch {best_epoch} with loss {best_loss:.2f}")
        nlp = spacy.load(best_model_path)
        shutil.rmtree(best_model_path, ignore_errors=True)
    nlp.to_disk(model_save_path)
    log_info(f"Fine-tuned spaCy NER model saved to: {model_save_path}")

    # --- Training summary and suggestions ---
    log_info(f"[SUMMARY] Best loss: {best_loss:.2f} at epoch {best_epoch}")
    if best_epoch < epochs:
        log_warning(f"[SUGGESTION] Consider lowering min_delta or increasing patience if you want longer training.")
    elif best_epoch == epochs:
        log_warning(f"[SUGGESTION] Model improved until the last epoch. Consider increasing epochs for further improvement.")
    log_warning(f"[SUGGESTION] Next run: patience={patience}, min_delta={min_delta:.2f}, epochs={epochs}")
    def normalize_entity_list(entity_list):
        return sorted(set(e.strip().title() for e in entity_list if e and isinstance(e, str)))
    # Update DB with new entities
    new_entities = {
        "CANDIDATE": normalize_entity_list(all_candidates),
        "PARTY": normalize_entity_list(all_parties),
        "COUNTY": normalize_entity_list(all_counties),
        "STATE": normalize_entity_list(all_states),
        "DISTRICT": normalize_entity_list(all_districts),
        "LOCATION": normalize_entity_list(all_locations),
        "VOTE_METHOD": normalize_entity_list(context_library.get("known_vote_methods", [])),
        "BALLOT_MEASURE": normalize_entity_list(context_library.get("known_ballot_measures", [])),
        "ELECTION_TYPES": normalize_entity_list(context_library.get("known_election_types", [])),
        "YEAR": normalize_entity_list(context_library.get("known_years", [])),
        "OFFICE": normalize_entity_list(context_library.get("known_offices", [])),
        "ELECTION_OFFICIAL": normalize_entity_list(context_library.get("known_election_officials", [])),
        "RESULTS": normalize_entity_list(context_library.get("known_results", [])),
        "VOTE_COUNT": normalize_entity_list(context_library.get("known_vote_counts", [])),
        "TOTAL": normalize_entity_list(context_library.get("known_totals", [])),
        "PERCENT": normalize_entity_list(context_library.get("known_percents", [])),
        "MISC": normalize_entity_list(context_library.get("known_misc", [])),
    }
    update_db_with_new_entities(new_entities, _safe_db_path(CONTEXT_DB_PATH))
    log_info(f"[DB] Updated with entities: {{ { {k: len(v) for k, v in new_entities.items()} } }}")

def get_all_confirmed_structures():
    """
    Retrieve all confirmed table structures from PostgreSQL using SQLAlchemy.
    Returns a list of dicts, not ORM objects, to avoid DetachedInstanceError.
    """
    with get_session() as session:
        rows = session.execute(
            select(TableStructure).where(TableStructure.confirmed_by_user == True)
        ).scalars().all()
        # Extract all needed fields while session is open
        result = []
        for row in rows:
            result.append({
                "contest_title": row.contest_title,
                "headers": orjson.loads(row.headers) if isinstance(row.headers, (str, bytes, bytearray)) else row.headers,
                "context": orjson.loads(row.context) if isinstance(row.context, (str, bytes, bytearray)) else row.context,
                "original_structure": getattr(row, "original_structure", {}),
                "corrected_structure": getattr(row, "corrected_structure", {}),
                "sample_rows": getattr(row, "sample_rows", [{}]),
            })
        return result
    
def run_manual_correction_bot():
    """
    Run the manual correction bot robustly as a module, capturing output and errors.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "webapp.parser.bots.manual_correction_bot", "--fields", "tables", "--enhanced"],
            check=True,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
        )
        log_info(result.stdout)
        if result.stderr:
            log_info(result.stderr)
    except subprocess.CalledProcessError as e:
        log_error(f"[ERROR] Manual correction bot failed: {e.stderr}")

def retrain_sentence_transformer(confirmed_structures, model_save_path=None, epochs=1, batch_size=8):
    """
    Fine-tunes the SentenceTransformer model on confirmed structures.
    Loads the existing model for further training if present, otherwise starts from base.
    Always saves to the same folder (no timestamp).
    """
    try:
        del model
    except NameError:
        pass
    gc.collect()
    
    train_examples = []
    for struct in confirmed_structures:
        contest_title = struct.get("contest_title", "")
        headers = struct.get("headers", [])
        for header in headers:
            train_examples.append(InputExample(texts=[contest_title, header], label=1.0))
    if not train_examples:
        log_warning("No training examples found. Aborting retraining.")
        return

    base_dir = MODEL_DIR if 'MODEL_DIR' in globals() else os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../model"))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = model_save_path or os.path.join(base_dir, f"fine_tuned_table_headers_{timestamp}")
    os.makedirs(model_save_path, exist_ok=True)

    # Clean up old models
    fine_tuned_dirs = sorted(
        [d for d in os.listdir(base_dir) if d.startswith("fine_tuned_table_headers_") and os.path.isdir(os.path.join(base_dir, d))]
    )
    if len(fine_tuned_dirs) > 1:
        oldest = fine_tuned_dirs[0]
        oldest_path = os.path.join(base_dir, oldest)
        try:
            shutil.rmtree(oldest_path)
            log_info(f"[CLEANUP] Deleted oldest fine-tuned model directory: {oldest_path}")
        except Exception as e:
            log_warning(f"[WARN] Could not delete old model directory {oldest_path}: {e}")

    # Model loading
    model = None
    prev_model_dir = os.path.join(base_dir, "fine_tuned_table_headers")
    model_files = ["config.json", "pytorch_model.bin", "model.safetensors", "tf_model.h5", "model.ckpt.index", "flax_model.msgpack"]
    model_files_exist = any(os.path.exists(os.path.join(prev_model_dir, f)) for f in model_files)
    if model_files_exist:
        log_info(f"Attempting to load existing model from {prev_model_dir} for further fine-tuning...")
        try:
            model = ModelRegistry.get_sentence_transformer(model_name=prev_model_dir, use_finetuned=False)
        except Exception as e:
            log_warning(f"[WARN] Failed to load existing model: {e}")
            model = None
    if model is None:
        log_warning("Falling back to base model (all-MiniLM-L6-v2).")
        try:
            model = ModelRegistry.get_sentence_transformer(model_name="all-MiniLM-L6-v2", use_finetuned=False)
        except Exception as e:
            log_error(f"[ERROR] Could not load base SentenceTransformer: {e}")
            return

    # Defensive: clean up incomplete/corrupt model directory before saving
    for f in model_files:
        fpath = os.path.join(model_save_path, f)
        if os.path.exists(fpath) and os.path.getsize(fpath) == 0:
            log_info(f"[CLEANUP] Removing empty/corrupt file: {fpath}")
            try:
                os.remove(fpath)
            except Exception:
                pass

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model)
    log_info(f"Retraining SentenceTransformer on {len(train_examples)} pairs for {epochs} epoch(s)...")
    try:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=10,
            show_progress_bar=True
        )
    except Exception as e:
        log_error(f"[ERROR] Model training failed: {e}")
        return
    try:
        safe_model_save(model, model_save_path)
        log_info(f"Fine-tuned model saved to: {model_save_path}")
        canonical_dir = os.path.join(base_dir, "fine_tuned_table_headers")
        try:
            if os.path.exists(canonical_dir):
                shutil.rmtree(canonical_dir, ignore_errors=True)
            shutil.copytree(model_save_path, canonical_dir)
            log_info(f"[INFO] Copied new model to canonical directory: {canonical_dir}")
        except Exception as e:
            log_warning(f"[WARN] Could not update canonical model directory: {e}")
    except Exception as e:
        log_error(f"[ERROR] Model save failed: {e}")
        
def segment_hash(segment):
    """Generate a stable hash for a DOM segment based on tag, attrs, and first 200 chars of HTML."""
    tag = segment.get("tag", "")
    attrs = segment.get("attrs", {})
    html = segment.get("html", "")[:200]
    # Use OPT_SORT_KEYS for orjson
    return hashlib.sha256((tag + orjson.dumps(attrs, option=orjson.OPT_SORT_KEYS).decode() + html).encode("utf-8")).hexdigest()

def load_cached_segment_hashes(context_library):
    """Return a set of all segment_hashes in the context library."""
    return {seg.get("segment_hash") for seg in context_library.get("cached_segments", [])}

def scan_in_memory_ner_examples(train_data, verbose=False):
    """Scan a list of (text, annots) NER examples for misalignments using spaCy's offsets_to_biluo_tags."""
    import spacy
    from spacy.training import offsets_to_biluo_tags
    nlp = spacy.blank("en")
    misaligned = []
    for text, annots in train_data:
        try:
            tags = offsets_to_biluo_tags(nlp.make_doc(text), annots["entities"])
            if "-" in tags:
                misaligned.append((text, annots["entities"]))
                if verbose:
                    log_warning(f"MISALIGNED: {text} {annots['entities']}")
        except Exception as e:
            misaligned.append((text, annots["entities"]))
            if verbose:
                log_error(f"ERROR: {text} {annots['entities']} ({e})")
    return misaligned

def ensure_table_structures_exists():
    """
    Ensure the 'table_structures' table exists in the database.
    If not, create all tables defined in models.
    """
    engine = create_engine(POSTGRES_URL)
    inspector = inspect(engine)
    if 'table_structures' not in inspector.get_table_names():
        log_info("[INFO] 'table_structures' table not found. Creating all tables...")
        Base.metadata.create_all(engine)
        log_info("[INFO] All tables created.")
    else:
        log_info("[INFO] 'table_structures' table exists.")

def main():
    ensure_table_structures_exists()
    if os.getenv("REVIEW_WITH_MANUAL_BOT", "false").lower() == "true":
        run_manual_correction_bot()

    # --- Self-cleaner for NER training data ---
    ner_train_jsonl = os.path.join(LOG_DIR, "spacy_ner_train_data.jsonl")
    clean_misaligned_ner_jsonl(ner_train_jsonl)

    confirmed_structures = get_all_confirmed_structures()
    console.table(f"Found {len(confirmed_structures)} confirmed table structures.")

    # Log user feedback/corrections for ML
    feedback_log_path = os.path.join(LOG_DIR, "structure_feedback_log.jsonl")
    os.makedirs(os.path.dirname(feedback_log_path), exist_ok=True)
    for struct in confirmed_structures:
        old_structure_info = struct.get("original_structure", {})
        structure_info = struct.get("corrected_structure", {})
        headers = struct.get("headers", [])
        data = struct.get("sample_rows", [{}])
        with open(feedback_log_path, "ab") as f:
            f.write(orjson.dumps({
                "original_structure": old_structure_info,
                "corrected_structure": structure_info,
                "headers": headers,
                "sample_row": data[0] if data else {},
            }, option=orjson.OPT_APPEND_NEWLINE))
    context_library = load_context_library()
    log_debug("DEBUG: Loaded context library:", type(context_library))
    if not isinstance(context_library, dict):
        log_error("ERROR: Context library is not a dictionary. Check your context library loading logic.")
        raise ValueError("Context library must be a dictionary. Check your context library loading logic.")
    cached_hashes = load_cached_segment_hashes(context_library)
    deduped_train_data = []
    for struct in confirmed_structures:
        seg_hash = segment_hash(struct)
        if seg_hash not in cached_hashes:
            deduped_train_data.append(struct)
    console.table(f"Deduplicated to {len(deduped_train_data)} unique structures for training.")

    # Build NER training data (auto-label, dedupe, etc.)
    train_data = []
    all_candidates = set()
    all_parties = set()
    all_counties = set()
    all_states = set()
    all_districts = set()
    all_locations = set()
    extra_examples = load_spacy_ner_examples(
        os.path.join(LOG_DIR, "spacy_ner_train_data.jsonl")
    )
    if extra_examples:
        console.table(f"Loaded {len(extra_examples)} extra NER examples from log/spacy_ner_train_data.jsonl")
    train_data.extend(extra_examples)

    for struct in deduped_train_data:
        headers = struct["headers"]
        context = struct.get("context", {})
        context.update({
            "known_counties": context_library.get("known_counties", []),
            "known_cities": context_library.get("known_cities", []),
            "known_states": context_library.get("known_states", []),
            "known_candidates": context_library.get("known_candidates", []),
            "known_districts": context_library.get("known_districts", []),
        })
        context_candidates = extract_candidates_from_context(context)
        context["known_candidates"] = list(set(context.get("known_candidates", []) + context_candidates))
        all_candidates.update(context["known_candidates"])
        all_parties.update([p for p in re.findall(r"\b(?:Democratic|Republican|Libertarian|Green|Independent|Conservative|Working Families|Write-in|Other)\b", " ".join(headers), re.IGNORECASE)])
        all_counties.update(context.get("known_counties", []))
        all_states.update(context.get("known_states", []))
        all_districts.update(context.get("known_districts", []))
        all_locations.update(context.get("known_cities", []))
        for header in headers:
            if is_misaligned_text(header):
                continue
            entities = auto_label_header(header, context)
            if entities:
                entities = remove_overlapping_entities(entities)
                train_data.append((header, {"entities": entities}))

    # Scan in-memory NER examples for misalignments before retraining
    console.log("[INFO] Scanning in-memory NER training data for misalignments before retraining...")
    misaligned = scan_in_memory_ner_examples(train_data, verbose=True)
    if misaligned:
        console.panel(f"{len(misaligned)} misaligned NER examples found in final training data. Running diagnostics and launching manual_correction_bot. Aborting retraining.")
        misaligned_path = os.path.join(LOG_DIR, "spacy_ner_misaligned.jsonl")
        with open(misaligned_path, "wb") as f:
            for text, entities in misaligned:
                f.write(orjson.dumps({"text": text, "entities": entities}, option=orjson.OPT_APPEND_NEWLINE))
        try:
            subprocess.run([
                sys.executable, "-m", "webapp.parser.bots.scan_misaligned_ner", "--jsonl", misaligned_path
            ], check=True, cwd=PROJECT_ROOT, env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)})
        except Exception as e:
            console.table(f"scan_misaligned_ner diagnostics failed: {e}")
        run_manual_correction_bot()
        console.log("Please correct misalignments and rerun retraining.")
        sys.exit(2)

    # Use deduped_train_data for retraining
    retrain_sentence_transformer(
        deduped_train_data,
        epochs=int(os.getenv("SBERT_EPOCHS", 1)),
        batch_size=int(os.getenv("SBERT_BATCH_SIZE", 8))
    )
    retrain_spacy_ner_advanced(
        deduped_train_data,
        context_library,
        max_epochs=int(os.getenv("SPACY_NER_EPOCHS", 10)),
        patience=int(os.getenv("SPACY_NER_PATIENCE", 3)),
        min_delta=float(os.getenv("SPACY_NER_MIN_DELTA", 0.01)),
        batch_size=int(os.getenv("SPACY_NER_BATCH_SIZE", 32))
    )
    cluster_container_patterns()
    console.log("\n[SUMMARY] Table Structure Model Retraining Complete.")
    console.log("If you see repeated model save failures, close any file explorers or editors viewing the model directory.")
    console.log("If you see spaCy lexeme normalization warnings, you can ignore them for English. To suppress, install spacy-lookups-data and load the table if needed.")
    console.log("If you see spaCy entity alignment warnings, consider cleaning your training data or using the provided validation function.")
    gc.collect()

if __name__ == "__main__":
    main()
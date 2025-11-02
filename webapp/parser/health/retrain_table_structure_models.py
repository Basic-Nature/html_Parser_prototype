import copy
import datetime
import gc
import glob
import hashlib
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter
from importlib.util import find_spec
from types import ModuleType
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, runtime_checkable

import numpy as np
import orjson
import spacy
from sentence_transformers import InputExample, losses
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.language import Language
from spacy.lookups import Lookups
from spacy.training import Example, offsets_to_biluo_tags
from sqlalchemy import inspect, select
from torch.utils.data import DataLoader

from ..config import (
    CONTEXT_DB_PATH,
    LOG_DIR,
    MODEL_DIR,
    PROJECT_ROOT,
    REVIEW_WITH_MANUAL_BOT,
    SBERT_BATCH_SIZE,
    SBERT_EPOCHS,
    SPACY_NER_BATCH_SIZE,
    SPACY_NER_EPOCHS,
    SPACY_NER_MIN_DELTA,
    SPACY_NER_PATIENCE,
    get_sqlalchemy_engine,
    get_subprocess_env,
)
from ..Context_Integration.Context_Library.constants import (
    ELECTION_ENTITY_LABELS,
    ENTITY_PATTERNS,
    MISALIGNED_PATTERNS,
    PARTY_KEYWORDS,
)
from ..Context_Integration.librarian import load_context_library
from ..utils.db_utils import get_session
from ..utils.logger_singleton import console, logger
from ..utils.misc_utils import _safe_db_path
from ..utils.model_registry import ModelRegistry
from ..utils.models import (
    Base,
    Candidate,
    Contest,
    County,
    DeclarativeBaseProtocol,
    District,
    Entity,
    MetaDataProtocol,
    Office,
    Party,
    Result,
    State,
    TableStructure,
)
from ..utils.shared_logic import (
    get_or_create,
    safe_add,
    safe_commit,
    safe_encode,
    safe_execute,
    safe_get,
    safe_items,
    safe_model_save,
    safe_replace,
    safe_scalar_one_or_none,
    safe_update,
)


@runtime_checkable
class NERPipeProtocol(Protocol):
    def add_label(self, label: str) -> None: ...

class MakeDocProtocol(Protocol):
    def make_doc(self, text: str) -> Any: ...
    
# --- Logging Setup ---

# --- Advanced Entity Models (see models.py for full implementation) ---
# See previous answer for SQLAlchemy models: Party, State, County, District, Office, Candidate, Contest, Result, etc.

# --- Entity Extraction and Normalization ---
def normalize_entity(value: str) -> str:
    if not value or not isinstance(value, str):
        return ""
    return value.strip().title()

def normalize_entity_list(entity_list: List[str]) -> List[str]:
    return sorted(set(normalize_entity(e) for e in entity_list if e and isinstance(e, str)))

# --- Advanced DB Update Function ---
def update_advanced_entities(parsed_data: List[Dict[str, Any]], db_path: str) -> List[Any]:
    """
    Robustly upserts advanced entity rows into the database.
    Uses db_path for session, logs all actions, and returns a list of upserted Result objects.
    Handles missing/invalid data and logs errors per row.
    """

    results = []
    safe_db = _safe_db_path(db_path)
    with get_session(safe_db) as session:
        for row in parsed_data:
            try:
                party = get_or_create(session, Party, name=normalize_entity(safe_get(row, "party", "")))
                state = get_or_create(session, State, name=normalize_entity(safe_get(row, "state", "")))
                county = get_or_create(session, County, name=normalize_entity(safe_get(row, "county", "")), state=state)
                office = get_or_create(session, Office, name=normalize_entity(safe_get(row, "office", "")))
                district = get_or_create(session, District, name=normalize_entity(safe_get(row, "district", "")), state=state)
                candidate = get_or_create(
                    session, Candidate,
                    name=normalize_entity(safe_get(row, "candidate", "")),
                    party=party, district=district, office=office
                )
                contest = get_or_create(
                    session, Contest,
                    title=normalize_entity(safe_get(row, "contest", "")),
                    year=safe_get(row, "year"),
                    state=state, county=county, district=district, office=office
                )
                result = get_or_create(
                    session, Result,
                    candidate=candidate, contest=contest,
                    votes=safe_get(row, "votes"),
                    percent=safe_get(row, "percent"),
                    is_winner=safe_get(row, "is_winner", False),
                    is_incumbent=safe_get(row, "is_incumbent", False),
                    vote_method=safe_get(row, "vote_method")
                )
                results.append(result)
                console.panel(f"Upserted result for candidate {getattr(candidate, 'name', '?')} in contest {getattr(contest, 'title', '?')}")
            except Exception as e:
                console.table(f"Failed to upsert entity row: {row} ({e})")
        safe_commit(session)
    console.log(f"Advanced entity DB update complete. Upserted {len(results)} results.")
    return results

def is_misaligned_text(text):
    for pat in MISALIGNED_PATTERNS:
        if re.match(pat, text):
            return True
    return False

def clean_misaligned_ner_jsonl(jsonl_path: str, extra_patterns=None) -> None:
    """
    Remove misaligned NER examples from a JSONL file based on patterns and alignment check.
    Keeps only valid, aligned examples.
    Uses safe_get and safe_replace for robust access and path handling.
    """
    nlp = spacy.blank("en")
    patterns = MISALIGNED_PATTERNS.copy()
    if extra_patterns:
        patterns.extend(extra_patterns)

    def is_misaligned(text: str) -> bool:
        for pat in patterns:
            if re.match(pat, text):
                return True
        return False

    cleaned = []
    misaligned = []
    if not os.path.exists(jsonl_path):
        logger.warning(f"[CLEAN] File not found: {jsonl_path}")
        return

    with open(jsonl_path, "rb") as f:
        for line in f:
            try:
                obj = orjson.loads(line)
            except Exception as e:
                logger.warning(f"[CLEAN] Could not parse line: {e}")
                continue
            text = safe_get(obj, "text", "")
            entities = safe_get(obj, "entities", [])
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
            except Exception as e:
                logger.warning(f"[CLEAN] Alignment check failed for text: {text[:50]}... ({e})")
                misaligned.append(obj)
                continue
            cleaned.append(obj)

    # Use safe_replace for path handling
    misaligned_path = safe_replace(jsonl_path, ".jsonl", "_misaligned.jsonl")
    with open(jsonl_path, "wb") as f:
        for obj in cleaned:
            f.write(orjson.dumps(obj, option=orjson.OPT_APPEND_NEWLINE))
    if misaligned:
        with open(misaligned_path, "wb") as f:
            for obj in misaligned:
                f.write(orjson.dumps(obj, option=orjson.OPT_APPEND_NEWLINE))
        logger.info(f"[CLEAN] Removed {len(misaligned)} misaligned NER examples. Saved to {misaligned_path}")
    logger.info(f"[CLEAN] Cleaned NER training data saved to {jsonl_path}. Remaining: {len(cleaned)}")

def append_training_data(new_data, path="spacy_ner_train_data.jsonl") -> None:
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

def save_training_data_jsonl(train_data, path="spacy_ner_train_data.jsonl") -> None:
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
    logger.info(f"Saved spaCy NER training data to {safe_path}")

def cluster_container_patterns(log_dir=None, n_clusters=5) -> None:
    """
    Cluster container HTML snippets and metadata for ML/NLP training.
    Prints cluster assignments and common selectors/classes/headings.
    """
    if log_dir is None:
        log_dir = LOG_DIR

    htmls = []
    meta = []
    for path in glob.glob(os.path.join(log_dir, "failed_container_*.json")):
        try:
            with open(path, "rb") as f:
                entry = orjson.loads(f.read())
                htmls.append(safe_get(entry, "html", ""))
                meta.append(entry)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
    if not htmls:
        logger.info("No failed containers to cluster.")
        return

    vectorizer = TfidfVectorizer(max_features=200, stop_words="english")
    X = vectorizer.fit_transform(htmls)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(X)
    clusters = [[] for _ in range(n_clusters)]
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(meta[i])

    for idx, group in enumerate(clusters):
        logger.info(f"\n=== Cluster {idx+1} ({len(group)} containers) ===")
        selectors = [safe_get(g, "selector", None) for g in group]
        parent_classes = [safe_get(g, "parent_class", None) for g in group]
        headings = [safe_get(g, "heading", None) for g in group]
        logger.info("  Common selectors: %s", Counter(selectors).most_common(3))
        logger.info("  Common parent classes: %s", Counter(parent_classes).most_common(3))
        logger.info("  Common headings: %s", Counter(headings).most_common(3))
        example_html = safe_replace(safe_get(group[0], "html", "")[:200], "\n", " ") if group else ""
        logger.info("  Example HTML snippet: %s", example_html)

def auto_label_header(header: str, context: dict = None) -> List[Tuple[int, int, str]]:
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

def extract_candidates_from_context(context) -> List[str]:
    candidates = set(safe_get(context, "known_candidates", []))
    for contest in safe_get(context, "contests", []):
        t = safe_get(contest, "title", "") if isinstance(contest, dict) else contest
        for match in re.findall(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", str(t)):
            candidates.add(match)
    return list(candidates)

def entity_frequency_analysis(train_data) -> None:
    counter = Counter()
    for _, annots in train_data:
        for _, _, label in annots["entities"]:
            counter[label] += 1
    logger.info("Entity frequency:", counter)

def update_db_with_new_entities(new_entities, db_path: str = None) -> None:
    """
    Update the Entity table in PostgreSQL with new entities using SQLAlchemy.
    All DB operations are wrapped in safe_* helpers.
    """
    safe_path = _safe_db_path(db_path)
    with get_session(safe_path) as session:
        for entity_type, values in safe_items(new_entities):
            for value in values:
                result = safe_execute(session, select(Entity).where(Entity.entity_type == entity_type, Entity.value == value))
                exists = safe_scalar_one_or_none(result) if result is not None else None
                if not exists:
                    safe_add(session, Entity(entity_type=entity_type, value=value))
        safe_commit(session)
    logger.info(f"Updated DB with new entities: {{ { {k: len(v) for k, v in safe_items(new_entities)} } }}")

def load_spacy_ner_examples(jsonl_path) -> List[Tuple[str, Dict[str, Any]]]:
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

def remove_overlapping_entities(entities) -> List[Tuple[int, int, str]]:
    """
    Remove overlapping and duplicate-span entities from a list of (start, end, label) tuples.
    Keeps the longest span first, then next non-overlapping, and only one label per span.
    Priority is determined by the order of ELECTION_ENTITY_LABELS in constants.py.
    """
    def label_rank(label) -> int:
        try:
            return ELECTION_ENTITY_LABELS.index(label)
        except ValueError:
            return len(ELECTION_ENTITY_LABELS)
    entities = sorted(entities, key=lambda x: (x[0], -(x[1] - x[0]), label_rank(x[2])))
    result = []
    last_end = -1
    seen_spans = set()
    for start, end, label in entities:
        if start >= last_end and (start, end) not in seen_spans:
            result.append((start, end, label))
            last_end = end
            seen_spans.add((start, end))
    return result

def validate_training_data(
    train_data,
    nlp: MakeDocProtocol,
    logged=None
) -> List[Tuple[str, Dict[str, Any]]]:
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
                    logger.warning(f"Skipping misaligned entity in: {text}")
                continue
            valid_data.append((text, annots))
        except Exception as e:
            if logged:
                logger.warning(f"Error validating entity alignment: {e}")
    return valid_data

def retrain_spacy_ner_advanced(
    confirmed_structures,
    context_library=None,
    model_save_path="fine_tuned_spacy_ner",
    max_epochs=None,
    patience=3,
    min_delta=0.01,
    batch_size=32
) -> None:
    # Create blank English model
    nlp: Language = spacy.blank("en")

    # Try to use GPU if available (cross-platform)
    try:
        if hasattr(spacy, "prefer_gpu") and callable(spacy.prefer_gpu):
            used_gpu: bool = spacy.prefer_gpu()
            if used_gpu:
                logger.info("[INFO] spaCy using GPU for training.")
            else:
                logger.info("[INFO] spaCy using CPU for training.")
        else:
            logger.info("[INFO] spaCy GPU preference not available.")
    except Exception as e:
        logger.warning(f"[spaCy] Could not check GPU availability: {e}")

    # --- Robust lexeme normalization loading (optional, cross-platform, type-annotated) ---
    try:
        lookups_mod: Optional[ModuleType] = find_spec("spacy.lookups") if callable(find_spec) else None
        if lookups_mod and hasattr(Lookups(), "add_table"):
            lookups: Lookups = Lookups()
            lookups_data_loader: Optional[Any] = getattr(getattr(spacy, "lookups", None), "load_lookups_data", None)
            if callable(lookups_data_loader):
                loaded = lookups_data_loader("en", tables=["lexeme_norm"])
                get_table_fn: Optional[Any] = getattr(loaded, "get_table", None)
                if callable(get_table_fn):
                    lexeme_norm_table = get_table_fn("lexeme_norm")
                    lookups.add_table("lexeme_norm", lexeme_norm_table)
                    nlp.vocab.lookups = lookups
    except Exception as e:
        logger.warning(f"[spaCy] Could not load lexeme normalization table. You may ignore this for English. Error: {e}")

    # Add NER pipe and labels with type annotations
    ner = nlp.add_pipe("ner") if "ner" not in nlp.pipe_names else nlp.get_pipe("ner")
    assert isinstance(ner, NERPipeProtocol), "NER pipe does not implement add_label(str)"
    for label in ELECTION_ENTITY_LABELS:
        ner.add_label(str(label))

    # --- Data Preparation ---
    known_context = copy.deepcopy(context_library) if context_library else {}
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
        logger.info(f"Loaded {len(extra_examples)} extra NER examples from log/spacy_ner_train_data.jsonl")
    train_data.extend(extra_examples)

    # --- Auto-labeling confirmed_structures ---
    for struct in confirmed_structures:
        headers = safe_get(struct, "headers", [])
        context = copy.deepcopy(safe_get(struct, "context", {}))
        # Merge known context for robustness, always use safe_get and deduplicate
        for k in ["known_counties", "known_cities", "known_states", "known_candidates", "known_districts"]:
            context[k] = list(set(safe_get(context, k, []) + safe_get(known_context, k, [])))
        # Extract candidates from context, deduplicate, and filter by length/threshold if needed
        context_candidates = extract_candidates_from_context(context)
        # Optionally filter out very short/likely-noisy candidates (e.g., < 3 chars)
        context["known_candidates"] = list(set(
            c for c in (safe_get(context, "known_candidates", []) + context_candidates)
            if isinstance(c, str) and len(c.strip()) > 2
        ))
        all_candidates.update(context["known_candidates"])
        # Use PARTY_KEYWORDS for robust party extraction, only add if match is not too short
        party_pattern = r"\b(" + "|".join(re.escape(p) for p in PARTY_KEYWORDS) + r")\b"
        found_parties = [
            p for p in re.findall(party_pattern, " ".join(headers), re.IGNORECASE)
            if isinstance(p, str) and len(p.strip()) > 2
        ]
        all_parties.update(found_parties)
        # Defensive: only add non-empty, non-trivial values for all entity sets
        all_counties.update([c for c in safe_get(context, "known_counties", []) if c and isinstance(c, str) and len(c.strip()) > 2])
        all_states.update([s for s in safe_get(context, "known_states", []) if s and isinstance(s, str) and len(s.strip()) > 1])
        all_districts.update([d for d in safe_get(context, "known_districts", []) if d and isinstance(d, str) and len(d.strip()) > 0])
        all_locations.update([
            city
            for city in safe_get(context, "known_cities", [])
            if city and isinstance(city, str) and len(city.strip()) > 1
        ])

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
        logger.warning(f"[NER] Skipped {misaligned_count} misaligned examples. Saved to {misaligned_path}")
    train_data = valid_data
    save_training_data_jsonl(train_data)
    entity_frequency_analysis(train_data)
    logger.info(f"[NER] Used {len(train_data)} valid examples, skipped {misaligned_count} misaligned.")

    # --- Training ---
    examples = []
    for text, annots in train_data:
        doc = nlp.make_doc(text)
        annots["entities"] = remove_overlapping_entities(annots["entities"])
        example = Example.from_dict(doc, annots)
        examples.append(example)
    if not examples:
        logger.warning("No NER training examples found. Skipping spaCy NER retraining.")
        return

    optimizer = nlp.begin_training()
    optimizer.learn_rate = 0.001

    # --- Dynamic Early Stopping and Adaptive min_delta ---
    epochs = max_epochs or SPACY_NER_EPOCHS
    patience = SPACY_NER_PATIENCE
    min_delta = SPACY_NER_MIN_DELTA
    batch_size = SPACY_NER_BATCH_SIZE
    no_improve = 0
    best_loss = float("inf")
    best_model_path = model_save_path + "_best"
    best_epoch = 0
    loss_history = []

    logger.info(f"[INFO] Starting spaCy NER training for up to {epochs} epochs, batch size {batch_size}...")
    for i in range(epochs):
        losses = {}
        random.shuffle(examples)
        with logger.progress_bar(f"spaCy NER epoch {i+1}", total=len(range(0, len(examples), batch_size))) as update_progress:
            for batch_idx, batch in enumerate([examples[j:j+batch_size] for j in range(0, len(examples), batch_size)]):
                nlp.update(batch, sgd=optimizer, drop=0.2, losses=losses)
                update_progress(batch_idx + 1)
        epoch_loss = losses.get("ner", 0)
        loss_history.append(epoch_loss)
        logger.info(f"spaCy NER retraining epoch {i+1}, loss: {epoch_loss:.4f}")

        # --- Dynamic min_delta: scale with loss magnitude ---
        if i == 0 and min_delta < 1:
            min_delta = max(0.01, epoch_loss * 0.01)
            logger.info(f"[AUTO] Adjusted min_delta to {min_delta:.2f} based on initial loss.")

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
            logger.info(f"[INFO] New best model saved at epoch {i+1} with smoothed loss {smoothed_loss:.2f}")
        else:
            no_improve += 1

        # --- Dynamic patience: extend if still improving fast ---
        if no_improve >= patience:
            logger.info(f"[INFO] Early stopping at epoch {i+1} (no improvement for {patience} epochs).")
            break
        if i > 2 and smoothed_loss < 0.5 * loss_history[0] and patience < 8:
            patience += 1
            logger.info(f"[AUTO] Increased patience to {patience} due to rapid improvement.")

    # Restore best model
    if os.path.exists(best_model_path):
        logger.info(f"[INFO] Restoring best model from epoch {best_epoch} with loss {best_loss:.2f}")
        nlp = spacy.load(best_model_path)
        shutil.rmtree(best_model_path, ignore_errors=True)
    nlp.to_disk(model_save_path)
    logger.info(f"Fine-tuned spaCy NER model saved to: {model_save_path}")

    # --- Training summary and suggestions ---
    logger.info(f"[SUMMARY] Best loss: {best_loss:.2f} at epoch {best_epoch}")
    if best_epoch < epochs:
        logger.warning("[SUGGESTION] Consider lowering min_delta or increasing patience if you want longer training.")
    elif best_epoch == epochs:
        logger.warning("[SUGGESTION] Model improved until the last epoch. Consider increasing epochs for further improvement.")
    logger.warning(f"[SUGGESTION] Next run: patience={patience}, min_delta={min_delta:.2f}, epochs={epochs}")

    # --- Robust DB update with all entity types ---
    new_entities = {}
    for label in ELECTION_ENTITY_LABELS:
        if label == "CANDIDATE":
            values = all_candidates
        elif label == "PARTY":
            values = all_parties
        elif label == "COUNTY":
            values = all_counties
        elif label == "STATE":
            values = all_states
        elif label == "DISTRICT":
            values = all_districts
        elif label == "LOCATION":
            values = all_locations
        else:
            # Try both plural and singular keys, fallback to empty list
            values = safe_get(context_library, f"known_{label.lower()}s", [])
            if not values:
                values = safe_get(context_library, f"known_{label.lower()}", [])
        new_entities[label] = normalize_entity_list(values)
    update_db_with_new_entities(new_entities, _safe_db_path(CONTEXT_DB_PATH))
    logger.info(f"[DB] Updated with entities: {{ { {k: len(v) for k, v in new_entities.items()} } }}")

def get_all_confirmed_structures() -> List[Dict[str, Any]]:
    """
    Retrieve all confirmed table structures from PostgreSQL using SQLAlchemy.
    Returns a list of dicts, not ORM objects, to avoid DetachedInstanceError.
    """
    with get_session() as session:
        rows = session.execute(
            select(TableStructure).where(TableStructure.confirmed_by_user.is_(True))
        ).scalars().all()
        # Extract all needed fields while session is open
        result = []
        for row in rows:
            result.append({
                "contest": row.contest,
                "headers": orjson.loads(row.headers) if isinstance(row.headers, (str, bytes, bytearray)) else row.headers,
                "context": orjson.loads(row.context) if isinstance(row.context, (str, bytes, bytearray)) else row.context,
                "original_structure": getattr(row, "original_structure", {}),
                "corrected_structure": getattr(row, "corrected_structure", {}),
                "sample_rows": getattr(row, "sample_rows", [{}]),
            })
        return result
    
def run_manual_correction() -> None:
    """
    Run the manual correction bot robustly as a module, capturing output and errors.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "webapp.parser.bots.manual_correction", "--fields", "tables", "--enhanced"],
            check=True,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env=get_subprocess_env()
        )
        logger.info(result.stdout)
        if result.stderr:
            logger.info(result.stderr)
    except subprocess.CalledProcessError as e:
        logger.error(f"[ERROR] Manual correction bot failed: {e.stderr}")

def retrain_sentence_transformer(confirmed_structures, model_save_path=None, epochs=1, batch_size=8) -> None:
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
        contest = safe_get(struct, "contest", "")
        headers = safe_get(struct, "headers", [])
        for header in headers:
            train_examples.append(InputExample(texts=[contest, header], label=1.0))
    if not train_examples:
        logger.warning("No training examples found. Aborting retraining.")
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
            logger.info(f"[CLEANUP] Deleted oldest fine-tuned model directory: {oldest_path}")
        except Exception as e:
            logger.warning(f"[WARN] Could not delete old model directory {oldest_path}: {e}")

    # Model loading
    model = None
    prev_model_dir = os.path.join(base_dir, "fine_tuned_table_headers")
    model_files = ["config.json", "pytorch_model.bin", "model.safetensors", "tf_model.h5", "model.ckpt.index", "flax_model.msgpack"]
    model_files_exist = any(os.path.exists(os.path.join(prev_model_dir, f)) for f in model_files)
    if model_files_exist:
        logger.info(f"Attempting to load existing model from {prev_model_dir} for further fine-tuning...")
        try:
            model = ModelRegistry.get_sentence_transformer(model_name=prev_model_dir, use_finetuned=False)
        except Exception as e:
            logger.warning(f"[WARN] Failed to load existing model: {e}")
            model = None
    if model is None:
        logger.warning("Falling back to base model (all-MiniLM-L6-v2).")
        try:
            model = ModelRegistry.get_sentence_transformer(model_name="all-MiniLM-L6-v2", use_finetuned=False)
        except Exception as e:
            logger.error(f"[ERROR] Could not load base SentenceTransformer: {e}")
            return

    # Defensive: clean up incomplete/corrupt model directory before saving
    for f in model_files:
        fpath = os.path.join(model_save_path, f)
        if os.path.exists(fpath) and os.path.getsize(fpath) == 0:
            logger.info(f"[CLEANUP] Removing empty/corrupt file: {fpath}")
            try:
                os.remove(fpath)
            except Exception:
                pass

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model)
    logger.info(f"Retraining SentenceTransformer on {len(train_examples)} pairs for {epochs} epoch(s)...")
    try:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=10,
            show_progress_bar=True
        )
    except Exception as e:
        logger.error(f"[ERROR] Model training failed: {e}")
        return
    try:
        safe_model_save(model, model_save_path)
        logger.info(f"Fine-tuned model saved to: {model_save_path}")
        canonical_dir = os.path.join(base_dir, "fine_tuned_table_headers")
        try:
            if os.path.exists(canonical_dir):
                shutil.rmtree(canonical_dir, ignore_errors=True)
            shutil.copytree(model_save_path, canonical_dir)
            logger.info(f"[INFO] Copied new model to canonical directory: {canonical_dir}")
        except Exception as e:
            logger.warning(f"[WARN] Could not update canonical model directory: {e}")
    except Exception as e:
        logger.error(f"[ERROR] Model save failed: {e}")
        
def segment_hash(segment) -> str:
    """Generate a stable hash for a DOM segment based on tag, attrs, and first 200 chars of HTML."""
    tag = safe_get(segment, "tag", "")
    attrs = safe_get(segment, "attrs", {})
    html = safe_get(segment, "html", "")[:200]
    # Use OPT_SORT_KEYS for orjson
    attrs_json = safe_encode(attrs, sort_keys=True)
    hash_input = tag + attrs_json + html
    return hashlib.sha256(safe_encode(hash_input)).hexdigest()

def load_cached_segment_hashes(context_library) -> Set[str]:
    """Return a set of all segment_hashes in the context library."""
    return {safe_get(seg, "segment_hash") for seg in safe_get(context_library, "cached_segments", [])}

def scan_in_memory_ner_examples(train_data, verbose=False) -> List[Tuple[str, List[Tuple[int, int, str]]]]:
    """Scan a list of (text, annots) NER examples for misalignments using spaCy's offsets_to_biluo_tags."""
    nlp = spacy.blank("en")
    misaligned = []
    for text, annots in train_data:
        try:
            tags = offsets_to_biluo_tags(nlp.make_doc(text), annots["entities"])
            if "-" in tags:
                misaligned.append((text, annots["entities"]))
                if verbose:
                    logger.warning(f"MISALIGNED: {text} {annots['entities']}")
        except Exception as e:
            misaligned.append((text, annots["entities"]))
            if verbose:
                logger.error(f"ERROR: {text} {annots['entities']} ({e})")
    return misaligned

def ensure_table_structures_exists() -> None:
    """
    Ensure the 'table_structures' table exists in the database.
    If not, create all tables defined in models.
    Enhanced for clarity and robust error handling.
    """
    engine = get_sqlalchemy_engine()
    inspector = inspect(engine)

    try:
        table_names = inspector.get_table_names()
        logger.debug(f"[DB] Existing tables in DB: {table_names}")
    except Exception as e:
        logger.error(f"[DB] Could not retrieve table names: {e}")
        table_names = []

    # Use Protocol for type annotation
    base: DeclarativeBaseProtocol = Base
    metadata: MetaDataProtocol = base.metadata

    if 'table_structures' not in table_names:
        logger.info("[INFO] 'table_structures' table not found. Creating all tables from Base.metadata...")
        if not metadata.tables:
            logger.warning("[DB] Base.metadata.tables is empty. No models registered? Did you import all model classes?")
        try:
            metadata.create_all(engine)
            logger.info("[INFO] All tables created via Base.metadata.create_all(engine).")
        except Exception as e:
            logger.error(f"[DB] Failed to create tables: {e}")
            raise
    else:
        logger.info("[INFO] 'table_structures' table exists.")

def main() -> None:
    ensure_table_structures_exists()
    if REVIEW_WITH_MANUAL_BOT:
        run_manual_correction()

    # --- Self-cleaner for NER training data ---
    ner_train_jsonl = os.path.join(LOG_DIR, "spacy_ner_train_data.jsonl")
    clean_misaligned_ner_jsonl(ner_train_jsonl)

    confirmed_structures = get_all_confirmed_structures()
    console.table(f"Found {len(confirmed_structures)} confirmed table structures.")

    # Log user feedback/corrections for ML
    feedback_log_path = os.path.join(LOG_DIR, "structure_feedback_log.jsonl")
    os.makedirs(os.path.dirname(feedback_log_path), exist_ok=True)
    for struct in confirmed_structures:
        old_structure_info = safe_get(struct, "original_structure", {})
        structure_info = safe_get(struct, "corrected_structure", {})
        headers = safe_get(struct, "headers", [])
        data = safe_get(struct, "sample_rows", [{}])
        with open(feedback_log_path, "ab") as f:
            f.write(orjson.dumps({
                "original_structure": old_structure_info,
                "corrected_structure": structure_info,
                "headers": headers,
                "sample_row": data[0] if data else {},
            }, option=orjson.OPT_APPEND_NEWLINE))
    context_library = load_context_library()
    logger.debug("DEBUG: Loaded context library:", type(context_library))
    if not isinstance(context_library, dict):
        logger.error("ERROR: Context library is not a dictionary. Check your context library loading logic.")
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
        headers = safe_get(struct, "headers", [])
        context = safe_get(struct, "context", {})
        safe_update(context, {
            "known_counties": safe_get(context_library, "known_counties", []),
            "known_cities": safe_get(context_library, "known_cities", []),
            "known_states": safe_get(context_library, "known_states", []),
            "known_candidates": safe_get(context_library, "known_candidates", []),
            "known_districts": safe_get(context_library, "known_districts", []),
        })
        context_candidates = extract_candidates_from_context(context)
        context["known_candidates"] = list(set(safe_get(context, "known_candidates", []) + context_candidates))
        all_candidates.update(context["known_candidates"])
        party_pattern = r"\b(" + "|".join(re.escape(p) for p in PARTY_KEYWORDS) + r")\b"
        all_parties.update([
            p for p in re.findall(party_pattern, " ".join(headers), re.IGNORECASE)
        ])
        all_counties.update(safe_get(context, "known_counties", []))
        all_states.update(safe_get(context, "known_states", []))
        all_districts.update(safe_get(context, "known_districts", []))
        all_locations.update(safe_get(context, "known_cities", []))
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
        console.panel(f"{len(misaligned)} misaligned NER examples found in final training data. Running diagnostics and launching manual_correction. Aborting retraining.")
        misaligned_path = os.path.join(LOG_DIR, "spacy_ner_misaligned.jsonl")
        with open(misaligned_path, "wb") as f:
            for text, entities in misaligned:
                f.write(orjson.dumps({"text": text, "entities": entities}, option=orjson.OPT_APPEND_NEWLINE))
        try:
            subprocess.run([
                sys.executable, "-m", "webapp.parser.bots.scan_misaligned_ner", "--jsonl", misaligned_path
            ], check=True, cwd=PROJECT_ROOT, env=get_subprocess_env())
        except Exception as e:
            console.table(f"scan_misaligned_ner diagnostics failed: {e}")
        run_manual_correction()
        console.log("Please correct misalignments and rerun retraining.")
        sys.exit(2)

    # Use deduped_train_data for retraining
    retrain_sentence_transformer(
        deduped_train_data,
        epochs=SBERT_EPOCHS,
        batch_size=SBERT_BATCH_SIZE
    )
    retrain_spacy_ner_advanced(
        deduped_train_data,
        context_library,
        max_epochs=SPACY_NER_EPOCHS,
        patience=SPACY_NER_PATIENCE,
        min_delta=SPACY_NER_MIN_DELTA,
        batch_size=SPACY_NER_BATCH_SIZE
    )
    cluster_container_patterns()
    console.log("\n[SUMMARY] Table Structure Model Retraining Complete.")
    console.log("If you see repeated model save failures, close any file explorers or editors viewing the model directory.")
    console.log("If you see spaCy lexeme normalization warnings, you can ignore them for English. To suppress, install spacy-lookups-data and load the table if needed.")
    console.log("If you see spaCy entity alignment warnings, consider cleaning your training data or using the provided validation function.")
    gc.collect()

if __name__ == "__main__":
    main()
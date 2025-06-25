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
from ..utils.model_registry import ModelRegistry
from collections import Counter
from sentence_transformers import InputExample, losses
from torch.utils.data import DataLoader
from ..utils.shared_logic import load_context_library
from ..utils.db_utils import _safe_db_path
from ..config import CONTEXT_DB_PATH, MODEL_DIR, PROJECT_ROOT, POSTGRES_URL

import spacy
from spacy.training import Example, offsets_to_biluo_tags
from spacy.lookups import Lookups
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import logging
import argparse
import gc
from sqlalchemy.orm import Session
from sqlalchemy import select, inspect
from webapp.parser.utils.db_utils import get_session, create_engine
from webapp.parser.utils.models import TableStructure, Entity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("manual_correction_bot")

ELECTION_ENTITY_LABELS = [
    "CONTEST", "CANDIDATE", "PARTY", "COUNTY", "STATE", "DISTRICT", "VOTE_METHOD",
    "BALLOT_TYPE", "PRECINCT", "TOTAL", "PERCENT", "YEAR", "ELECTION_TYPE", "OFFICE", "MISC",
    "BALLOT_MEASURE", "LOCATION", "DATE", "INCUMBENT", "WINNER", "LOSER", "WRITE_IN", "UNOPPOSED", "PROPOSITION", 
    "AMENDMENT", "DISTRICT_TYPE", "JURISDICTION", "ELECTION_OFFICIAL", "RESULTS", "VOTE_COUNT", "AFFIDAVIT", "OTHER"   
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
    (r"\bgeneral|primary|special\b", "ELECTION_TYPE"),
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
    (r"\b(?:election|vote|poll|referendum|plebiscite)\b", "ELECTION_TYPE"),  
    (r"\b(?:candidate|nominee|aspirant|hopeful)\b", "CANDIDATE"),
    (r"\b(?:election official|poll worker|election judge|inspector)\b", "ELECTION_OFFICIAL"),
    
    # Add more as needed
]

def safe_model_save(model, model_save_path, retries=3):

    for attempt in range(1, retries+1):
        try:
            model.save(model_save_path)
            print(f"[INFO] Model saved successfully on attempt {attempt}.")
            return
        except Exception as e:
            print(f"[WARN] Model save failed (attempt {attempt}): {e}")
            time.sleep(2 * attempt)
            gc.collect()
    # Try saving to a temp dir and moving
    tmp_path = model_save_path + "_tmp"
    try:
        model.save(tmp_path)
        shutil.rmtree(model_save_path, ignore_errors=True)
        shutil.move(tmp_path, model_save_path)
        print(f"[INFO] Model saved via temp path workaround.")
    except Exception as e:
        print(f"[ERROR] Final model save failed: {e}\nIf you see repeated save failures, close any file explorers or editors viewing the model directory.")

def append_training_data(new_data, path="spacy_ner_train_data.jsonl"):
    """
    Appends new training data to a JSONL file in the log directory, deduplicating by text/entities,
    and adds a timestamp to each entry. Uses the log_dir as the parent of the model_dir for safety.
    """
    main_dir = MODEL_DIR if 'MODEL_DIR' in globals() else os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../model"))
    log_dir = os.path.abspath(os.path.join(os.path.dirname(main_dir), "log"))
    os.makedirs(log_dir, exist_ok=True)
    safe_path = os.path.abspath(os.path.join(log_dir, path))
    # Harden: ensure safe_path is inside log_dir
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
    # Save to the log/ directory at the project root
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../log"))
    os.makedirs(log_dir, exist_ok=True)
    filename = os.path.basename(path)
    filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', filename)
    safe_path = os.path.join(log_dir, filename)
    if not os.path.abspath(safe_path).startswith(log_dir):
        raise ValueError("Unsafe path detected for training data output!")
    with open(safe_path, "wb") as f:
        for text, annots in train_data:
            f.write(orjson.dumps({"text": text, "entities": annots["entities"]}, option=orjson.OPT_APPEND_NEWLINE))
    print(f"Saved spaCy NER training data to {safe_path}")

def cluster_container_patterns(log_dir=None, n_clusters=5):
    """
    Cluster container HTML snippets and metadata for ML/NLP training.
    Prints cluster assignments and common selectors/classes/headings.
    """


    if log_dir is None:
        log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../log"))

    htmls = []
    meta = []
    for path in glob.glob(os.path.join(log_dir, "failed_container_*.json")):
        with open(path, "rb") as f:
            entry = orjson.loads(f.read())
            htmls.append(entry.get("html", ""))
            meta.append(entry)
    if not htmls:
        print("No failed containers to cluster.")
        return

    vectorizer = TfidfVectorizer(max_features=200, stop_words="english")
    X = vectorizer.fit_transform(htmls)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(X)
    clusters = [[] for _ in range(n_clusters)]
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(meta[i])

    for idx, group in enumerate(clusters):
        print(f"\n=== Cluster {idx+1} ({len(group)} containers) ===")
        selectors = [g.get("selector") for g in group]
        parent_classes = [g.get("parent_class") for g in group]
        headings = [g.get("heading") for g in group]
        print("  Common selectors:", Counter(selectors).most_common(3))
        print("  Common parent classes:", Counter(parent_classes).most_common(3))
        print("  Common headings:", Counter(headings).most_common(3))
        print("  Example HTML snippet:", group[0].get("html", "")[:200].replace("\n", " ") if group else "")

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
    print("Entity frequency:", counter)

def update_db_with_new_entities(new_entities, db_path):
    """
    Update the Entity table in PostgreSQL with new entities using SQLAlchemy.
    """
    with get_session() as session:
        for entity_type, values in new_entities.items():
            for value in values:
                exists = session.execute(
                    select(Entity).where(Entity.entity_type == entity_type, Entity.value == value)
                ).scalar_one_or_none()
                if not exists:
                    session.add(Entity(entity_type=entity_type, value=value))
        session.commit()
    print(f"Updated DB with new entities: {{ { {k: len(v) for k,v in new_entities.items()} } }}")

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
    "CANDIDATE", "PARTY", "VOTE_METHOD", "PRECINCT", "DISTRICT", "COUNTY", "STATE", "TOTAL", "ELECTION_TYPE", "OFFICE", "WRITE_IN", "MISC", "BALLOT_MEASURE", "LOCATION", "DATE", "INCUMBENT", "WINNER", "LOSER", "UNOPPOSED", "PROPOSITION", "AMENDMENT", "DISTRICT_TYPE", "JURISDICTION", "ELECTION_OFFICIAL", "RESULTS", "VOTE_COUNT", "AFFIDAVIT", "OTHER"
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

def validate_training_data(train_data, nlp, logger=None):
    """
    Validate and skip misaligned spaCy NER training examples to avoid [W030] warnings.
    Pre-check alignment before creating Example.
    """
    valid_data = []
    for text, annots in train_data:
        try:
            tags = offsets_to_biluo_tags(nlp.make_doc(text), annots["entities"])
            if "-" in tags:
                if logger:
                    logger.warning(f"Skipping misaligned entity in: {text}")
                continue
            valid_data.append((text, annots))
        except Exception as e:
            if logger:
                logger.warning(f"Error validating entity alignment: {e}")
    return valid_data

def retrain_spacy_ner_advanced(confirmed_structures, context_library=None, model_save_path="fine_tuned_spacy_ner"):
    import importlib
    nlp = spacy.blank("en")
    # --- Robust lexeme normalization loading ---
    try:
        lookups_mod = importlib.util.find_spec("spacy.lookups")
        if lookups_mod and hasattr(Lookups(), "add_table"):
            lookups = Lookups()
            # Only attempt to load lexeme_norm if the function exists
            if hasattr(spacy.lookups, "load_lookups_data"):
                lookups.add_table("lexeme_norm", spacy.lookups.load_lookups_data("en", tables=["lexeme_norm"]).get_table("lexeme_norm"))
                nlp.vocab.lookups = lookups
    except Exception as e:
        print("[spaCy] Could not load lexeme normalization table. You may ignore this for English. To suppress, install spacy-lookups-data and load the table if needed. Error:", e)

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    for label in ELECTION_ENTITY_LABELS:
        ner.add_label(label)

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
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../log/spacy_ner_train_data.jsonl"))
    )
    if extra_examples:
        print(f"Loaded {len(extra_examples)} extra NER examples from log/spacy_ner_train_data.jsonl")
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
            entities = auto_label_header(header, context)
            if entities:
                # Remove overlapping entities before adding to train_data
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
                if logger:
                    logger.warning(f"Skipping misaligned entity in: {text}")
                continue
            valid_data.append((text, annots))
        except Exception as e:
            misaligned_count += 1
            misaligned_examples.append({"text": text, "entities": annots["entities"], "error": str(e)})
            if logger:
                logger.warning(f"Error validating entity alignment: {e}")
    if misaligned_examples:
        # Save misaligned examples for review
        misaligned_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../log/spacy_ner_misaligned.jsonl"))
        with open(misaligned_path, "wb") as f:
            for ex in misaligned_examples:
                f.write(orjson.dumps(ex, option=orjson.OPT_APPEND_NEWLINE))
        print(f"[NER] Skipped {misaligned_count} misaligned examples. Saved to {misaligned_path}")
    train_data = valid_data
    save_training_data_jsonl(train_data)
    entity_frequency_analysis(train_data)
    print(f"[NER] Used {len(train_data)} valid examples, skipped {misaligned_count} misaligned.")

    # Convert to spaCy Example objects
    examples = []
    for text, annots in train_data:
        doc = nlp.make_doc(text)
        annots["entities"] = remove_overlapping_entities(annots["entities"])
        example = Example.from_dict(doc, annots)
        examples.append(example)
    if not examples:
        print("No NER training examples found. Skipping spaCy NER retraining.")
        return
    optimizer = nlp.begin_training()
    for i in range(10):
        losses = {}
        nlp.update(examples, drop=0.2, losses=losses)
        if "ner" in losses:
            print(f"spaCy NER retraining epoch {i+1}, loss: {losses['ner']}")
        else:
            print(f"spaCy NER retraining epoch {i+1}, loss: N/A")
    nlp.to_disk(model_save_path)
    print(f"Fine-tuned spaCy NER model saved to: {model_save_path}")

    # Update DB with new entities
    new_entities = {
        "CANDIDATE": list(all_candidates),
        "PARTY": list(all_parties),
        "COUNTY": list(all_counties),
        "STATE": list(all_states),
        "DISTRICT": list(all_districts),
        "LOCATION": list(all_locations),
        "VOTE_METHOD": list(context_library.get("known_vote_methods", [])),
        "BALLOT_MEASURE": list(context_library.get("known_ballot_measures", [])),
        "ELECTION_TYPE": list(context_library.get("known_election_types", [])),
        "YEAR": list(context_library.get("known_years", [])),
        "MISC": list(context_library.get("known_misc", [])),
        "OFFICE": list(context_library.get("known_offices", [])),
        "ELECTION_OFFICIAL": list(context_library.get("known_election_officials", [])),
        "RESULTS": list(context_library.get("known_results", [])),
        "VOTE_COUNT": list(context_library.get("known_vote_counts", [])),        
    }
    update_db_with_new_entities(new_entities, _safe_db_path(CONTEXT_DB_PATH))
    
def get_all_confirmed_structures():
    """
    Retrieve all confirmed table structures from PostgreSQL using SQLAlchemy.
    """
    with get_session() as session:
        rows = session.execute(
            select(TableStructure).where(TableStructure.confirmed_by_user == True)
        ).scalars().all()
    return [
        {
            "contest_title": row.contest_title,
            "headers": orjson.loads(row.headers) if isinstance(row.headers, (str, bytes, bytearray)) else row.headers,
            "context": orjson.loads(row.context) if isinstance(row.context, (str, bytes, bytearray)) else row.context
        }
        for row in rows
    ]

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
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Manual correction bot failed: {e.stderr}")

def retrain_sentence_transformer(confirmed_structures, model_save_path=None):
    """
    Fine-tunes the SentenceTransformer model on confirmed structures.
    Loads the existing model for further training if present, otherwise starts from base.
    Always saves to the same folder (no timestamp).
    """
    train_examples = []
    for struct in confirmed_structures:
        contest_title = struct.get("contest_title", "")
        headers = struct.get("headers", [])
        for header in headers:
            train_examples.append(InputExample(texts=[contest_title, header], label=1.0))
    if not train_examples:
        print("No training examples found. Aborting retraining.")
        return

    base_dir = MODEL_DIR if 'MODEL_DIR' in globals() else os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../model"))
    model_save_path = model_save_path or os.path.join(base_dir, "fine_tuned_table_headers")
    os.makedirs(model_save_path, exist_ok=True)

    # Robust model loading: check for model files
    model = None
    model_files = ["config.json", "pytorch_model.bin", "model.safetensors", "tf_model.h5", "model.ckpt.index", "flax_model.msgpack"]
    model_files_exist = any(os.path.exists(os.path.join(model_save_path, f)) for f in model_files)
    if model_files_exist:
        print(f"Attempting to load existing model from {model_save_path} for further fine-tuning...")
        try:
            model = ModelRegistry.get_sentence_transformer(model_name=model_save_path, use_finetuned=False)
        except Exception as e:
            print(f"[WARN] Failed to load existing model: {e}")
            model = None
    if model is None:
        print("Falling back to base model (all-MiniLM-L6-v2).")
        try:
            model = ModelRegistry.get_sentence_transformer(model_name="all-MiniLM-L6-v2", use_finetuned=False)
        except Exception as e:
            print(f"[ERROR] Could not load base SentenceTransformer: {e}")
            return
    # Defensive: clean up incomplete/corrupt model directory before saving
    for f in model_files:
        fpath = os.path.join(model_save_path, f)
        if os.path.exists(fpath) and os.path.getsize(fpath) == 0:
            print(f"[CLEANUP] Removing empty/corrupt file: {fpath}")
            try:
                os.remove(fpath)
            except Exception:
                pass
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
    train_loss = losses.CosineSimilarityLoss(model)
    print(f"Retraining SentenceTransformer on {len(train_examples)} pairs...")
    try:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=10,
            show_progress_bar=True
        )
    except Exception as e:
        print(f"[ERROR] Model training failed: {e}")
        return
    try:
        safe_model_save(model, model_save_path)
        print(f"Fine-tuned model saved to: {model_save_path}")
    except Exception as e:
        print(f"[ERROR] Model save failed: {e}")
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
                    print(f"MISALIGNED: {text} {annots['entities']}")
        except Exception as e:
            misaligned.append((text, annots["entities"]))
            if verbose:
                print(f"ERROR: {text} {annots['entities']} ({e})")
    return misaligned

def ensure_table_structures_exists():
    """
    Ensure the 'table_structures' table exists in the database.
    If not, create all tables defined in models.
    """
    engine = create_engine(POSTGRES_URL)
    inspector = inspect(engine)
    if 'table_structures' not in inspector.get_table_names():
        print("[INFO] 'table_structures' table not found. Creating all tables...")
        from webapp.parser.utils.models import Base
        Base.metadata.create_all(engine)
        print("[INFO] All tables created.")
    else:
        print("[INFO] 'table_structures' table exists.")


def main():
    ensure_table_structures_exists()
    if os.getenv("REVIEW_WITH_MANUAL_BOT", "false").lower() == "true":
        run_manual_correction_bot()

    confirmed_structures = get_all_confirmed_structures()
    print(f"Found {len(confirmed_structures)} confirmed table structures.")

    # Log user feedback/corrections for ML ---
    feedback_log_path = os.path.join(PROJECT_ROOT, "log", "structure_feedback_log.jsonl")
    os.makedirs(os.path.dirname(feedback_log_path), exist_ok=True)
    for struct in confirmed_structures:
        # Assume struct contains both original and corrected structure info if available
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
    cached_hashes = load_cached_segment_hashes(context_library)
    deduped_train_data = []
    for struct in confirmed_structures:
        seg_hash = segment_hash(struct)
        if seg_hash not in cached_hashes:
            deduped_train_data.append(struct)
    print(f"Deduplicated to {len(deduped_train_data)} unique structures for training.")

    # Build NER training data (auto-label, dedupe, etc.)
    train_data = []
    all_candidates = set()
    all_parties = set()
    all_counties = set()
    all_states = set()
    all_districts = set()
    all_locations = set()
    
    # --- Load extra examples from JSONL file ---
    extra_examples = load_spacy_ner_examples(
        os.path.join(PROJECT_ROOT, "log", "spacy_ner_train_data.jsonl")
    )
    if extra_examples:
        print(f"Loaded {len(extra_examples)} extra NER examples from log/spacy_ner_train_data.jsonl")
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
        all_parties.update([p for p in re.findall(r"\\b(?:Democratic|Republican|Libertarian|Green|Independent|Conservative|Working Families|Write-in|Other)\\b", " ".join(headers), re.IGNORECASE)])
        all_counties.update(context.get("known_counties", []))
        all_states.update(context.get("known_states", []))
        all_districts.update(context.get("known_districts", []))
        all_locations.update(context.get("known_cities", []))
        
        for header in headers:
            # Skip problematic headers that cause repeated misalignments
            if header.strip().lower().startswith("totals -"):
                continue
            entities = auto_label_header(header, context)
            if entities:
                # Remove overlapping entities before adding to train_data
                entities = remove_overlapping_entities(entities)
                train_data.append((header, {"entities": entities}))

    # Scan in-memory NER examples for misalignments before retraining
    print("[INFO] Scanning in-memory NER training data for misalignments before retraining...")
    misaligned = scan_in_memory_ner_examples(train_data, verbose=True)
    if misaligned:
        print(f"[ERROR] {len(misaligned)} misaligned NER examples found in final training data. Running diagnostics and launching manual_correction_bot. Aborting retraining.")
        # Save misaligned examples for review
        misaligned_path = os.path.join(PROJECT_ROOT, "log", "spacy_ner_misaligned.jsonl")
        with open(misaligned_path, "wb") as f:
            for text, entities in misaligned:
                f.write(orjson.dumps({"text": text, "entities": entities}, option=orjson.OPT_APPEND_NEWLINE))
        # Run scan_misaligned_ner as a module for diagnostics (use --jsonl, not --input)
        try:
            subprocess.run([
                sys.executable, "-m", "webapp.parser.bots.scan_misaligned_ner", "--jsonl", misaligned_path
            ], check=True, cwd=PROJECT_ROOT, env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)})
        except Exception as e:
            print(f"[WARN] scan_misaligned_ner diagnostics failed: {e}")
        # Launch manual correction bot robustly as a module
        run_manual_correction_bot()
        print("[INFO] Please correct misalignments and rerun retraining.")
        sys.exit(2)

    # Use deduped_train_data for retraining
    retrain_sentence_transformer(deduped_train_data)
    retrain_spacy_ner_advanced(deduped_train_data, context_library)
    cluster_container_patterns()
    print("\n[SUMMARY] Table Structure Model Retraining Complete.")
    print("If you see repeated model save failures, close any file explorers or editors viewing the model directory.")
    print("If you see spaCy lexeme normalization warnings, you can ignore them for English. To suppress, install spacy-lookups-data and load the table if needed.")
    print("If you see spaCy entity alignment warnings, consider cleaning your training data or using the provided validation function.")
    gc.collect()
if __name__ == "__main__":
    main()

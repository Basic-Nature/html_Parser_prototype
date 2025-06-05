import os
import sqlite3
import json
import re
import datetime
from collections import Counter
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from ..utils.shared_logic import load_context_library
from ..Context_Integration.context_organizer import _safe_db_path
from ..config import CONTEXT_DB_PATH, MODEL_DIR

import spacy
from spacy.training import Example

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
    import time, shutil
    for attempt in range(retries):
        try:
            safe_model_save(model, model_save_path)
            return
        except Exception as e:
            print(f"[WARN] Model save failed (attempt {attempt+1}): {e}")
            time.sleep(2)
    # Try saving to a temp dir and moving
    tmp_path = model_save_path + "_tmp"
    try:
        safe_model_save(model, tmp_path)
        shutil.rmtree(model_save_path, ignore_errors=True)
        shutil.move(tmp_path, model_save_path)
        print(f"[INFO] Model saved via temp path workaround.")
    except Exception as e:
        print(f"[ERROR] Final model save failed: {e}")
        
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
        with open(safe_path, "r", encoding="utf-8") as f:
            for line in f:
                existing.add(line.strip())
    with open(safe_path, "a", encoding="utf-8") as f:
        for text, annots in new_data:
            entry = {
                "text": text,
                "entities": annots["entities"],
                "timestamp": datetime.datetime.now().isoformat()
            }
            line = json.dumps(entry, ensure_ascii=False)
            if line not in existing:
                f.write(line + "\n")

def save_training_data_jsonl(train_data, path="spacy_ner_train_data.jsonl"):
    # Save to the log/ directory at the project root
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../log"))
    os.makedirs(log_dir, exist_ok=True)
    filename = os.path.basename(path)
    filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', filename)
    safe_path = os.path.join(log_dir, filename)
    if not os.path.abspath(safe_path).startswith(log_dir):
        raise ValueError("Unsafe path detected for training data output!")
    with open(safe_path, "w", encoding="utf-8") as f:
        for text, annots in train_data:
            f.write(json.dumps({"text": text, "entities": annots["entities"]}, ensure_ascii=False) + "\n")
    print(f"Saved spaCy NER training data to {safe_path}")

def cluster_container_patterns(log_dir=None, n_clusters=5):
    """
    Cluster container HTML snippets and metadata for ML/NLP training.
    Prints cluster assignments and common selectors/classes/headings.
    """
    import glob
    import json
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    if log_dir is None:
        log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../log"))

    htmls = []
    meta = []
    for path in glob.glob(os.path.join(log_dir, "failed_container_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            entry = json.load(f)
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
    from collections import Counter
    counter = Counter()
    for _, annots in train_data:
        for _, _, label in annots["entities"]:
            counter[label] += 1
    print("Entity frequency:", counter)

def update_db_with_new_entities(new_entities, db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT,
            value TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    for entity_type, values in new_entities.items():
        for value in values:
            conn.execute(
                "INSERT OR IGNORE INTO entities (entity_type, value) VALUES (?, ?)",
                (entity_type, value)
            )
    conn.commit()
    conn.close()
    print(f"Updated DB with new entities: { {k: len(v) for k,v in new_entities.items()} }")

def load_spacy_ner_examples(jsonl_path):
    """
    Loads extra NER training examples from a JSONL file.
    Each line should be: {"text": ..., "entities": [[start, end, label], ...]}
    """
    examples = []
    if not os.path.exists(jsonl_path):
        return examples
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            entities = obj["entities"]
            examples.append((text, {"entities": entities}))
    return examples

def remove_overlapping_entities(entities):
    """
    Remove overlapping entities from a list of (start, end, label) tuples.
    Keeps the longest span first, then next non-overlapping, etc.
    """
    # Sort by start, then by longest span (descending)
    entities = sorted(entities, key=lambda x: (x[0], -(x[1] - x[0])))
    result = []
    last_end = -1
    for start, end, label in entities:
        if start >= last_end:
            result.append((start, end, label))
            last_end = end
        # else: skip this entity because it overlaps
    return result

def retrain_spacy_ner_advanced(confirmed_structures, context_library=None, model_save_path="fine_tuned_spacy_ner"):

    nlp = spacy.blank("en")
    try:
        from spacy.lookups import Lookups
        lookups = Lookups()
        lookups.add_table("lexeme_norm", spacy.lookups.load_lookups_data("en", tables=["lexeme_norm"]).get_table("lexeme_norm"))
        nlp.vocab.lookups = lookups
    except Exception as e:
        print("[spaCy] Could not load lexeme normalization table. You may ignore this if not using a supported language. Error:", e)

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
        all_parties.update([p for p in re.findall(r"\b(?:Democratic|Republican|Libertarian|Green|Independent|Conservative|Working Families|Write-in|Other)\b", " ".join(headers), re.IGNORECASE)])
        all_counties.update(context.get("known_counties", []))
        all_states.update(context.get("known_states", []))
        all_districts.update(context.get("known_districts", []))
        all_locations.update(context.get("known_cities", []))
        
        for header in headers:
            entities = auto_label_header(header, context)
            if entities:
                # PATCH: Remove overlapping entities before adding to train_data
                entities = remove_overlapping_entities(entities)
                train_data.append((header, {"entities": entities}))

    save_training_data_jsonl(train_data)
    entity_frequency_analysis(train_data)

    # Convert to spaCy Example objects
    examples = []
    for text, annots in train_data:
        doc = nlp.make_doc(text)
        # PATCH: Remove overlapping entities before creating Example (defensive)
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
            print(f"spaCy NER retraining epoch {i+1}, loss: {losses['ner']:.4f}")
        else:
            print(f"spaCy NER retraining epoch {i+1}, no NER loss reported.")

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
    db_path = _safe_db_path(CONTEXT_DB_PATH)
    conn = sqlite3.connect(db_path)
    cur = conn.execute(
        "SELECT contest_title, headers, context FROM table_structures WHERE confirmed_by_user = 1"
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "contest_title": row[0],
            "headers": json.loads(row[1]),
            "context": json.loads(row[2])
        }
        for row in rows
    ]

def run_manual_correction_bot():
    import subprocess
    # manual_correction_bot.py is in the same folder as this script
    script_path = os.path.join(os.path.dirname(__file__), "manual_correction_bot.py")
    subprocess.run(["python", script_path, "--fields", "tables", "--feedback", "--enhanced"])

def retrain_sentence_transformer(confirmed_structures, model_save_path=None):
    """
    Fine-tunes the SentenceTransformer model on confirmed structures.
    Loads the existing model for further training if present, otherwise starts from base.
    Always saves to the same folder (no timestamp).
    """
    train_examples = []
    for struct in confirmed_structures:
        contest_title = struct["contest_title"]
        headers = struct["headers"]
        for header in headers:
            train_examples.append(InputExample(texts=[contest_title, header], label=1.0))
    if not train_examples:
        print("No training examples found. Aborting retraining.")
        return

    # Always use the same folder for the model
    base_dir = MODEL_DIR if 'MODEL_DIR' in globals() else os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../model"))
    model_save_path = model_save_path or os.path.join(base_dir, "fine_tuned_table_headers")
    os.makedirs(model_save_path, exist_ok=True)

    # Load existing model for further training if present
    if os.path.exists(os.path.join(model_save_path, "config.json")):
        print(f"Loading existing model from {model_save_path} for further fine-tuning...")
        model = SentenceTransformer(model_save_path)
    else:
        print("No existing fine-tuned model found. Starting from base model.")
        model = SentenceTransformer("all-MiniLM-L6-v2")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
    train_loss = losses.CosineSimilarityLoss(model)

    print(f"Retraining SentenceTransformer on {len(train_examples)} pairs...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=10,
        show_progress_bar=True
    )

    safe_model_save(model, model_save_path)
    print(f"Fine-tuned model saved to: {model_save_path}")

def main():
    if os.getenv("REVIEW_WITH_MANUAL_BOT", "false").lower() == "true":
        run_manual_correction_bot()

    confirmed_structures = get_all_confirmed_structures()
    print(f"Found {len(confirmed_structures)} confirmed table structures.")

    # Log user feedback/corrections for ML ---
    feedback_log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../log/structure_feedback_log.jsonl"))
    os.makedirs(os.path.dirname(feedback_log_path), exist_ok=True)
    for struct in confirmed_structures:
        # Assume struct contains both original and corrected structure info if available
        old_structure_info = struct.get("original_structure", {})
        structure_info = struct.get("corrected_structure", {})
        headers = struct.get("headers", [])
        data = struct.get("sample_rows", [{}])
        with open(feedback_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "original_structure": old_structure_info,
                "corrected_structure": structure_info,
                "headers": headers,
                "sample_row": data[0] if data else {},
            }) + "\n")

    retrain_sentence_transformer(confirmed_structures)
    context_library = load_context_library()
    retrain_spacy_ner_advanced(confirmed_structures, context_library)
    cluster_container_patterns()
if __name__ == "__main__":
    main()
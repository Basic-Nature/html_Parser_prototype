import os
import sqlite3
import json
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from ..utils.shared_logic import load_context_library
from ..Context_Integration.context_organizer import _safe_db_path
from ..config import CONTEXT_DB_PATH

import spacy
from spacy.training import Example

ELECTION_ENTITY_LABELS = [
    "CONTEST", "CANDIDATE", "PARTY", "COUNTY", "STATE", "DISTRICT", "VOTE_METHOD",
    "BALLOT_TYPE", "PRECINCT", "TOTAL", "PERCENT", "YEAR", "ELECTION_TYPE", "OFFICE", "MISC"
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
    # Add more as needed
]

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

def auto_label_header(header: str, context: dict = None):
    labels = []
    for pattern, label in ENTITY_PATTERNS:
        for match in re.finditer(pattern, header, re.IGNORECASE):
            start, end = match.span()
            labels.append((start, end, label))
    if context:
        for label_type, values in [
            ("COUNTY", context.get("known_counties", [])),
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

def retrain_spacy_ner_advanced(confirmed_structures, context_library=None, model_save_path="fine_tuned_spacy_ner"):
    nlp = spacy.blank("en")
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

    for struct in confirmed_structures:
        headers = struct["headers"]
        context = struct.get("context", {})
        context.update({
            "known_counties": known_context.get("known_counties", []),
            "known_states": known_context.get("known_states", []),
            "known_candidates": known_context.get("known_candidates", []),
            "known_districts": known_context.get("known_districts", []),
        })
        # Auto-extract and aggregate entities for DB update
        context_candidates = extract_candidates_from_context(context)
        context["known_candidates"] = list(set(context.get("known_candidates", []) + context_candidates))
        all_candidates.update(context["known_candidates"])
        all_parties.update([p for p in re.findall(r"\b(?:Democratic|Republican|Libertarian|Green|Independent|Conservative|Working Families|Write-in|Other)\b", " ".join(headers), re.IGNORECASE)])
        all_counties.update(context.get("known_counties", []))
        all_states.update(context.get("known_states", []))
        all_districts.update(context.get("known_districts", []))
        for header in headers:
            entities = auto_label_header(header, context)
            if entities:
                train_data.append((header, {"entities": entities}))

    save_training_data_jsonl(train_data)
    entity_frequency_analysis(train_data)

    # Convert to spaCy Example objects
    examples = []
    for text, annots in train_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annots)
        examples.append(example)

    optimizer = nlp.begin_training()
    for i in range(10):
        losses = {}
        nlp.update(examples, drop=0.2, losses=losses)
        print(f"spaCy NER retraining epoch {i+1}, loss: {losses['ner']:.4f}")
    nlp.to_disk(model_save_path)
    print(f"Fine-tuned spaCy NER model saved to: {model_save_path}")

    # Update DB with new entities
    new_entities = {
        "CANDIDATE": list(all_candidates),
        "PARTY": list(all_parties),
        "COUNTY": list(all_counties),
        "STATE": list(all_states),
        "DISTRICT": list(all_districts),
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

def retrain_sentence_transformer(confirmed_structures, model_save_path="fine_tuned_table_headers"):
    train_examples = []
    for struct in confirmed_structures:
        contest_title = struct["contest_title"]
        headers = struct["headers"]
        for header in headers:
            train_examples.append(InputExample(texts=[contest_title, header], label=1.0))
    if not train_examples:
        print("No training examples found. Aborting retraining.")
        return

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
    model.save(model_save_path)
    print(f"Fine-tuned model saved to: {model_save_path}")

def main():
    if os.getenv("REVIEW_WITH_MANUAL_BOT", "false").lower() == "true":
        run_manual_correction_bot()

    confirmed_structures = get_all_confirmed_structures()
    print(f"Found {len(confirmed_structures)} confirmed table structures.")

    retrain_sentence_transformer(confirmed_structures)
    context_library = load_context_library()
    retrain_spacy_ner_advanced(confirmed_structures, context_library)

if __name__ == "__main__":
    main()
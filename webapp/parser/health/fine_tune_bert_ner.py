"""
BERT/RoBERTa NER Fine-Tuning Module for Election Data Extraction

This module implements HuggingFace Transformers fine-tuning for custom NER models
specialized in election data entity recognition (PERSON, ORG, CONTEST, PARTY, DISTRICT).

Usage:
    python -m webapp.parser.health.fine_tune_bert_ner

Environment Variables:
    BERT_NER_EPOCHS=3
    BERT_NER_BATCH_SIZE=16
    BERT_NER_LEARNING_RATE=2e-5
    BERT_NER_BASE_MODEL=dslim/bert-base-NER
"""
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import orjson
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from webapp.parser.config import LOG_DIR, MODEL_DIR
from webapp.parser.utils.db_utils import SessionLocal
from webapp.parser.utils.logger_singleton import logger

# Entity labels for election data
ELECTION_ENTITY_LABELS = [
    "O",  # Outside any entity
    "B-PERSON",  # Beginning of person name
    "I-PERSON",  # Inside person name
    "B-ORG",  # Beginning of organization
    "I-ORG",  # Inside organization
    "B-GPE",  # Beginning of geo-political entity
    "I-GPE",  # Inside geo-political entity
    "B-CONTEST",  # Beginning of contest name
    "I-CONTEST",  # Inside contest name
    "B-PARTY",  # Beginning of party name
    "I-PARTY",  # Inside party name
    "B-DISTRICT",  # Beginning of district
    "I-DISTRICT",  # Inside district
]

# Label to ID mapping
LABEL2ID = {label: idx for idx, label in enumerate(ELECTION_ENTITY_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def align_entity_spans_to_tokens(text: str, entities: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """
    Properly align entity character spans to token positions using BIO tagging.
    
    Args:
        text: Raw text string
        entities: List of {"start": int, "end": int, "label": str}
    
    Returns:
        (tokens, ner_tags) - parallel lists of tokens and BIO tags
    """
    # Tokenize preserving character positions
    tokens = []
    token_spans = []  # (start_char, end_char) for each token
    
    # Split on whitespace but track positions
    for match in re.finditer(r'\S+', text):
        tokens.append(match.group())
        token_spans.append((match.start(), match.end()))
    
    # Initialize tags as "O" (outside)
    ner_tags = ["O"] * len(tokens)
    
    # For each entity, find matching tokens using character offsets
    for entity in entities:
        ent_start = entity.get("start", 0)
        ent_end = entity.get("end", 0)
        label = entity.get("label", "")
        
        if not label or ent_start >= ent_end:
            continue
        
        first_token = True
        for i, (token_start, token_end) in enumerate(token_spans):
            # Check if token overlaps with entity span
            if token_start < ent_end and token_end > ent_start:
                # Token overlaps with entity
                if first_token:
                    ner_tags[i] = f"B-{label}"
                    first_token = False
                else:
                    ner_tags[i] = f"I-{label}"
    
    return tokens, ner_tags


def load_ner_data_from_db() -> List[Dict[str, Any]]:
    """Load verified NER training examples from PostgreSQL."""
    with SessionLocal() as session:
        results = session.execute(
            "SELECT text, entities FROM ner_training_data WHERE verified=TRUE ORDER BY created_at"
        ).fetchall()
        
        data = []
        for row in results:
            text = row[0]
            entities = row[1]  # JSONB: [{"start": 0, "end": 8, "label": "PERSON"}, ...]
            
            # Properly align entity spans to token positions using character offsets
            tokens, ner_tags = align_entity_spans_to_tokens(text, entities)
            
            data.append({"tokens": tokens, "ner_tags": ner_tags})
        
        logger.info(f"[BERT_NER] Loaded {len(data)} verified training examples from DB")
        return data


def load_ner_data_from_jsonl() -> List[Dict[str, Any]]:
    """Load NER training examples from JSONL logs (fallback if DB is empty)."""
    jsonl_path = os.path.join(LOG_DIR, "spacy_ner_train_data.jsonl")
    
    if not os.path.exists(jsonl_path):
        logger.warning(f"[BERT_NER] No JSONL training data found at {jsonl_path}")
        return []
    
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                example = orjson.loads(line)
                text = example.get("text", "")
                entities = example.get("entities", [])
                
                # Properly align entity spans to token positions using character offsets
                tokens, ner_tags = align_entity_spans_to_tokens(text, entities)
                
                data.append({"tokens": tokens, "ner_tags": ner_tags})
            except Exception as e:
                logger.warning(f"[BERT_NER] Failed to parse JSONL line: {e}")
                continue
    
    logger.info(f"[BERT_NER] Loaded {len(data)} training examples from JSONL")
    return data


def tokenize_and_align_labels(examples, tokenizer):
    """Tokenize text and align NER labels with subword tokens."""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128,
    )
    
    labels = []
    for i, label_seq in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Ignore padding
            elif word_idx != previous_word_idx:
                label_ids.append(LABEL2ID[label_seq[word_idx]])
            else:
                # For subword tokens, use I- tag if B- was used
                prev_label = label_seq[word_idx]
                if prev_label.startswith("B-"):
                    label_ids.append(LABEL2ID[prev_label.replace("B-", "I-")])
                else:
                    label_ids.append(LABEL2ID[prev_label])
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def fine_tune_bert_ner():
    """Fine-tune BERT/RoBERTa model for election NER."""
    # Load training data (prefer DB, fallback to JSONL)
    train_data = load_ner_data_from_db()
    if not train_data:
        logger.warning("[BERT_NER] No verified data in DB, falling back to JSONL")
        train_data = load_ner_data_from_jsonl()
    
    if not train_data:
        logger.error("[BERT_NER] No training data available. Aborting.")
        return
    
    # Split into train/validation (80/20)
    split_idx = int(len(train_data) * 0.8)
    train_examples = train_data[:split_idx]
    val_examples = train_data[split_idx:]
    
    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_dict({
        "tokens": [ex["tokens"] for ex in train_examples],
        "ner_tags": [ex["ner_tags"] for ex in train_examples],
    })
    val_dataset = Dataset.from_dict({
        "tokens": [ex["tokens"] for ex in val_examples],
        "ner_tags": [ex["ner_tags"] for ex in val_examples],
    })
    
    # Load base model and tokenizer
    base_model = os.environ.get("BERT_NER_BASE_MODEL", "dslim/bert-base-NER")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(ELECTION_ENTITY_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    
    # Tokenize datasets
    tokenized_train = train_dataset.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer),
        batched=True,
    )
    tokenized_val = val_dataset.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer),
        batched=True,
    )
    
    # Data collator for padding
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Training arguments
    output_dir = os.path.join(MODEL_DIR, "fine_tuned_bert_ner")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=int(os.environ.get("BERT_NER_EPOCHS", 3)),
        per_device_train_batch_size=int(os.environ.get("BERT_NER_BATCH_SIZE", 16)),
        per_device_eval_batch_size=16,
        learning_rate=float(os.environ.get("BERT_NER_LEARNING_RATE", 2e-5)),
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        push_to_hub=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    logger.info(f"[BERT_NER] Starting training on {len(train_examples)} examples...")
    trainer.train()
    
    # Save final model
    final_model_path = os.path.join(MODEL_DIR, "fine_tuned_bert_ner_production")
    trainer.save_model(final_model_path)
    logger.info(f"[BERT_NER] Fine-tuned model saved to {final_model_path}")
    
    # Evaluate
    results = trainer.evaluate()
    logger.info(f"[BERT_NER] Evaluation results: {results}")
    
    return final_model_path


if __name__ == "__main__":
    fine_tune_bert_ner()

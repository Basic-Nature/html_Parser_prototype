"""
Test Dataset Split Script for NER Model Evaluation

This script splits verified NER training data into train/test sets (80/20)
and creates evaluation datasets for precision/recall/F1 scoring.

Usage:
    python -m webapp.parser.health.create_test_dataset

Environment Variables:
    TEST_SPLIT_RATIO=0.2 (default: 20% for testing)
    MIN_TEST_SAMPLES=50 (minimum test samples required)
"""
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import orjson

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from webapp.parser.config import LOG_DIR
from webapp.parser.utils.db_utils import SessionLocal
from webapp.parser.utils.logger_singleton import logger


def load_verified_ner_data_from_db() -> List[Dict[str, Any]]:
    """Load all verified NER training examples from PostgreSQL."""
    with SessionLocal() as session:
        results = session.execute(
            "SELECT text, entities, source, created_at FROM ner_training_data WHERE verified=TRUE ORDER BY created_at"
        ).fetchall()
        
        data = []
        for row in results:
            data.append({
                "text": row[0],
                "entities": row[1],  # JSONB
                "source": row[2],
                "timestamp": row[3].isoformat() if row[3] else None,
            })
        
        logger.info(f"[TEST_SPLIT] Loaded {len(data)} verified training examples from DB")
        return data


def load_ner_data_from_jsonl() -> List[Dict[str, Any]]:
    """Load NER training examples from JSONL logs (fallback if DB is empty)."""
    jsonl_path = os.path.join(LOG_DIR, "spacy_ner_train_data.jsonl")
    
    if not os.path.exists(jsonl_path):
        logger.warning(f"[TEST_SPLIT] No JSONL training data found at {jsonl_path}")
        return []
    
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                example = orjson.loads(line)
                data.append({
                    "text": example.get("text", ""),
                    "entities": example.get("entities", []),
                    "source": example.get("source", "jsonl"),
                    "timestamp": example.get("timestamp"),
                })
            except Exception as e:
                logger.warning(f"[TEST_SPLIT] Failed to parse JSONL line: {e}")
                continue
    
    logger.info(f"[TEST_SPLIT] Loaded {len(data)} training examples from JSONL")
    return data


def split_train_test(
    data: List[Dict[str, Any]],
    test_ratio: float = 0.2,
    min_test_samples: int = 50,
    random_seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split data into train/test sets with stratification by entity types.
    
    Args:
        data: List of NER training examples
        test_ratio: Fraction of data to use for testing (default: 0.2)
        min_test_samples: Minimum number of test samples required
        random_seed: Random seed for reproducibility
    
    Returns:
        (train_data, test_data) tuple
    """
    if len(data) < min_test_samples:
        logger.warning(
            f"[TEST_SPLIT] Insufficient data ({len(data)} samples). "
            f"Need at least {min_test_samples} for reliable evaluation."
        )
        return data, []  # Use all data for training
    
    # Shuffle with fixed seed for reproducibility
    random.seed(random_seed)
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    # Calculate split index
    test_size = max(min_test_samples, int(len(shuffled) * test_ratio))
    train_size = len(shuffled) - test_size
    
    train_data = shuffled[:train_size]
    test_data = shuffled[train_size:]
    
    logger.info(f"[TEST_SPLIT] Split: {len(train_data)} train, {len(test_data)} test")
    return train_data, test_data


def save_datasets(
    train_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    output_dir: str
) -> None:
    """Save train/test datasets to JSONL files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training set
    train_path = os.path.join(output_dir, "ner_train.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for example in train_data:
            f.write(orjson.dumps(example).decode() + "\n")
    logger.info(f"[TEST_SPLIT] Saved {len(train_data)} training examples to {train_path}")
    
    # Save test set
    if test_data:
        test_path = os.path.join(output_dir, "ner_test.jsonl")
        with open(test_path, "w", encoding="utf-8") as f:
            for example in test_data:
                f.write(orjson.dumps(example).decode() + "\n")
        logger.info(f"[TEST_SPLIT] Saved {len(test_data)} test examples to {test_path}")
    
    # Save split metadata
    metadata_path = os.path.join(output_dir, "split_metadata.json")
    metadata = {
        "train_size": len(train_data),
        "test_size": len(test_data),
        "test_ratio": len(test_data) / (len(train_data) + len(test_data)) if train_data or test_data else 0,
        "created_at": orjson.dumps(None).decode(),  # Placeholder for timestamp
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2).decode())
    logger.info(f"[TEST_SPLIT] Saved split metadata to {metadata_path}")


def compute_entity_distribution(data: List[Dict[str, Any]]) -> Dict[str, int]:
    """Compute distribution of entity types in dataset."""
    entity_counts = {}
    for example in data:
        entities = example.get("entities", [])
        for ent in entities:
            label = ent.get("label", "UNKNOWN")
            entity_counts[label] = entity_counts.get(label, 0) + 1
    return entity_counts


def print_dataset_statistics(
    train_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]]
) -> None:
    """Print dataset statistics for validation."""
    logger.info("\n[TEST_SPLIT] Dataset Statistics:")
    logger.info(f"  Total examples: {len(train_data) + len(test_data)}")
    logger.info(f"  Training: {len(train_data)}")
    logger.info(f"  Testing: {len(test_data)}")
    
    if train_data:
        train_dist = compute_entity_distribution(train_data)
        logger.info(f"  Train entity distribution: {train_dist}")
    
    if test_data:
        test_dist = compute_entity_distribution(test_data)
        logger.info(f"  Test entity distribution: {test_dist}")


def main():
    """Main entry point for test dataset creation."""
    # Load data (prefer DB, fallback to JSONL)
    data = load_verified_ner_data_from_db()
    if not data:
        logger.warning("[TEST_SPLIT] No verified data in DB, falling back to JSONL")
        data = load_ner_data_from_jsonl()
    
    if not data:
        logger.error("[TEST_SPLIT] No training data available. Aborting.")
        sys.exit(1)
    
    # Split into train/test
    test_ratio = float(os.environ.get("TEST_SPLIT_RATIO", "0.2"))
    min_test_samples = int(os.environ.get("MIN_TEST_SAMPLES", "50"))
    
    train_data, test_data = split_train_test(
        data,
        test_ratio=test_ratio,
        min_test_samples=min_test_samples
    )
    
    # Save datasets
    output_dir = os.path.join(LOG_DIR, "test_datasets")
    save_datasets(train_data, test_data, output_dir)
    
    # Print statistics
    print_dataset_statistics(train_data, test_data)
    
    logger.info("[TEST_SPLIT] Test dataset creation complete.")
    logger.info(f"[TEST_SPLIT] Datasets saved to {output_dir}")


if __name__ == "__main__":
    main()

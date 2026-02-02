"""
health_config.py

Configuration and constants for health monitoring system.
Defines thresholds, model paths, and integrity check parameters.
"""
from pathlib import Path

from ..config import LOG_DIR, MODEL_DIR, PROJECT_ROOT

# === Integrity Monitor Settings ===

# Download cache settings
DOWNLOAD_TTL_SECONDS = 15 * 60  # 15 minutes
DOWNLOAD_CACHE_QUOTA = 50 * 1024 * 1024  # 50MB
SESSION_STORAGE_QUOTA = 5 * 1024 * 1024  # 5MB

# Neural network settings
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for high-priority flags
HEALTH_SCORE_THRESHOLD_HIGH = 0.7  # Score >= this is high priority
HEALTH_SCORE_THRESHOLD_MEDIUM = 0.5  # Score >= this is medium priority

# Context library settings
CONTEXT_LIBRARY_MAX_INTEGRITY_ENTRIES = 100  # Keep last N integrity checks
CONTEXT_LIBRARY_PATH = PROJECT_ROOT / "webapp" / "parser" / "Context_Integration" / "Context_Library" / "context_library.json"

# Model paths
INTEGRITY_MODEL_PATH = Path(MODEL_DIR) / "integrity_model.pt"
TABLE_STRUCTURE_MODEL_PATH = Path(MODEL_DIR) / "table_structure.pt"

# === HuggingFace Model Settings ===

# Preferred models (privacy-focused, local-first)
HUGGINGFACE_MODELS = {
    "sentence_embedding": "sentence-transformers/all-MiniLM-L6-v2",
    "ner": "dslim/bert-base-NER",
    "classification": "distilbert-base-uncased",
    "zero_shot": "facebook/bart-large-mnli"
}

# Fallback to offline mode if no internet
HUGGINGFACE_OFFLINE_MODE = False

# === Context-Aware State Checks ===

# Critical fields that must be present for high-confidence validation
CRITICAL_FIELDS = ["state", "county", "contest", "year"]

# Suspicious patterns that trigger integrity flags
SUSPICIOUS_PATTERNS = [
    r"test",
    r"demo",
    r"fake",
    r"sample",
    r"invalid",
    r"placeholder",
    r"lorem\s+ipsum"
]

# Risk factor weights for priority calculation
RISK_WEIGHTS = {
    "missing_state": 0.15,
    "missing_county": 0.12,
    "missing_contest": 0.10,
    "missing_year": 0.08,
    "suspicious_keyword": 0.20,
    "nlp_confidence_low": 0.10,
    "entity_count_low": 0.05,
    "flag_count_high": 0.20
}

# === Logging & Monitoring ===

INTEGRITY_LOG_PATH = Path(LOG_DIR) / "integrity_monitor.jsonl"
INTEGRITY_LOG_MAX_SIZE = 10 * 1024 * 1024  # 10MB max, then rotate
INTEGRITY_LOG_RETENTION_DAYS = 30

# Health router orchestration timeouts
CORRECTION_BOT_TIMEOUT = 600  # 10 minutes
MIGRATION_TIMEOUT = 1800  # 30 minutes
RETRAINING_TIMEOUT = 3600  # 1 hour

# === Feature Engineering ===

# Feature vector dimensions for neural network
FEATURE_VECTOR_DIM = 128

# Feature composition (must sum to FEATURE_VECTOR_DIM)
FEATURE_BINARY_DIM = 32  # Binary flags (field presence, etc.)
FEATURE_NUMERIC_DIM = 32  # Numeric values (counts, scores, etc.)
FEATURE_EMBEDDING_DIM = 64  # Text embeddings

# Normalization bounds
MAX_FLAG_COUNT = 10
MAX_ENTITY_COUNT = 10
MAX_RISK_FACTOR_COUNT = 5

# === Deduplication Settings ===

# Cache key format: "{principal}:{file_name}"
CACHE_KEY_SEPARATOR = ":"

# LRU eviction target (percentage of quota to maintain after eviction)
CACHE_EVICTION_TARGET_PERCENT = 0.8

# Session sharing: allow download cache sharing across sessions with same principal
ENABLE_CROSS_SESSION_DEDUPLICATION = True

# === Export / Persistence ===

# Fields to include in context library integrity exports
INTEGRITY_EXPORT_FIELDS = [
    "session_id",
    "timestamp",
    "health_score",
    "confidence",
    "priority",
    "context_summary",
    "nlp_entities",
    "risk_factors",
    "recommendations"
]

# Maximum JSON export size before compression
MAX_JSON_EXPORT_SIZE = 1024 * 1024  # 1MB

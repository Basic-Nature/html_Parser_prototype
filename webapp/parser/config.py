"""
Central configuration module for the Smart Elections Parser Webapp.

- All constants and environment variable lookups are defined here.
- This file is imported by all modules needing configuration, paths, or environment-based toggles.
- No Flask app or runtime logic should be placed here—only configuration and helpers.
"""

import os
import threading
from pathlib import Path
import orjson
import urllib.parse
import psycopg2
from sqlalchemy import create_engine
from .utils.logger_singleton import logger
from azure.identity import DefaultAzureCredential
try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    # python-dotenv not installed (e.g., on Azure), skip loading .env
    pass


# === Project Structure & Paths ===

# Absolute path to the project root (parent of 'webapp')
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Absolute path to the 'webapp' directory
BASE_DIR = PROJECT_ROOT / "webapp"

# Path to the parser directory (where config.py is located)
PARSER_DIR = BASE_DIR / "parser"

# Path to the SQLite context database (used for local context storage)
CONTEXT_DB_PATH = PARSER_DIR / "Context_Integration" / "Context_Library" / "context_elections.db"

# Path to the context library JSON file (used for ML/NLP and context enrichment)
CONTEXT_LIBRARY_PATH = PARSER_DIR / "Context_Integration" / "Context_Library" / "context_library.json"
CONTEXT_LIBRARY_DIR = CONTEXT_LIBRARY_PATH.parent

# Directory for all vocabularies used by ML/NLP models
VOCAB_DIR = PARSER_DIR / "Context_Integration" / "vocab"
VOCAB_DIR.mkdir(parents=True, exist_ok=True)

# Input and output directories at the project root
INPUT_DIR = PROJECT_ROOT / "input"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR = PROJECT_ROOT / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Path to the download manifest file (used for tracking downloads)
DOWNLOAD_MANIFEST = INPUT_DIR / ".download_manifest.jsonl"

# Path to the list of URLs to process
URL_LIST_FILE = PARSER_DIR / "urls.txt"
SEED_URLS_IF_EMPTY = os.environ.get("SEED_URLS_IF_EMPTY", "true").lower() in ("1","true","yes")
if not URL_LIST_FILE.exists():
    with open(URL_LIST_FILE, "w", encoding="utf-8") as f:
        f.write("# Add your URLs here, one per line.\n")
elif URL_LIST_FILE.stat().st_size == 0 and SEED_URLS_IF_EMPTY:
    # Only seed if env allows; otherwise leave truly empty so we don't “overwrite”
    with open(URL_LIST_FILE, "w", encoding="utf-8") as f:
        f.write("# Add your URLs here, one per line.\n")

# Path to the file tracking processed URLs (used for deduplication/caching)
PROCESSED_URLS_FILE = CONTEXT_DB_PATH.parent / ".processed_urls"

# Log and cache directories (inside Context_Library for consistency)
LOG_DIR = CONTEXT_LIBRARY_DIR / "log"
CACHE_DIR = CONTEXT_LIBRARY_DIR / "cache"
LOG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Run history (NDJSON lines: start/end events for parser runs)
RUN_HISTORY_FILE = LOG_DIR / "run_history.ndjson"
# Ensure file exists (optional – create empty if missing)
if not RUN_HISTORY_FILE.exists():
    try:
        RUN_HISTORY_FILE.touch()
    except Exception:
        pass

# Path to the cache file for processed URLs (used for deduplication)
OUTPUT_CACHE = CACHE_DIR / "output_cache.json"

# Path to the disk cache for embeddings (used for caching ML/NLP embeddings)
DISK_CACHE_PATH = CACHE_DIR / "embedding_disk_cache.pkl"
MISSING_LOG_PATH = LOG_DIR / "missing_embeddings_log.jsonl"

# Directory for storing ML/NLP models (defaults to parent of webapp)
MODEL_DIR = PROJECT_ROOT

# Path to the context cache file (used for caching context lookups)
CONTEXT_CACHE_PATH = CACHE_DIR / "context_cache.json"

# === Database Configuration ===

DEPLOY_ENV = os.environ.get("DEPLOY_ENV", "").lower()  # "azure" or "local"

if DEPLOY_ENV == "azure":
    POSTGRES_USER_RAW = os.environ.get("POSTGRES_USER", "")
    POSTGRES_PASSWORD_RAW = os.environ.get("POSTGRES_PASSWORD", "")
    POSTGRES_DB = os.environ.get("POSTGRES_DB", "")
    POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "")
    POSTGRES_PORT = os.environ.get("POSTGRES_PORT") or "5432"
    # URL-encoded ONLY for building DSNs that need escaping
    POSTGRES_USER = urllib.parse.quote_plus(POSTGRES_USER_RAW)
    POSTGRES_PASSWORD = urllib.parse.quote_plus(POSTGRES_PASSWORD_RAW)
    POSTGRES_URL = (
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    POSTGRES_SERVICE_NAME = None
else:
    POSTGRES_USER_RAW = os.environ.get("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD_RAW = os.environ.get("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB = os.environ.get("POSTGRES_DB", "postgres")
    POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.environ.get("POSTGRES_PORT") or "5432"
    POSTGRES_USER = urllib.parse.quote_plus(POSTGRES_USER_RAW)
    POSTGRES_PASSWORD = urllib.parse.quote_plus(POSTGRES_PASSWORD_RAW)
    POSTGRES_URL = (
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    POSTGRES_SERVICE_NAME = os.environ.get("POSTGRES_SERVICE_NAME", "postgresql-x64-17")
    
# Auth mode: "password" (default) or "aad"
POSTGRES_AUTH = os.environ.get("POSTGRES_AUTH", "password").lower()
POSTGRES_AAD_USER = os.environ.get("POSTGRES_AAD_USER")  # DB role name created for your MI/user
 
# === LLM & Pipeline Configuration ===

# LLM provider and model (used for OpenAI or other LLM integrations)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4-turbo")
LLM_API_KEY = os.environ.get("LLM_API_KEY")
LLM_SYSTEM_PROMPT = os.environ.get("LLM_SYSTEM_PROMPT")
LLM_EXTRA_INSTRUCTIONS = os.environ.get("LLM_EXTRA_INSTRUCTIONS")
USER_NAME = os.environ.get("USER", "system")

# Path to table detection model (for ML/NLP table structure tasks)
TABLE_MODEL_PATH = os.environ.get("TABLE_MODEL_PATH", str(MODEL_DIR / "table_detector.pt"))

# === Feature Toggles & Pipeline Options ===

# Feature toggles (enable/disable features via environment variables)
ENABLE_ENHANCED = os.environ.get("ENABLE_ENHANCED", "true").lower() == "true"
CORRECTION_MODE = os.environ.get("CORRECTION_MODE", "feedback").lower()
INTEGRITY_CHECK = os.environ.get("INTEGRITY_CHECK", "false").lower() == "true"
UPDATE_DB = os.environ.get("UPDATE_DB", "true").lower() == "true"
FILTER_CONTEXT_KEY = os.environ.get("FILTER_CONTEXT_KEY")
FILTER_VALUE = os.environ.get("FILTER_VALUE")
FIELDS = os.environ.get("FIELDS")
CONTEXT_PATH = os.environ.get("CONTEXT_PATH")
LOG_DIR_ENV = os.environ.get("LOG_DIR")
DRY_RUN = os.environ.get("DRY_RUN", "false").lower() == "true"
NO_COORDINATOR = os.environ.get("NO_COORDINATOR", "false").lower() == "true"
NO_ORGANIZER = os.environ.get("NO_ORGANIZER", "false").lower() == "true"
BATCH_MODE = os.environ.get("BATCH_MODE", "false").lower() == "true"
FAST_MODE = os.environ.get("FAST_MODE", "false").lower() == "true"
FLUSH_CACHE = os.environ.get("FLUSH_CACHE", "false").lower() == "true"
CACHE_EXPIRE_DAYS = os.environ.get("CACHE_EXPIRE_DAYS")
EXPORT_AUDIT_LOG = os.environ.get("EXPORT_AUDIT_LOG")
REST_API = os.environ.get("REST_API", "false").lower() == "true"
SELF_HEAL = os.environ.get("SELF_HEAL", "false").lower() == "true"
MAX_RETRIES = os.environ.get("MAX_RETRIES")
COOLDOWN = os.environ.get("COOLDOWN")
DB_PATH = os.environ.get("DB_PATH")
ENABLE_SEGMENT_LABEL_PROMPT = os.environ.get("ENABLE_SEGMENT_LABEL_PROMPT", "true").lower() == "true"
DEFAULT_CAPTCHA_TIMEOUT = int(os.environ.get("CAPTCHA_TIMEOUT", "300"))
DISABLE_HTML_FALLBACK = os.environ.get("DISABLE_HTML_FALLBACK", "0").lower() in ("1", "true", "yes")

ENABLE_OCR = os.environ.get("ENABLE_OCR", "true").lower() in ("1","true","yes")
# Force OCR even if PyMuPDF returns text (for debugging tricky PDFs)
ENABLE_OCR_FORCE = os.environ.get("ENABLE_OCR_FORCE", "false").lower() in ("1","true","yes")
# Optional binaries (Windows)
POPPLER_PATH = os.environ.get("POPPLER_PATH") or None
TESSERACT_CMD = os.environ.get("TESSERACT_CMD") or None
# OCR debug output folder
OCR_DEBUG_DIR = OUTPUT_DIR / "ocr_debug"
OCR_DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# Cache options
CACHE_PROCESSED_URLS = os.environ.get("CACHE_PROCESSED", "true").lower() == "true"
CACHE_LOCK = threading.Lock()
CACHE_RESET = os.environ.get("CACHE_RESET", "false").lower() == "true"

# Headless browser and timeout settings
HEADLESS_DEFAULT = os.environ.get("HEADLESS", "true").lower() == "true"
TIMEOUT_SEC = int(os.environ.get("CAPTCHA_TIMEOUT", "300"))
INCLUDE_TIMESTAMP_IN_FILENAME = os.environ.get("TIMESTAMP_IN_FILENAME", "true").lower() == "true"
ENABLE_PARALLEL = os.environ.get("ENABLE_PARALLEL", "false").lower() == "true"
ENABLE_AI_ANALYSIS = os.environ.get("ENABLE_AI_ANALYSIS", "false").lower() == "true"
ENABLE_REALTIME_STREAM = os.environ.get("ENABLE_REALTIME_STREAM", "false").lower() == "true"
FORCE_PARSE_INPUT_FILE = os.environ.get("FORCE_PARSE_INPUT_FILE", "false").lower() == "true"
FORCE_PARSE_FORMAT = os.environ.get("FORCE_PARSE_FORMAT", "").strip().lower()
MAX_URLS_DISPLAYED = os.environ.get("MAX_URLS_DISPLAYED")

# Pipeline worker and error thresholds (used by web_pipeline.py and related modules)
PIPELINE_MAX_WORKERS = int(os.environ.get("PIPELINE_MAX_WORKERS", 2))
PIPELINE_MAX_ERRORS = int(os.environ.get("PIPELINE_MAX_ERRORS", 5))
PIPELINE_HEARTBEAT_INTERVAL = int(os.environ.get("PIPELINE_HEARTBEAT_INTERVAL", 10))

# User feedback toggle (used for enabling feedback prompts in output_utils, etc.)
ENABLE_USER_FEEDBACK = os.environ.get("ENABLE_USER_FEEDBACK", "false").lower() == "true"

# === ML/NLP Training Configuration ===

# Sentence-BERT (SBERT) training parameters
SBERT_EPOCHS = int(os.environ.get("SBERT_EPOCHS", 1))
SBERT_BATCH_SIZE = int(os.environ.get("SBERT_BATCH_SIZE", 8))

# spaCy NER training parameters
SPACY_NER_EPOCHS = int(os.environ.get("SPACY_NER_EPOCHS", 10))
SPACY_NER_PATIENCE = int(os.environ.get("SPACY_NER_PATIENCE", 3))
SPACY_NER_MIN_DELTA = float(os.environ.get("SPACY_NER_MIN_DELTA", 0.01))
SPACY_NER_BATCH_SIZE = int(os.environ.get("SPACY_NER_BATCH_SIZE", 32))

# Manual review bot toggle (used to enable manual correction bot in retraining)
REVIEW_WITH_MANUAL_BOT = os.environ.get("REVIEW_WITH_MANUAL_BOT", "false").lower() == "true"

# Data Framework API endpoint (configurable)
DATA_API_URL = os.environ.get("DATA_API_URL", "/api/warehouse_election_results")

# === Helper Functions ===

def get_subprocess_env():
    """
    Returns a copy of os.environ with PYTHONPATH set to PROJECT_ROOT.
    Use this for subprocesses that need to import project modules.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    return env

def get_supported_formats():
    """
    Returns a list of supported file formats for input.
    Priority:
      1. SUPPORTED_FORMATS env variable (comma-separated, e.g. ".csv,.json")
      2. context_library.json 'supported_formats' key (if context_library is a dict)
      3. Default: [".json", ".csv", ".pdf"]
    """
    env_formats = os.environ.get("SUPPORTED_FORMATS")
    if env_formats:
        return [
            ext if ext.startswith('.') else f'.{ext}'
            for ext in env_formats.split(",")
        ]
    try:
        if CONTEXT_LIBRARY_PATH.exists():
            with open(CONTEXT_LIBRARY_PATH, "rb") as f:
                context_library = orjson.loads(f.read())
            if isinstance(context_library, dict):
                formats_raw = context_library.get("supported_formats", [".json", ".csv", ".pdf"])
                if isinstance(formats_raw, list):
                    return formats_raw
                elif isinstance(formats_raw, str):
                    import json
                    parsed = json.loads(formats_raw)
                    return parsed if isinstance(parsed, list) else [".json", ".csv", ".pdf"]
    except Exception:
        pass  # Optionally log error here
    return [".json", ".csv", ".pdf"]

SUPPORTED_FORMATS = [ext for ext in get_supported_formats() if ext.lower() not in [".html", "html"]]

def get_sqlalchemy_engine():
    """
    Returns an SQLAlchemy engine that uses:
    - Password auth when POSTGRES_AUTH=password
    - Entra (AAD) token when POSTGRES_AUTH=aad
    Falls back to password if AAD fails and creds exist (unless DB_STRICT_AAD=true).
    """
    strict_aad = os.environ.get("DB_STRICT_AAD", "false").lower() in ("1","true","yes")
    connect_kwargs = {"connect_timeout": 10, "application_name": "ballotlens-webapp"}

    if POSTGRES_AUTH == "aad":
        if not POSTGRES_AAD_USER:
            logger.error("[DB] AAD mode requested but POSTGRES_AAD_USER is empty.")
            if strict_aad:
                raise RuntimeError("AAD auth requested but POSTGRES_AAD_USER missing")
        def _connect_with_aad():
            cred = DefaultAzureCredential(exclude_interactive_browser_credential=True)
            tok = cred.get_token("https://ossrdbms-aad.database.windows.net/.default")
            logger.info(f"[DB] Got AAD token (exp={tok.expires_on}) for host={POSTGRES_HOST}, db={POSTGRES_DB}, user={POSTGRES_AAD_USER}")
            return psycopg2.connect(
                host=POSTGRES_HOST,
                dbname=POSTGRES_DB,
                user=POSTGRES_AAD_USER,
                password=tok.token,
                port=int(POSTGRES_PORT or 5432),
                sslmode="require",
                **connect_kwargs
            )
        try:
            logger.info(f"[DB] Connecting via AAD user={POSTGRES_AAD_USER} host={POSTGRES_HOST} db={POSTGRES_DB}")
            return create_engine(
                "postgresql+psycopg2://",
                creator=_connect_with_aad,
                pool_pre_ping=True,
                future=True
            )
        except Exception as e:
            logger.error(f"[DB][AAD] Connection failed: {e}")
            if strict_aad:
                raise
            if POSTGRES_USER_RAW and POSTGRES_PASSWORD_RAW:
                logger.warning("[DB][AAD] Falling back to password auth.")
                return create_engine(
                    POSTGRES_URL,
                    pool_pre_ping=True,
                    future=True,
                    connect_args=connect_kwargs
                )
            raise

    # Password path
    logger.info(f"[DB] Connecting via password user={POSTGRES_USER_RAW} host={POSTGRES_HOST} db={POSTGRES_DB}")
    return create_engine(
        POSTGRES_URL,
        pool_pre_ping=True,
        future=True,
        connect_args=connect_kwargs
    )

__all__ = [
    # Core paths
    "PROJECT_ROOT","BASE_DIR","PARSER_DIR","INPUT_DIR","OUTPUT_DIR","UPLOADS_DIR",
    "CONTEXT_DB_PATH","CONTEXT_LIBRARY_PATH","CONTEXT_LIBRARY_DIR","VOCAB_DIR",
    "DOWNLOAD_MANIFEST","URL_LIST_FILE","PROCESSED_URLS_FILE","LOG_DIR","CACHE_DIR",
    "RUN_HISTORY_FILE","OUTPUT_CACHE","DISK_CACHE_PATH","MISSING_LOG_PATH",
    "MODEL_DIR","CONTEXT_CACHE_PATH",

    # DB settings
    "DEPLOY_ENV","POSTGRES_USER_RAW","POSTGRES_PASSWORD_RAW","POSTGRES_DB",
    "POSTGRES_HOST","POSTGRES_PORT","POSTGRES_USER","POSTGRES_PASSWORD",
    "POSTGRES_URL","POSTGRES_SERVICE_NAME", "POSTGRES_AUTH",
    "POSTGRES_AAD_USER","get_sqlalchemy_engine",

    # LLM
    "LLM_PROVIDER","LLM_MODEL","LLM_API_KEY","LLM_SYSTEM_PROMPT",
    "LLM_EXTRA_INSTRUCTIONS","USER_NAME","TABLE_MODEL_PATH",

    # Feature toggles / options
    "ENABLE_ENHANCED","CORRECTION_MODE","INTEGRITY_CHECK","UPDATE_DB",
    "FILTER_CONTEXT_KEY","FILTER_VALUE","FIELDS","CONTEXT_PATH","LOG_DIR_ENV",
    "DRY_RUN","NO_COORDINATOR","NO_ORGANIZER","BATCH_MODE","FAST_MODE",
    "FLUSH_CACHE","CACHE_EXPIRE_DAYS","EXPORT_AUDIT_LOG","REST_API","SELF_HEAL",
    "MAX_RETRIES","COOLDOWN","DB_PATH","ENABLE_SEGMENT_LABEL_PROMPT",
    "DEFAULT_CAPTCHA_TIMEOUT", "CACHE_PROCESSED_URLS",
    "CACHE_LOCK","CACHE_RESET","HEADLESS_DEFAULT","TIMEOUT_SEC",
    "INCLUDE_TIMESTAMP_IN_FILENAME","ENABLE_PARALLEL","ENABLE_AI_ANALYSIS",
    "ENABLE_REALTIME_STREAM","FORCE_PARSE_INPUT_FILE","FORCE_PARSE_FORMAT",
    "MAX_URLS_DISPLAYED","PIPELINE_MAX_WORKERS","PIPELINE_MAX_ERRORS",
    "PIPELINE_HEARTBEAT_INTERVAL","ENABLE_USER_FEEDBACK", "DISABLE_HTML_FALLBACK",

    # Training params
    "SBERT_EPOCHS","SBERT_BATCH_SIZE","SPACY_NER_EPOCHS","SPACY_NER_PATIENCE",
    "SPACY_NER_MIN_DELTA","SPACY_NER_BATCH_SIZE","REVIEW_WITH_MANUAL_BOT",

    # Formats
    "SUPPORTED_FORMATS","SEED_URLS_IF_EMPTY",

    # Helpers
    "get_subprocess_env","get_supported_formats",
    
    # OCR paths
    "ENABLE_OCR","ENABLE_OCR_FORCE","POPPLER_PATH","TESSERACT_CMD","OCR_DEBUG_DIR",
]

# === END OF CONFIGURATION ===

# -------------------------------------------------------------------------
# ENVIRONMENT VARIABLES (Settable in Azure, .env, or OS environment)
# -------------------------------------------------------------------------
# PROJECT_ROOT, BASE_DIR, CONTEXT_DB_PATH, CONTEXT_LIBRARY_PATH, VOCAB_DIR, INPUT_DIR, OUTPUT_DIR, URL_LIST_FILE, PROCESSED_URLS_FILE, LOG_DIR, CACHE_DIR, MODEL_DIR, CONTEXT_CACHE_PATH
# (These are computed, not settable.)

# Settable environment variables:
# - POSTGRES_USER
# - POSTGRES_PASSWORD
# - POSTGRES_DB
# - POSTGRES_HOST
# - POSTGRES_PORT
# - POSTGRES_URL
# - POSTGRES_SERVICE_NAME
# - LLM_PROVIDER
# - LLM_MODEL
# - LLM_API_KEY
# - LLM_SYSTEM_PROMPT
# - LLM_EXTRA_INSTRUCTIONS
# - USER
# - TABLE_MODEL_PATH
# - ENABLE_ENHANCED
# - CORRECTION_MODE
# - INTEGRITY_CHECK
# - UPDATE_DB
# - FILTER_CONTEXT_KEY
# - FILTER_VALUE
# - FIELDS
# - CONTEXT_PATH
# - LOG_DIR
# - DRY_RUN
# - NO_COORDINATOR
# - NO_ORGANIZER
# - BATCH_MODE
# - FAST_MODE
# - FLUSH_CACHE
# - CACHE_EXPIRE_DAYS
# - EXPORT_AUDIT_LOG
# - REST_API
# - SELF_HEAL
# - MAX_RETRIES
# - COOLDOWN
# - DB_PATH
# - ENABLE_SEGMENT_LABEL_PROMPT
# - CAPTCHA_TIMEOUT
# - ENABLE_OCR
# - CACHE_PROCESSED
# - CACHE_RESET
# - HEADLESS
# - TIMESTAMP_IN_FILENAME
# - ENABLE_PARALLEL
# - ENABLE_AI_ANALYSIS
# - ENABLE_REALTIME_STREAM
# - FORCE_PARSE_INPUT_FILE
# - FORCE_PARSE_FORMAT
# - MAX_URLS_DISPLAYED
# - PIPELINE_MAX_WORKERS
# - PIPELINE_MAX_ERRORS
# - PIPELINE_HEARTBEAT_INTERVAL
# - ENABLE_USER_FEEDBACK
# - SBERT_EPOCHS
# - SBERT_BATCH_SIZE
# - SPACY_NER_EPOCHS
# - SPACY_NER_PATIENCE
# - SPACY_NER_MIN_DELTA
# - SPACY_NER_BATCH_SIZE
# - REVIEW_WITH_MANUAL_BOT
# - SUPPORTED_FORMATS

# -------------------------------------------------------------------------
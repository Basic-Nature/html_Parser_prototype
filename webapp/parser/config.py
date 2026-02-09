"""
Central configuration module for the Smart Elections Parser Webapp.

- All constants and environment variable lookups are defined here.
- This file is imported by all modules needing configuration, paths, or environment-based toggles.
- No Flask app or runtime logic should be placed here—only configuration and helpers.
"""
import os
import threading
import urllib.parse
from pathlib import Path

import orjson
import psycopg2
from azure.identity import DefaultAzureCredential
from sqlalchemy import create_engine

from .utils.logger_singleton import logger

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

# Static quick-copy directory (session-scoped assets)
QUICK_COPY_DIR = BASE_DIR / "static" / "quick_copy"
QUICK_COPY_DIR.mkdir(parents=True, exist_ok=True)

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

# === Verified Data Database Configuration (DL1/DL2 Classification) ===
# Separate database for storing verified election data with full audit trail
VERIFIED_DATA_DB_HOST = os.environ.get("VERIFIED_DATA_DB_HOST", os.environ.get("POSTGRES_HOST", "localhost"))
VERIFIED_DATA_DB_PORT = os.environ.get("VERIFIED_DATA_DB_PORT", "5432")
VERIFIED_DATA_DB_NAME = os.environ.get("VERIFIED_DATA_DB_NAME", "verified_data")
VERIFIED_DATA_DB_USER = os.environ.get("VERIFIED_DATA_DB_USER", os.environ.get("POSTGRES_USER", "postgres"))
VERIFIED_DATA_DB_PASSWORD = os.environ.get("VERIFIED_DATA_DB_PASSWORD", os.environ.get("POSTGRES_PASSWORD", ""))
 
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
try:
    BATCH_MAX_WORKERS = max(1, int(os.environ.get("BATCH_MAX_WORKERS", "2")))
except (TypeError, ValueError):
    BATCH_MAX_WORKERS = 2
REST_API = os.environ.get("REST_API", "false").lower() == "true"
SELF_HEAL = os.environ.get("SELF_HEAL", "false").lower() == "true"
MAX_RETRIES = os.environ.get("MAX_RETRIES")
COOLDOWN = os.environ.get("COOLDOWN")
DB_PATH = os.environ.get("DB_PATH")
ENABLE_SEGMENT_LABEL_PROMPT = os.environ.get("ENABLE_SEGMENT_LABEL_PROMPT", "true").lower() == "true"
DEFAULT_CAPTCHA_TIMEOUT = int(os.environ.get("CAPTCHA_TIMEOUT", "300"))
DISABLE_HTML_FALLBACK = os.environ.get("DISABLE_HTML_FALLBACK", "0").lower() in ("1", "true", "yes")

ENABLE_OCR = os.environ.get("ENABLE_OCR", "true").lower() in ("1","true","yes")

ENABLE_CAMELOT = True
CAMELOT_MIN_SCORE = 0.9
CAMELOT_HYBRID_FILL = True
CAMELOT_MERGE_COMPAT = True
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

# === Security & Safety Limits ===
try:
    MAX_UPLOAD_SIZE_MB = max(1, int(os.environ.get("MAX_UPLOAD_SIZE_MB", "100")))
except Exception:
    MAX_UPLOAD_SIZE_MB = 100
MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
try:
    MAX_PDF_PAGES = max(1, int(os.environ.get("MAX_PDF_PAGES", "200")))
except Exception:
    MAX_PDF_PAGES = 200
try:
    MAX_CSV_ROWS = max(1000, int(os.environ.get("MAX_CSV_ROWS", "100000")))
except Exception:
    MAX_CSV_ROWS = 100000
try:
    MAX_XLSX_BYTES = max(1, int(os.environ.get("MAX_XLSX_BYTES", str(50 * 1024 * 1024))))
except Exception:
    MAX_XLSX_BYTES = 50 * 1024 * 1024
try:
    MAX_DOWNLOAD_BYTES = max(1, int(os.environ.get("MAX_DOWNLOAD_BYTES", str(100 * 1024 * 1024))))
except Exception:
    MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024

URL_ALLOWLIST_SUFFIXES = [
    s.strip().lower()
    for s in os.environ.get("URL_ALLOWLIST_SUFFIXES", ".gov,.us").split(",")
    if s.strip()
]
URL_ALLOWLIST_HOSTS = [
    s.strip().lower()
    for s in os.environ.get("URL_ALLOWLIST_HOSTS", "").split(",")
    if s.strip()
]
ALLOW_GOOGLE_DOCS = os.environ.get("ALLOW_GOOGLE_DOCS", "false").lower() in ("1", "true", "yes")
GOOGLE_DOCS_ALLOWED_HOSTS = [
    "docs.google.com",
    "drive.google.com",
    "spreadsheets.google.com",
]
if ALLOW_GOOGLE_DOCS:
    for host in GOOGLE_DOCS_ALLOWED_HOSTS:
        if host not in URL_ALLOWLIST_HOSTS:
            URL_ALLOWLIST_HOSTS.append(host)
URL_ENFORCE_ALLOWLIST = os.environ.get("URL_ENFORCE_ALLOWLIST", "true").lower() in ("1", "true", "yes")
URL_BLOCK_PRIVATE_IPS = os.environ.get("URL_BLOCK_PRIVATE_IPS", "true").lower() in ("1", "true", "yes")
try:
    URL_MAX_REDIRECTS = max(0, int(os.environ.get("URL_MAX_REDIRECTS", "3")))
except Exception:
    URL_MAX_REDIRECTS = 3

ALLOW_LEGACY_OUTPUT_DOWNLOAD = os.environ.get("ALLOW_LEGACY_OUTPUT_DOWNLOAD", "false").lower() in ("1", "true", "yes")
try:
    MAX_SOCKET_EVENT_BYTES = max(1024, int(os.environ.get("MAX_SOCKET_EVENT_BYTES", "65536")))
except Exception:
    MAX_SOCKET_EVENT_BYTES = 65536
try:
    MAX_SOCKET_LOG_BYTES = max(2048, int(os.environ.get("MAX_SOCKET_LOG_BYTES", "131072")))
except Exception:
    MAX_SOCKET_LOG_BYTES = 131072

# Agent selection / navigation hardening
ENABLE_SELENIUM_FALLBACK = os.environ.get("ENABLE_SELENIUM_FALLBACK", "false").lower() in ("1", "true", "yes")
try:
    NAV_MAX_ATTEMPTS = max(1, int(os.environ.get("NAV_MAX_ATTEMPTS", "2")))
except Exception:
    NAV_MAX_ATTEMPTS = 2
try:
    NAV_TIMEOUT_PLAYWRIGHT_MS = max(5000, int(os.environ.get("NAV_TIMEOUT_PLAYWRIGHT_MS", "60000")))
except Exception:
    NAV_TIMEOUT_PLAYWRIGHT_MS = 60000
try:
    NAV_TIMEOUT_SELENIUM_MS = max(5000, int(os.environ.get("NAV_TIMEOUT_SELENIUM_MS", "60000")))
except Exception:
    NAV_TIMEOUT_SELENIUM_MS = 60000

# Fuzzy matching policy defaults (used by fec_lookup and reporting tools)
try:
    MIN_FUZZY_SCORE_AUTO = int(os.environ.get("MIN_FUZZY_SCORE_AUTO", "90"))
except Exception:
    MIN_FUZZY_SCORE_AUTO = 90
try:
    MIN_FUZZY_SCORE_MANUAL = int(os.environ.get("MIN_FUZZY_SCORE_MANUAL", "70"))
except Exception:
    MIN_FUZZY_SCORE_MANUAL = 70
FUZZY_SCORER = os.environ.get("FUZZY_SCORER", "auto").strip() or "auto"

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
    3. Default: [".json", ".csv", ".pdf", ".txt", ".xlsx"]
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
                formats_raw = context_library.get("supported_formats", [".json", ".csv", ".pdf", ".txt", ".xlsx"])
                if isinstance(formats_raw, list):
                    return formats_raw
                elif isinstance(formats_raw, str):
                    import json
                    parsed = json.loads(formats_raw)
                    return parsed if isinstance(parsed, list) else [".json", ".csv", ".pdf", ".txt", ".xlsx"]
    except Exception:
        pass  # Optionally log error here
    return [".json", ".csv", ".pdf", ".txt", ".xlsx"]

SUPPORTED_FORMATS = [ext for ext in get_supported_formats() if ext.lower() not in [".html", "html"]]
SUPPORTED_EXTENSION_SET = {
    ext.lower() if ext.startswith(".") else f".{ext.lower()}"
    for ext in SUPPORTED_FORMATS
}

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
                logger.info("[DB][AAD] Falling back to password auth.")
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
    "PIPELINE_HEARTBEAT_INTERVAL","ENABLE_USER_FEEDBACK", "DISABLE_HTML_FALLBACK", "ENABLE_CAMELOT",
    "MIN_FUZZY_SCORE_AUTO","MIN_FUZZY_SCORE_MANUAL","FUZZY_SCORER",
    "CAMELOT_MIN_SCORE","CAMELOT_HYBRID_FILL","CAMELOT_MERGE_COMPAT",

    # Training params
    "SBERT_EPOCHS","SBERT_BATCH_SIZE","SPACY_NER_EPOCHS","SPACY_NER_PATIENCE",
    "SPACY_NER_MIN_DELTA","SPACY_NER_BATCH_SIZE","REVIEW_WITH_MANUAL_BOT",

    # Formats
    "SUPPORTED_FORMATS","SUPPORTED_EXTENSION_SET","SEED_URLS_IF_EMPTY",

    # Helpers
    "get_subprocess_env","get_supported_formats",
    
    # OCR paths & toggles
    "ENABLE_OCR","ENABLE_OCR_FORCE","POPPLER_PATH","TESSERACT_CMD","OCR_DEBUG_DIR",
    
    # OCR Tuning Parameters
    "OCR_CONFIDENCE_THRESHOLD","OCR_MIN_ALPHA_SIGNAL","OCR_AVG_CONF_ACCEPT",
    "OCR_DPI_MIN","OCR_DPI_MAX","OCR_DPI_STEP","OCR_PSM_LIST","OCR_OEM_LIST",
    "OCR_PREPROCESS_VARIANTS","OCR_SAMPLE_BUDGET","OCR_MAX_RUNS",
    "OCR_ORIENTATION_THRESHOLD","OCR_DENSE_LINE_THRESHOLD",
    "OCR_TABLE_SIGNAL_MIN_COLS","OCR_TABLE_SIGNAL_MIN_ROWS",
    "OCR_MARKUP_HTML_TAG_RATIO","OCR_DEBUG_SAVE_IMAGES",
    "OCR_FAST_MODE_DPI_LIMIT","OCR_FAST_MODE_SAMPLE_LIMIT",
    "PDF_FAST_MODE","PDF_PROBE_MAX_PAGES",
    "TABLE_BUILDER_AUTO_ACCEPT_THRESHOLD","TABLE_BUILDER_LOW_CONFIDENCE_THRESHOLD",
    "HEADER_CONFIDENCE_THRESHOLD","HEADER_INSERT_CONFIDENCE_THRESHOLD",
    "SEGMENT_ML_LABEL_THRESHOLD","SEGMENT_ML_LABEL_THRESHOLD_STRICT",
    "ENTITY_LINKING_THRESHOLD",
    "CONTEST_VERIFY_THRESHOLD","CONTEST_VERIFY_FLOOR_NO_MODEL",
    "CONTEST_FEEDBACK_THRESHOLD","CONTEST_FEEDBACK_MIN_THRESHOLD",
    "CONTEST_AUTO_CONFIDENCE_THRESHOLD",
    "SLOW_NLP_AUDIT_THRESHOLD","SLOW_NLP_AUDIT_MIN_HITS",
    # OCR helpers
    "get_ocr_config_dict","log_ocr_config_summary",
    # ML quality metrics
    "build_extraction_quality_metrics","log_extraction_quality",
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

# === OCR Tuning Configuration ===
# Centralized OCR parameters for PDF parsing with ML-ready tuning support
# All values can be overridden via environment variables

# Core Quality Thresholds
OCR_CONFIDENCE_THRESHOLD = int(os.environ.get("OCR_CONFIDENCE_THRESHOLD", "30"))
OCR_MIN_ALPHA_SIGNAL = int(os.environ.get("OCR_MIN_ALPHA_SIGNAL", "200"))
OCR_AVG_CONF_ACCEPT = float(os.environ.get("OCR_AVG_CONF_ACCEPT", "70.0"))

# Adaptive Search Space
OCR_DPI_MIN = int(os.environ.get("OCR_DPI_MIN", "200"))
OCR_DPI_MAX = int(os.environ.get("OCR_DPI_MAX", "350"))
OCR_DPI_STEP = int(os.environ.get("OCR_DPI_STEP", "50"))
OCR_PSM_LIST = [int(x.strip()) for x in os.environ.get("OCR_PSM_LIST", "6,4,3,11,12,1,13").split(",") if x.strip()]
OCR_OEM_LIST = [int(x.strip()) for x in os.environ.get("OCR_OEM_LIST", "1,3,2,0").split(",") if x.strip()]
OCR_PREPROCESS_VARIANTS = [x.strip() for x in os.environ.get("OCR_PREPROCESS_VARIANTS", "none,gray,thresh,sharp_contrast").split(",") if x.strip()]

# Search Budget & Convergence
OCR_SAMPLE_BUDGET = int(os.environ.get("OCR_SAMPLE_BUDGET", "12"))
OCR_MAX_RUNS = int(os.environ.get("OCR_MAX_RUNS", "20"))

# Orientation & Layout
OCR_ORIENTATION_THRESHOLD = float(os.environ.get("OCR_ORIENTATION_THRESHOLD", "10.0"))
OCR_DENSE_LINE_THRESHOLD = int(os.environ.get("OCR_DENSE_LINE_THRESHOLD", "500"))

# Table Detection Heuristics
OCR_TABLE_SIGNAL_MIN_COLS = int(os.environ.get("OCR_TABLE_SIGNAL_MIN_COLS", "2"))
OCR_TABLE_SIGNAL_MIN_ROWS = int(os.environ.get("OCR_TABLE_SIGNAL_MIN_ROWS", "3"))

# Markup Detection
OCR_MARKUP_HTML_TAG_RATIO = float(os.environ.get("OCR_MARKUP_HTML_TAG_RATIO", "0.3"))

# Debug & Fast Mode
OCR_DEBUG_SAVE_IMAGES = os.environ.get("OCR_DEBUG_SAVE_IMAGES", "1").lower() in ("1", "true", "yes")
OCR_FAST_MODE_DPI_LIMIT = int(os.environ.get("OCR_FAST_MODE_DPI_LIMIT", "250"))
OCR_FAST_MODE_SAMPLE_LIMIT = int(os.environ.get("OCR_FAST_MODE_SAMPLE_LIMIT", "6"))

# PDF-specific fast mode (enables aggressive optimization for PDF parsing)
PDF_FAST_MODE = os.environ.get("PDF_FAST_MODE", "false").lower() in ("1", "true", "yes")
PDF_PROBE_MAX_PAGES = int(os.environ.get("PDF_PROBE_MAX_PAGES", "5"))

# --- Confidence Thresholds (Table Structure + Header Mapping) ---
# These values tune ML/NLP confidence gates for table/header validation.
TABLE_BUILDER_AUTO_ACCEPT_THRESHOLD = float(
    os.environ.get("TABLE_BUILDER_AUTO_ACCEPT_THRESHOLD", "0.94")
)
TABLE_BUILDER_LOW_CONFIDENCE_THRESHOLD = float(
    os.environ.get("TABLE_BUILDER_LOW_CONFIDENCE_THRESHOLD", "0.75")
)
HEADER_CONFIDENCE_THRESHOLD = float(
    os.environ.get("HEADER_CONFIDENCE_THRESHOLD", "0.88")
)
HEADER_INSERT_CONFIDENCE_THRESHOLD = float(
    os.environ.get("HEADER_INSERT_CONFIDENCE_THRESHOLD", "0.88")
)

# --- NLP/ML Conjunction Thresholds ---
# Tune for unbiased recognition on names/places and higher-precision labeling.
SEGMENT_ML_LABEL_THRESHOLD = float(
    os.environ.get("SEGMENT_ML_LABEL_THRESHOLD", "0.80")
)
SEGMENT_ML_LABEL_THRESHOLD_STRICT = float(
    os.environ.get("SEGMENT_ML_LABEL_THRESHOLD_STRICT", "0.88")
)
ENTITY_LINKING_THRESHOLD = float(
    os.environ.get("ENTITY_LINKING_THRESHOLD", "0.85")
)

# --- Contest Selection Thresholds ---
CONTEST_VERIFY_THRESHOLD = float(
    os.environ.get("CONTEST_VERIFY_THRESHOLD", "0.78")
)
CONTEST_VERIFY_FLOOR_NO_MODEL = float(
    os.environ.get("CONTEST_VERIFY_FLOOR_NO_MODEL", "0.70")
)
CONTEST_FEEDBACK_THRESHOLD = float(
    os.environ.get("CONTEST_FEEDBACK_THRESHOLD", "0.82")
)
CONTEST_FEEDBACK_MIN_THRESHOLD = float(
    os.environ.get("CONTEST_FEEDBACK_MIN_THRESHOLD", "0.65")
)
CONTEST_AUTO_CONFIDENCE_THRESHOLD = float(
    os.environ.get("CONTEST_AUTO_CONFIDENCE_THRESHOLD", "0.94")
)

# --- Slow NLP Audit Thresholds (web pipeline) ---
SLOW_NLP_AUDIT_THRESHOLD = float(
    os.environ.get("SLOW_NLP_AUDIT_THRESHOLD", "0.60")
)
SLOW_NLP_AUDIT_MIN_HITS = int(
    os.environ.get("SLOW_NLP_AUDIT_MIN_HITS", "1")
)

# --- OCR telemetry helpers (kept lightweight, no external deps) ---
def get_ocr_config_dict(config_module=None) -> dict:
    """Return a dictionary snapshot of OCR-related tuning and environment.

    If a module is provided, values are read via getattr; otherwise, local constants are used.
    """
    src = config_module if config_module is not None else globals()
    # Helper to read from module or local globals
    def _get(name, default=None):
        try:
            if src is globals():
                return globals().get(name, default)
            return getattr(src, name, default)
        except Exception:
            return default

    return {
        # Toggles & paths
        "ENABLE_OCR": _get("ENABLE_OCR", True),
        "ENABLE_OCR_FORCE": _get("ENABLE_OCR_FORCE", False),
        "POPPLER_PATH": _get("POPPLER_PATH"),
        "TESSERACT_CMD": _get("TESSERACT_CMD"),
        "OCR_DEBUG_DIR": str(_get("OCR_DEBUG_DIR")) if _get("OCR_DEBUG_DIR") is not None else None,
        # Thresholds
        "OCR_CONFIDENCE_THRESHOLD": _get("OCR_CONFIDENCE_THRESHOLD", 30),
        "OCR_MIN_ALPHA_SIGNAL": _get("OCR_MIN_ALPHA_SIGNAL", 200),
        "OCR_AVG_CONF_ACCEPT": _get("OCR_AVG_CONF_ACCEPT", 70.0),
        # Search space
        "OCR_DPI_MIN": _get("OCR_DPI_MIN", 200),
        "OCR_DPI_MAX": _get("OCR_DPI_MAX", 350),
        "OCR_DPI_STEP": _get("OCR_DPI_STEP", 50),
        "OCR_PSM_LIST": list(_get("OCR_PSM_LIST", [])) or [6,4,3,11,12,1,13],
        "OCR_OEM_LIST": list(_get("OCR_OEM_LIST", [])) or [1,3,2,0],
        "OCR_PREPROCESS_VARIANTS": list(_get("OCR_PREPROCESS_VARIANTS", [])) or ["none","gray","thresh","sharp_contrast"],
        # Budget & convergence
        "OCR_SAMPLE_BUDGET": _get("OCR_SAMPLE_BUDGET", 12),
        "OCR_MAX_RUNS": _get("OCR_MAX_RUNS", 20),
        # Orientation & layout
        "OCR_ORIENTATION_THRESHOLD": _get("OCR_ORIENTATION_THRESHOLD", 10.0),
        "OCR_DENSE_LINE_THRESHOLD": _get("OCR_DENSE_LINE_THRESHOLD", 500),
        # Table heuristics
        "OCR_TABLE_SIGNAL_MIN_COLS": _get("OCR_TABLE_SIGNAL_MIN_COLS", 2),
        "OCR_TABLE_SIGNAL_MIN_ROWS": _get("OCR_TABLE_SIGNAL_MIN_ROWS", 3),
        # Markup detection
        "OCR_MARKUP_HTML_TAG_RATIO": _get("OCR_MARKUP_HTML_TAG_RATIO", 0.3),
        # Fast/debug
        "OCR_DEBUG_SAVE_IMAGES": _get("OCR_DEBUG_SAVE_IMAGES", True),
        "OCR_FAST_MODE_DPI_LIMIT": _get("OCR_FAST_MODE_DPI_LIMIT", 250),
        "OCR_FAST_MODE_SAMPLE_LIMIT": _get("OCR_FAST_MODE_SAMPLE_LIMIT", 6),
        "PDF_FAST_MODE": _get("PDF_FAST_MODE", False),
        "PDF_PROBE_MAX_PAGES": _get("PDF_PROBE_MAX_PAGES", 5),
    }

def log_ocr_config_summary(config_module, logger, session_id=None) -> None:
    """Emit a concise log line summarizing active OCR config.

    Uses the existing SharedLogger style (level/type/message) with an attached snapshot.
    """
    try:
        snapshot = get_ocr_config_dict(config_module)
        logger.info({
            "level": "INFO",
            "type": "status",
            "message": "[OCR] Active tuning parameters",
            "session_id": session_id,
            "ocr_config": snapshot,
        })
    except Exception:
        # Avoid throwing from logging helper; keep it best-effort
        pass

def build_extraction_quality_metrics(
    headers: list[str],
    data: list[dict],
    metadata: dict,
    handler_name: str = "unknown",
    session_id: str | None = None,
) -> dict:
    """Build standardized quality metrics for ML analysis and telemetry.

    Captures extraction quality indicators that ML models can use to:
    - Correlate tuning parameters with output quality
    - Identify patterns in successful vs. failed extractions
    - Tune adaptive search strategies dynamically
    - Flag anomalous or low-confidence results for review

    Args:
        headers: Column headers extracted from source
        data: Row data (list of dicts)
        metadata: Handler-specific metadata dict
        handler_name: Name of handler (pdf, html, csv, json, etc.)
        session_id: Optional session identifier

    Returns:
        dict: Quality metrics snapshot with structure:
            {
                "handler": str,
                "row_count": int,
                "column_count": int,
                "empty_row_ratio": float,
                "null_cell_ratio": float,
                "avg_row_density": float,
                "header_completeness": float,
                "data_type_diversity": int,
                "has_numeric_columns": bool,
                "has_text_columns": bool,
                "extraction_confidence": float | None,
                "ocr_metrics": dict | None,  # OCR-specific metrics (if applicable)
                "table_metrics": dict | None,  # Table structure metrics (if applicable)
                "session_id": str | None,
            }
    """
    import re
    from collections import Counter

    metrics = {
        "handler": handler_name,
        "row_count": len(data),
        "column_count": len(headers),
        "session_id": session_id,
    }

    if not data:
        # Empty dataset - minimal metrics
        metrics.update({
            "empty_row_ratio": 1.0,
            "null_cell_ratio": 1.0,
            "avg_row_density": 0.0,
            "header_completeness": 1.0 if headers else 0.0,
            "data_type_diversity": 0,
            "has_numeric_columns": False,
            "has_text_columns": False,
            "extraction_confidence": 0.0,
        })
        return metrics

    # Calculate empty row ratio (rows with all empty/null values)
    empty_rows = sum(1 for row in data if all(not str(v).strip() for v in row.values()))
    metrics["empty_row_ratio"] = empty_rows / len(data) if data else 0.0

    # Calculate null cell ratio (empty cells / total cells)
    total_cells = len(data) * len(headers)
    null_cells = sum(
        sum(1 for v in row.values() if not str(v).strip())
        for row in data
    )
    metrics["null_cell_ratio"] = null_cells / total_cells if total_cells > 0 else 0.0

    # Calculate average row density (non-empty cells per row)
    row_densities = [
        sum(1 for v in row.values() if str(v).strip()) / len(headers) if headers else 0.0
        for row in data
    ]
    metrics["avg_row_density"] = sum(row_densities) / len(row_densities) if row_densities else 0.0

    # Header completeness (non-empty headers / total headers)
    non_empty_headers = sum(1 for h in headers if str(h).strip())
    metrics["header_completeness"] = non_empty_headers / len(headers) if headers else 0.0

    # Data type diversity (unique inferred types across all cells)
    type_pattern_counts = Counter()
    for row in data[:min(100, len(data))]:  # Sample first 100 rows for performance
        for val in row.values():
            val_str = str(val).strip()
            if not val_str:
                type_pattern_counts["empty"] += 1
            elif re.fullmatch(r"-?\d+", val_str):
                type_pattern_counts["integer"] += 1
            elif re.fullmatch(r"-?\d+\.\d+", val_str):
                type_pattern_counts["float"] += 1
            elif re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2,4}", val_str):
                type_pattern_counts["date"] += 1
            elif val_str.lower() in {"true", "false", "yes", "no"}:
                type_pattern_counts["boolean"] += 1
            else:
                type_pattern_counts["text"] += 1
    metrics["data_type_diversity"] = len([t for t in type_pattern_counts if t != "empty"])
    metrics["has_numeric_columns"] = any(t in type_pattern_counts for t in ["integer", "float"])
    metrics["has_text_columns"] = "text" in type_pattern_counts

    # Extract OCR-specific metrics if available (multiple formats supported)
    ocr_metrics = None
    
    # Format 1: Nested ocr_stats dict (legacy/test format)
    if "ocr_stats" in metadata and isinstance(metadata["ocr_stats"], dict):
        stats = metadata["ocr_stats"]
        ocr_metrics = {
            "avg_confidence": stats.get("avg_confidence"),
            "min_confidence": stats.get("min_confidence"),
            "ocr_run_count": stats.get("ocr_run_count"),
            "ocr_time_sec": stats.get("ocr_time_sec"),
            "ocr_pages_processed": stats.get("ocr_pages_processed"),
        }
    
    # Format 2: Direct metadata fields (PDF handler format)
    elif any(k in metadata for k in ["ocr_confidence_avg", "ocr_runs", "ocr_used"]):
        ocr_metrics = {
            "avg_confidence": metadata.get("ocr_confidence_avg"),
            "min_confidence": metadata.get("ocr_min_confidence"),  # May not always be present
            "ocr_run_count": metadata.get("ocr_runs"),
            "ocr_time_sec": metadata.get("ocr_time_sec"),  # May not always be present
            "ocr_pages_processed": metadata.get("ocr_pages_processed"),  # May not always be present
        }
        # Remove None values for cleaner output
        ocr_metrics = {k: v for k, v in ocr_metrics.items() if v is not None}
    metrics["ocr_metrics"] = ocr_metrics if ocr_metrics else None

    # Extract table structure metrics if available
    table_metrics = None
    if any(k in metadata for k in ["layout_table_rows", "page_line_total", "pdf_page_total"]):
        table_metrics = {
            "layout_rows": metadata.get("layout_table_rows"),
            "layout_cols": metadata.get("layout_table_cols"),
            "page_count": metadata.get("pdf_page_total"),
            "line_count": metadata.get("page_line_total"),
            "table_confidence": metadata.get("table_extraction_confidence"),
        }
        # Remove None values
        table_metrics = {k: v for k, v in table_metrics.items() if v is not None}
    metrics["table_metrics"] = table_metrics if table_metrics else None

    # Overall extraction confidence (weighted heuristic based on multiple factors)
    confidence_factors = []
    weights = []
    
    # Data completeness (weight: 0.3)
    if metrics["avg_row_density"] > 0:
        confidence_factors.append(metrics["avg_row_density"])
        weights.append(0.3)
    
    # Header quality (weight: 0.2)
    if metrics["header_completeness"] > 0:
        confidence_factors.append(metrics["header_completeness"])
        weights.append(0.2)
    
    # Non-empty rows (weight: 0.2)
    if 1 - metrics["empty_row_ratio"] > 0:
        confidence_factors.append(1 - metrics["empty_row_ratio"])
        weights.append(0.2)
    
    # OCR quality (weight: 0.3 - most important for PDF extractions)
    if ocr_metrics and ocr_metrics.get("avg_confidence"):
        ocr_conf = ocr_metrics["avg_confidence"]
        # Normalize if confidence is in 0-100 range
        ocr_normalized = ocr_conf / 100.0 if ocr_conf > 1.0 else ocr_conf
        confidence_factors.append(ocr_normalized)
        weights.append(0.3)
    
    # Weighted average (or simple average if no weights)
    if confidence_factors:
        if len(weights) == len(confidence_factors):
            # Normalize weights to sum to 1.0
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            metrics["extraction_confidence"] = sum(
                f * w for f, w in zip(confidence_factors, normalized_weights)
            )
        else:
            # Fallback to simple average
            metrics["extraction_confidence"] = sum(confidence_factors) / len(confidence_factors)
    else:
        metrics["extraction_confidence"] = None

    return metrics


def log_extraction_quality(
    headers: list[str],
    data: list[dict],
    metadata: dict,
    handler_name: str,
    logger,
    session_id: str | None = None,
) -> dict:
    """Build and log extraction quality metrics for ML analysis.

    This is the main entrypoint for handlers to report quality metrics.
    Calls build_extraction_quality_metrics() and logs the result.

    Returns:
        dict: The quality metrics snapshot (same as build_extraction_quality_metrics)
    """
    try:
        quality = build_extraction_quality_metrics(
            headers, data, metadata, handler_name, session_id
        )
        logger.info({
            "level": "INFO",
            "type": "ml_quality",
            "message": f"[ML] Extraction quality metrics ({handler_name})",
            "session_id": session_id,
            "quality_metrics": quality,
        })
        return quality
    except Exception as e:
        # Best-effort logging; don't throw
        logger.warning({
            "level": "WARNING",
            "type": "ml_quality",
            "message": f"[ML] Failed to build quality metrics: {e}",
            "session_id": session_id,
        })
        return {}


# === Verification Framework Configuration ===

# Path to verification audit trail (DL2 → DL1 verification decisions)
VERIFICATION_LOG_DIR = CONTEXT_LIBRARY_DIR / "verification"
VERIFICATION_LOG_DIR.mkdir(parents=True, exist_ok=True)
VERIFICATION_LOG_FILE = VERIFICATION_LOG_DIR / "verification_log.jsonl"

# DL1/DL2 Verification Storage (Local Filesystem)
# DL1: Human-verified ground truth (authoritative source of truth)
# DL2: AI-extracted working dataset (subject to hallucination)
# NOTE: Both DL1 and DL2 are now stored in CONTEXT_LIBRARY_DIR/verification
# See webapp/parser/verification/local_dl_sync.py for sync management
# (No external dependencies - completely local filesystem-based)

# Verification workflow toggles
ENABLE_VERIFICATION_FRAMEWORK = os.environ.get("ENABLE_VERIFICATION_FRAMEWORK", "true").lower() in ("1", "true", "yes")
ALLOW_UNVERIFIED_EXPORTS = os.environ.get("ALLOW_UNVERIFIED_EXPORTS", "false").lower() in ("1", "true", "yes")

# QA Framework: Require certificate authentication for data assurance endpoints
# SECURITY: Defaults to TRUE (cert auth required). Set to "false" only for local development.
# Production deployments MUST use certificate authentication (X-ARR-ClientCert header).
QA_REQUIRE_CERT_AUTH = os.environ.get("QA_REQUIRE_CERT_AUTH", "true").lower() in ("1", "true", "yes")

# Verification confidence threshold for automatic DL1 promotion
# (ADMIN_FULL_TRUST can override, but ROOT_ADMIN required for bypass)
try:
    MIN_VERIFICATION_CONFIDENCE = float(os.environ.get("MIN_VERIFICATION_CONFIDENCE", "0.85"))
except Exception:
    MIN_VERIFICATION_CONFIDENCE = 0.85

# Maximum time to keep DL2 (extracted) rows before requiring verification
# (0 = indefinite; > 0 = days before archival)
try:
    DL2_RETENTION_DAYS = max(0, int(os.environ.get("DL2_RETENTION_DAYS", "90")))
except Exception:
    DL2_RETENTION_DAYS = 90

# System authorship & mission (immutable)
SYSTEM_AUTHOR = "Juancarlos Barragan"
SYSTEM_AUTHOR_DOB = "1996-03-18"
SYSTEM_AUTHOR_LOCATION = "6858 S 12th Ave, Tucson, AZ"
SYSTEM_GOVERNANCE_FILE = PROJECT_ROOT / "SYSTEM_GOVERNANCE.md"
SYSTEM_MISSION = (
    "Protect the voice of the people by preserving the accurate count of legitimate votes. "
    "Detect unintentional data errors at acceptable thresholds."
)
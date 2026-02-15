import errno
import glob
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import orjson
from sqlalchemy import inspect

from ..config import (
    BATCH_MODE,
    CACHE_DIR,
    CACHE_EXPIRE_DAYS,
    CONTEXT_PATH,
    COOLDOWN,
    CORRECTION_MODE,
    DB_PATH,
    DRY_RUN,
    ENABLE_ENHANCED,
    EXPORT_AUDIT_LOG,
    FAST_MODE,
    FIELDS,
    FILTER_CONTEXT_KEY,
    FILTER_VALUE,
    FLUSH_CACHE,
    INTEGRITY_CHECK,
    LOG_DIR,
    MAX_RETRIES,
    MODEL_DIR,
    NO_COORDINATOR,
    NO_ORGANIZER,
    PROJECT_ROOT,
    REST_API,
    SELF_HEAL,
    UPDATE_DB,
)
from ..Context_Integration.librarian import load_context_library
from ..utils.db_utils import get_engine
from ..utils.logger_singleton import console, logger
from ..utils.models import Base
from .integrity_monitor import get_integrity_monitor
from .navigation_feedback_ingest import ingest_navigation_feedback

# =============================================================================
# LOCAL LEARNING SYSTEM: Election Data Integrity & Accuracy Preservation
# =============================================================================
# This system learns from ingested election data to improve parsing accuracy
# and preserve data integrity across sessions. All data is stored locally.
# No external API calls - fully self-contained machine learning pipeline.
#
# Key Design Principles:
# 1. Local persistence via context_library.json for continuous learning
# 2. Feature extraction from successful + failed parsing attempts
# 3. Confidence scoring based on pattern recognition from historical data
# 4. SQL backend (warehoused election results) provides training signals
# 5. Internal NLP/ML only: spaCy NER, sentence-transformers, scikit-learn
# 6. Optional HuggingFace local models (no cloud dependencies)
#
# Learning Loop:
# - Session processes election data -> IntegrityMonitor captures features
# - High-priority/anomalous sessions -> persist to context_library.json
# - ML pipeline trains on historical patterns (state, county, contest)
# - Future sessions benefit from learned patterns -> improved accuracy
# =============================================================================

class LocalLearningEngine:
    """Manages local ML training and inference for election data accuracy."""
    
    def __init__(self):
        self.monitor = get_integrity_monitor()
        self.training_data_path = os.path.join(LOG_DIR, "training_data.jsonl")
        self.model_checkpoint = os.path.join(MODEL_DIR, "election_accuracy_model.pt")
        
    def ingest_training_signal(self, session_context, success, quality_metrics):
        """Capture learning signal from successful/failed parsing."""
        signal = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state": session_context.get("state"),
            "county": session_context.get("county"),
            "contest": session_context.get("contest"),
            "handler": session_context.get("handler"),
            "success": success,
            "metrics": quality_metrics,
            "source": "parser_feedback"
        }
        # Append to local training data log
        try:
            with open(self.training_data_path, "a", encoding="utf-8") as f:
                f.write(orjson.dumps(signal).decode() + "\n")
        except Exception as e:
            logger.warning(f"[LocalLearning] Failed to record training signal: {e}")
    
    def get_learned_accuracy_score(self, session_context):
        """Query learned patterns to get expected accuracy for this context."""
        # Uses IntegrityMonitor's cached historical knowledge
        state = session_context.get("state", "")
        county = session_context.get("county", "")
        
        # Pattern matching from context_library
        try:
            library = load_context_library()
            checks = library.get("integrity_checks", [])
            
            # Find similar historical contexts
            matches = [
                c for c in checks
                if c.get("context_summary", {}).get("state") == state
                and c.get("context_summary", {}).get("county") == county
            ]
            
            if matches:
                scores = [float(m.get("health_score", 0.5)) for m in matches]
                avg_score = sum(scores) / len(scores)
                return avg_score
        except Exception as e:
            logger.debug(f"[LocalLearning] Pattern lookup failed: {e}")
        
        return 0.5  # Default neutral score

# Initialize learning engine (singleton)
_learning_engine = None

def get_learning_engine():
    """Get or create LocalLearningEngine instance."""
    global _learning_engine
    if _learning_engine is None:
        _learning_engine = LocalLearningEngine()
    return _learning_engine

ORCHESTRATION_PLUGINS = []

def register_orchestration_plugin(plugin_func):
    ORCHESTRATION_PLUGINS.append(plugin_func)

def run_orchestration_plugins(context=None):
    suggestions = []
    for plugin in ORCHESTRATION_PLUGINS:
        try:
            suggestions.extend(plugin(context))
        except Exception as e:
            logger.error(f"[BOT ROUTER][PLUGIN ERROR] {e}")
    return suggestions

def preclean_json_logs(log_dirs, required_files=None, max_line_length=200000):
    """
    Clean all JSON/JSONL files in log_dirs.
    Quarantine corrupt lines, salvage valid lines, and create missing required files.
    """
    
    # Clean all .jsonl, .ndjson, and .json files
    for log_dir in log_dirs:
        for suf in [".jsonl", ".ndjson", ".json"]:
            for path in glob.glob(os.path.join(log_dir, f"*{suf}")):
                import json
                
                # Detect if .json is block JSON or JSONL by checking first non-whitespace char
                # .jsonl and .ndjson files are always line-delimited
                is_block_json = False
                if suf == ".json":
                    try:
                        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
                            first_char = None
                            for char in f:
                                if char.strip():
                                    first_char = char.strip()[0]
                                    break
                            # If file starts with array/object, assume block JSON
                            if first_char in "{[":
                                is_block_json = True
                    except Exception:
                        pass
                
                # Handle block JSON files separately
                if is_block_json:
                    try:
                        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
                            json.load(f)
                        print(f"[FIXED] Block JSON validated: {path}")
                    except json.JSONDecodeError as e:
                        print(f"[CORRUPT] Block JSON invalid in {path}: {str(e)}")
                        corrupt_path = path + ".corrupt"
                        with open(corrupt_path, "w", encoding="utf-8") as out:
                            out.write(f"Block JSON parsing failed:\n{str(e)}\n")
                    continue
                
                # Handle JSONL/line-delimited files
                valid_lines = []
                corrupt_lines = []
                with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        if len(line) > max_line_length:
                            corrupt_lines.append((i, "<line too long>", "exceeds max_line_length"))
                            continue
                        try:
                            # Try to parse as JSON
                            json.loads(line)
                            valid_lines.append(line)
                        except json.JSONDecodeError as e:
                            # Try to fix common issues
                            fixed = line
                            fixed = re.sub(r",\s*$", "", fixed)
                            fixed = re.sub(r"'", '"', fixed)
                            fixed = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', fixed)
                            fixed = fixed.replace('\ufeff', '').replace('\x00', '')
                            try:
                                json.loads(fixed)
                                valid_lines.append(fixed)
                            except json.JSONDecodeError:
                                corrupt_lines.append((i, line, str(e)))
                # Write back valid lines
                with open(path, "w", encoding="utf-8") as out:
                    for line in valid_lines:
                        out.write(line + "\n")
                # Save corrupt lines for review
                if corrupt_lines:
                    corrupt_path = path + ".corrupt"
                    with open(corrupt_path, "w", encoding="utf-8") as out:
                        for i, line, err in corrupt_lines:
                            out.write(f"Line {i+1}: {line}\nError: {err}\n\n")
                    print(f"[CORRUPT] {len(corrupt_lines)} lines saved to {corrupt_path}")
                print(f"[FIXED] Salvaged {len(valid_lines)}/{len(valid_lines)+len(corrupt_lines)} lines in {path}")

    # Ensure required files exist
    if required_files:
        for req in required_files:
            if not os.path.exists(req):
                with open(req, "w", encoding="utf-8") as f:
                    pass  # create empty file
                print(f"[INFO] Created missing required file: {req}")

class BotPipeline:
    def __init__(self):
        self.results = {}
        self.context = None
        self.last_run = None
        self.ai_suggestions = []
        self.lockfile = os.path.join(PROJECT_ROOT, "pipeline.lock")

    def ensure_db_tables(self):
        try:
            engine = get_engine()
            Base.metadata.create_all(engine)
            inspector = inspect(engine)
            table_names = inspector.get_table_names()
            from rich.table import Table
            table = Table(title="[MODELS] Tables present after creation")
            table.add_column("Table Name", style="green")
            for name in table_names:
                table.add_row(name)
            console.table(table)
            logger.info("[MODELS] All tables created successfully.")
            logger.info("[PIPELINE] DB tables ensured.")
            self.results['db_tables'] = 'success'
            return True
        except Exception as e:
            logger.error(f"[PIPELINE] DB table check failed: {e}")
            self.results['db_tables'] = 'fail'
            return False
        
    def build_correction_args(self):
        args = []
        if str(ENABLE_ENHANCED).lower() == "true":
            args.append("--enhanced")
        if str(CORRECTION_MODE).lower() == "feedback":
            args.append("--feedback")
        else:
            args.append("--auto")
        if str(INTEGRITY_CHECK).lower() == "true":
            args.append("--integrity")
        if str(UPDATE_DB).lower() == "true":
            args.append("--update-db")
        
        # Internal NLP/ML only - no external API dependencies
        # Uses local spaCy, sentence-transformers, and scikit-learn models
        
        if FILTER_CONTEXT_KEY:
            args.extend(["--filter-context-key", FILTER_CONTEXT_KEY])
        if FILTER_VALUE:
            args.extend(["--filter-value", FILTER_VALUE])
        if FIELDS:
            args.extend(["--fields"] + FIELDS.split(","))
        if CONTEXT_PATH:
            args.extend(["--context", CONTEXT_PATH])
        if LOG_DIR:
            args.extend(["--log-dir", LOG_DIR])
        if str(DRY_RUN).lower() == "true":
            args.append("--dry-run")
        if str(NO_COORDINATOR).lower() == "true":
            args.append("--no-coordinator")
        if str(NO_ORGANIZER).lower() == "true":
            args.append("--no-organizer")
        if str(BATCH_MODE).lower() == "true":
            args.append("--batch")
        if str(FAST_MODE).lower() == "true":
            args.append("--fast")
        if str(FLUSH_CACHE).lower() == "true":
            args.append("--flush-cache")
        if CACHE_EXPIRE_DAYS:
            args.extend(["--cache-expire-days", CACHE_EXPIRE_DAYS])
        if EXPORT_AUDIT_LOG:
            args.extend(["--export-audit-log", EXPORT_AUDIT_LOG])
        if str(REST_API).lower() == "true":
            args.append("--rest-api")
        if str(SELF_HEAL).lower() == "true":
            args.append("--self-heal")
            if MAX_RETRIES:
                args.extend(["--max-retries", MAX_RETRIES])
            if COOLDOWN:
                args.extend(["--cooldown", COOLDOWN])
        if DB_PATH:
            args.extend(["--db-path", DB_PATH])
        return args

    def run_manual_correction(self, mode="enhanced", extra_args=None, retries=1, timeout=600):
        """
        Optimized wrapper for manual_correction for end-of-pipeline use.
        Runs in enhanced/manual mode for safe, interactive correction.
        """
        args = self.build_correction_args()
        # Remove conflicting modes and always use enhanced/manual
        args = [a for a in args if a not in ["--auto", "--feedback", "--batch", "--fast", "--self-heal"]]
        args.append("--enhanced")
        # Add extra arguments if provided
        if extra_args:
            args.extend(extra_args)
        # Always add context and log-dir
        log_dir_path = Path(LOG_DIR) if not isinstance(LOG_DIR, Path) else LOG_DIR
        context_path = Path(LOG_DIR) / "context_library.json" if not isinstance(CONTEXT_PATH, Path) else CONTEXT_PATH
        args.extend([
            "--context", str(context_path),
            "--log-dir", str(log_dir_path)
        ])
        # Check for new entries before running
        if not self.has_new_entries(LOG_DIR, CACHE_DIR):
            logger.info("[health_router] No new entries for manual correction. Skipping enhanced mode and exiting gracefully.")
            self.results['manual_correction'] = 'skipped'
            return True  # Graceful exit
        # Try running with retries and timeout
        for attempt in range(1, retries + 1):
            try:
                logger.info(f"[health_router] Running manual_correction (enhanced mode, attempt={attempt}) with args: {args}")
                cmd = [sys.executable, "-m", "webapp.parser.health.manual_correction_bot"] + args
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=timeout)
                logger.info(f"[health_router] manual_correction stdout:\n{result.stdout[:1000]}")
                if result.returncode == 0:
                    logger.info("[health_router] manual_correction completed successfully.")
                    self.results['manual_correction'] = 'success'
                    return True
                else:
                    logger.warning(f"[health_router] manual_correction failed (attempt {attempt}): {result.stderr}")
                    time.sleep(2)
            except subprocess.TimeoutExpired:
                logger.error(f"[health_router] manual_correction timed out after {timeout} seconds (attempt {attempt}).")
            except Exception as e:
                logger.error(f"[health_router] manual_correction exception: {e}")
        logger.error("[health_router] manual_correction failed after all retries.")
        self.results['manual_correction'] = 'fail'
        return False

    def has_new_entries(self, log_dir, cache_dir):
        # Use the provided log_dir and cache_dir, not just LOG_DIR/CACHE_DIR from config
        log_dirs = [Path(log_dir), Path(cache_dir)]
        new_entries_found = False

        for dir_path in log_dirs:
            if not dir_path.exists():
                continue
            cmd = [
                sys.executable, "-m", "webapp.parser.health.manual_correction_bot",
                "--log-dir", str(dir_path),
                "--fields", "all",
                "--dry-run"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
            # Look for "Discovered X new entries" or "Discovered X valid entries" in output
            for line in result.stdout.splitlines():
                if ("Discovered" in line and "new entries" in line) or ("Discovered" in line and "valid entries" in line):
                    try:
                        # Try to extract the number between "Discovered" and "entries"
                        parts = line.split("Discovered")[1].split("entries")[0].strip()
                        count = int(''.join(filter(str.isdigit, parts)))
                        if count > 0:
                            new_entries_found = True
                            break
                    except Exception:
                        continue
            if new_entries_found:
                break

        # Additional check: if any .jsonl/.json file in log/cache dir is non-empty and not .corrupt
        if not new_entries_found:
            for dir_path in log_dirs:
                for suf in [".jsonl", ".json"]:
                    for file in dir_path.glob(f"*{suf}"):
                        if file.stat().st_size > 0 and not file.name.endswith(".corrupt"):
                            new_entries_found = True
                            break
                    if new_entries_found:
                        break
                if new_entries_found:
                    break

        return new_entries_found

    def lock(self):
        try:
            fd = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write("locked")
            return True
        except FileExistsError:
            logger.info("[INFO] Pipeline already running or ran.")
            return False
        except OSError as e:
            if e.errno == errno.EEXIST:
                logger.info("[INFO] Pipeline already running or ran.")
                return False
            else:
                raise

    def unlock(self):
        try:
            os.remove(self.lockfile)
        except Exception:
            pass

    def self_heal_loop(self, max_retries=3, cooldown=2):
        for attempt in range(1, max_retries + 1):
            logger.info(f"\n[SELF-HEAL] Attempt {attempt}...")
            exit_code = self.scan_misaligned()
            if exit_code == 0:
                logger.info("[SELF-HEAL] Data is clean. Exiting self-heal mode.")
                return 0
            logger.warning("[SELF-HEAL] Misalignments found. Launching manual_correction...")
            self.manual_correction(args=self.build_correction_args())
            logger.warning(f"[SELF-HEAL] Sleeping {cooldown}s before rescanning...")
            time.sleep(cooldown)
        logger.warning("[SELF-HEAL] Max retries reached. Some misalignments may remain.")
        return 2

    def run(self):
        if not self.lock():
            return
        try:
            self.last_run = datetime.now().isoformat()
            logger.info(f"[PIPELINE] Starting pipeline at {self.last_run}")

            # 0. Pre-clean all logs/cache/library files
            log_dirs = [LOG_DIR, CACHE_DIR, os.path.join(LOG_DIR, "log"), os.path.join(LOG_DIR, "cache")]
            required_files = [
                os.path.join(LOG_DIR, "spacy_ner_train_data.jsonl"),
                os.path.join(LOG_DIR, "context_library.json"),
            ]
            preclean_json_logs(log_dirs, required_files=required_files)

            # 1. Ensure DB tables
            if not self.ensure_db_tables():
                logger.error("[PIPELINE] DB table creation failed. Aborting pipeline.")
                return

            # 2. Clean logs/cache and migrate context
            clean_success = self.clean_and_migrate()
            if not clean_success:
                logger.error("[PIPELINE] Clean/migrate failed. Skipping retrain and correction.")
                return

            # 3. Fix corrupted JSON files before any processing
            try:
                cmd = [sys.executable, "-m", "webapp.parser.health.manual_correction_bot", "--fix-corrupt-json"]
                subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
                logger.info("[PIPELINE] Corrupted JSON files checked and fixed.")
            except Exception as e:
                logger.warning(f"[PIPELINE] Could not fix corrupted JSON files: {e}")

            try:
                nav_ingested = ingest_navigation_feedback(LOG_DIR)
                if nav_ingested:
                    logger.info(f"[PIPELINE] Staged {nav_ingested} navigation feedback entries for correction.")
                    self.results["navigation_feedback"] = f"processed:{nav_ingested}"
                else:
                    logger.info("[PIPELINE] No new navigation feedback entries detected.")
                    self.results["navigation_feedback"] = "none"
            except Exception as exc:
                logger.error(f"[PIPELINE] Navigation feedback ingestion failed: {exc}")
                self.results["navigation_feedback"] = "fail"

            # 4. Scan for misaligned NER examples
            misaligned = self.scan_misaligned()
            if misaligned == 2:
                logger.warning("[PIPELINE] Misaligned NER examples found. Self-heal loop will be handled by scan_misaligned_ner.")
            elif misaligned == 1:
                logger.warning("[PIPELINE] scan_misaligned_ner failed or file missing. Proceeding with caution.")
            elif misaligned == 0:
                logger.info("[PIPELINE] No misaligned NER examples found. Proceeding to manual correction.")

            # 5. Optimized orchestration for manual correction (scan_misaligned_ner already handled misalignments)
            has_entries = self.has_new_entries(LOG_DIR, CACHE_DIR)
            if not has_entries:
                logger.info("[PIPELINE] No new entries for manual correction, but running manual_correction anyway (may update existing entries).")
            else:
                logger.info("[PIPELINE] New entries detected for manual correction. Running manual_correction.")

            extra_args = []
            if str(INTEGRITY_CHECK).lower() == "true":
                extra_args.append("--integrity")
            if EXPORT_AUDIT_LOG:
                extra_args.extend(["--export-audit-log", EXPORT_AUDIT_LOG])
            if str(FLUSH_CACHE).lower() == "true":
                extra_args.append("--flush-cache")
            if CACHE_EXPIRE_DAYS:
                extra_args.extend(["--cache-expire-days", CACHE_EXPIRE_DAYS])
            logger.info("[PIPELINE] Running manual_correction in auto mode for context correction.")
            self.run_manual_correction(mode="auto", extra_args=extra_args)

            # 6. Retrain models (only if previous steps succeeded)
            retrain_success = self.retrain_models()
            if not retrain_success:
                logger.warning("[PIPELINE] Model retraining failed.")

            # 7. Reload context library after corrections
            self.context = load_context_library()
            logger.debug("DEBUG: Loaded context library:", type(self.context))
            if not isinstance(self.context, dict):
                logger.error("ERROR: Context library is not a dictionary. Check your context library loading logic.")
                raise ValueError("Context library must be a dictionary. Check your context library loading logic.")

            # 8. Post-process context (organizer, coordinator, integrity)
            self.context_postprocess()

            # 9. Run orchestration plugins
            self.run_orchestration_plugins()

            # 10. Self-improvement suggestions (local NLP/ML + static fallback)
            self.self_improve()

            # 11. Print pipeline summary
            self.print_summary()

        except Exception as e:
            logger.error(f"[PIPELINE] Unhandled exception: {e}")
            self.results['pipeline'] = 'fail'
        finally:
            self.unlock()

    def manual_correction(self, args=None):
        try:
            cmd = [sys.executable, "-m", "webapp.parser.health.manual_correction_bot"]
            if args:
                cmd.extend(args)
            subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
            self.results['manual_correction'] = 'success'
            return True
        except Exception as e:
            logger.error(f"[PIPELINE] manual_correction failed: {e}")
            self.results['manual_correction'] = 'fail'
            return False

    def retrain_models(self):
        try:
            cmd = [sys.executable, "-m", "webapp.parser.health.retrain_table_structure_models"]
            subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
            self.results['retrain_models'] = 'success'
            return True
        except Exception as e:
            logger.error(f"[PIPELINE] retrain_table_structure_models failed: {e}")
            self.results['retrain_models'] = 'fail'
            return False

    def scan_misaligned(self):
        try:
            cmd = [sys.executable, "-m", "webapp.parser.health.scan_misaligned_ner"]
            result = subprocess.run(cmd, cwd=PROJECT_ROOT)
            exit_code = result.returncode
            self.results['scan_misaligned'] = 'clean' if exit_code == 0 else 'misaligned'
            return exit_code
        except Exception as e:
            logger.error(f"[PIPELINE] scan_misaligned_ner failed: {e}")
            self.results['scan_misaligned'] = 'fail'
            return 2

    def clean_and_migrate(self):
        try:
            cmd = [sys.executable, "-m", "webapp.parser.health.log_cache_cleaner_bot"]
            subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
            cmd = [sys.executable, "-m", "webapp.parser.health.context_migration"]
            subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
            self.results['clean_migrate'] = 'success'
            return True
        except Exception as e:
            logger.error(f"[PIPELINE] Clean/migrate failed: {e}")
            self.results['clean_migrate'] = 'fail'
            return False

    def context_postprocess(self):
        try:
            # Example: add your context modules here
            from ..Context_Integration import (
                Integrity_check,
                context_coordinator,
                context_organizer,
            )
            contests = self.context.get("contests", [])
            if contests:
                Integrity_check.print_integrity_summary(contests)
                self.results["integrity_check"] = "success"
            else:
                self.results["integrity_check"] = "no_contests"
            organizer = context_organizer.ContextOrganizer()
            organizer.organize_context(self.context)
            self.results["context_organizer"] = "success"
            coordinator = context_coordinator.ContextCoordinator()
            coordinator.organize_and_enrich(self.context)
            self.results["context_coordinator"] = "success"
        except Exception as e:
            logger.error(f"[PIPELINE] Context modules failed: {e}")
            self.results["integrity_check"] = "fail"
            self.results["context_organizer"] = "fail"
            self.results["context_coordinator"] = "fail"

    def run_orchestration_plugins(self):
        try:
            plugin_results = run_orchestration_plugins(self.context)
            self.results["orchestration_plugins"] = "success" if plugin_results else "none"
        except Exception as e:
            logger.error(f"[PIPELINE] Orchestration plugins failed: {e}")
            self.results["orchestration_plugins"] = "fail"

    def self_improve(self):
        suggestion = None
        # Use LOCAL LEARNING ENGINE for session health and improvement suggestions
        try:
            learning_engine = get_learning_engine()

            # Prepare session context from current results
            session_context = {
                "contest": self.results.get("contest") or self.context.get("contest"),
                "state": self.results.get("state") or self.context.get("state"),
                "county": self.results.get("county") or self.context.get("county"),
                "handler": "health_router",
                "session_id": "health_bot"
            }

            # Get learned accuracy score based on historical patterns
            learned_score = learning_engine.get_learned_accuracy_score(session_context)

            # Get integrity monitor assessment
            monitor = get_integrity_monitor()
            flags = self.results.get("integrity_issues", [])
            health_result = monitor.assess_session_health(session_context, flags)

            # Merge learning engine insights
            health_result["learned_accuracy_score"] = learned_score
            health_result["learning_engine"] = "active"

            console.log(f"[HEALTH] Integrity score: {health_result['health_score']:.2f} (confidence: {health_result['confidence']:.2f})")
            console.log(f"[HEALTH] Learned accuracy: {learned_score:.2f} | Priority: {health_result['priority']}")
            console.log(f"[HEALTH] Recommendations: {health_result['recommendations']}")
            return health_result
        except Exception as e:
            logger.error(f"[HEALTH] Local learning analysis failed: {e}")

        # Fallback: rule-based suggestions from historical patterns
        if self.results.get("scan_misaligned") == "misaligned":
            suggestion = "Consider running manual_correction with --self-heal or retraining models based on learned patterns."
        else:
            suggestion = "Pipeline ran clean. Monitor logs for anomalies."
        logger.info(f"[PIPELINE][STATIC SUGGESTION]: {suggestion}")
        self.ai_suggestions.append(suggestion)

    def print_summary(self):
        logger.info("\n[PIPELINE] Run Summary:")
        for k, v in self.results.items():
            logger.info(f"  {k:<20}: {v}")
        if self.ai_suggestions:
            console.print("\n[PIPELINE] AI Suggestions:")
            for s in self.ai_suggestions:
                console.print(f"  - {s}")

if __name__ == "__main__":
    pipeline = BotPipeline()
    pipeline.run()
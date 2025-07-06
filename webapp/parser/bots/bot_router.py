import os
import sys
import time
import subprocess
import orjson
import errno
from datetime import datetime
from sqlalchemy import inspect
from ..utils.shared_logger import log_info, log_error, summarize_logs
from ..bots.log_cache_cleaner_bot import run_log_cache_cleaner
from ..bots.context_migration import migrate_all
from ..bots.scan_misaligned_ner import scan_misaligned
from ..bots.manual_correction_bot import find_log_files, load_jsonl
from ..bots.librarian import load_context_library
from ..utils.models import TableStructure, Base
from ..utils.db_utils import get_engine
from ..config import LOG_DIR, CACHE_DIR, PROJECT_ROOT
try:
    import openai
except ImportError:
    openai = None

ORCHESTRATION_PLUGINS = []

def register_orchestration_plugin(plugin_func):
    ORCHESTRATION_PLUGINS.append(plugin_func)

def run_orchestration_plugins(context=None):
    suggestions = []
    for plugin in ORCHESTRATION_PLUGINS:
        try:
            suggestions.extend(plugin(context))
        except Exception as e:
            log_error(f"[BOT ROUTER][PLUGIN ERROR] {e}")
    return suggestions

class BotPipeline:
    def __init__(self):
        self.results = {}
        self.context = None
        self.last_run = None
        self.llm_suggestions = []
        self.lockfile = os.path.join(PROJECT_ROOT, "pipeline.lock")

    def ensure_db_tables(self):
        try:
            engine = get_engine()
            inspector = inspect(engine)
            # Explicitly reference TableStructure to ensure it's registered with SQLAlchemy's metadata
            _ = TableStructure 
            if 'table_structures' not in inspector.get_table_names():
                Base.metadata.create_all(engine)
                log_info("[MODELS] Creating all tables in the configured database...")
            log_info("[MODELS] Tables present after creation: %s" % inspector.get_table_names())
            log_info("[MODELS] All tables created successfully.")
            log_info("[PIPELINE] DB tables ensured.")
            self.results['db_tables'] = 'success'
            return True
        except Exception as e:
            log_error(f"[PIPELINE] DB table check failed: {e}")
            self.results['db_tables'] = 'fail'
            return False

    def clean_and_migrate(self):
        try:
            errors = run_log_cache_cleaner()
            migrate_all()
            if errors:
                log_error(f"[PIPELINE] Cleaning errors: {errors}")
                self.results['clean_migrate'] = 'fail'
                return False
            self.results['clean_migrate'] = 'success'
            return True
        except Exception as e:
            log_error(f"[PIPELINE] Clean/migrate failed: {e}")
            self.results['clean_migrate'] = 'fail'
            return False

    def scan_misaligned(self):
        try:
            exit_code = scan_misaligned()
            self.results['scan_misaligned'] = 'clean' if exit_code == 0 else 'misaligned'
            return exit_code
        except Exception as e:
            log_error(f"[PIPELINE] scan_misaligned_ner failed: {e}")
            self.results['scan_misaligned'] = 'fail'
            return 2

    def build_correction_args(self):
        args = []
        if os.getenv("ENABLE_ENHANCED", "true").lower() == "true":
            args.append("--enhanced")
        if os.getenv("CORRECTION_MODE", "feedback").lower() == "feedback":
            args.append("--feedback")
        else:
            args.append("--auto")
        if os.getenv("INTEGRITY_CHECK", "false").lower() == "true":
            args.append("--integrity")
        if os.getenv("UPDATE_DB", "true").lower() == "true":
            args.append("--update-db")
        llm_api_key = os.getenv("LLM_API_KEY")
        llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        llm_model = os.getenv("LLM_MODEL", "gpt-4-turbo")
        if llm_api_key:
            args.extend([
                "--llm-api-key", llm_api_key,
                "--llm-provider", llm_provider,
                "--llm-model", llm_model
            ])
            if llm_provider == "anthropic" and os.getenv("ANTHROPIC_SYSTEM_PROMPT"):
                args.extend(["--llm-system-prompt", os.getenv("ANTHROPIC_SYSTEM_PROMPT")])
            elif llm_provider == "gemini" and os.getenv("GEMINI_SYSTEM_PROMPT"):
                args.extend(["--llm-system-prompt", os.getenv("GEMINI_SYSTEM_PROMPT")])
            elif llm_provider == "local" and os.getenv("LOCAL_LLM_PATH"):
                args.extend(["--llm-model-path", os.getenv("LOCAL_LLM_PATH")])
            if os.getenv("LLM_SYSTEM_PROMPT"):
                args.extend(["--llm-system-prompt", os.getenv("LLM_SYSTEM_PROMPT")])
            if os.getenv("LLM_EXTRA_INSTRUCTIONS"):
                args.extend(["--llm-extra-instructions", os.getenv("LLM_EXTRA_INSTRUCTIONS")])
        if os.getenv("FILTER_CONTEXT_KEY"):
            args.extend(["--filter-context-key", os.getenv("FILTER_CONTEXT_KEY")])
        if os.getenv("FILTER_VALUE"):
            args.extend(["--filter-value", os.getenv("FILTER_VALUE")])
        if os.getenv("FIELDS"):
            args.extend(["--fields"] + os.getenv("FIELDS").split(","))
        if os.getenv("CONTEXT_PATH"):
            args.extend(["--context", os.getenv("CONTEXT_PATH")])
        if os.getenv("LOG_DIR"):
            args.extend(["--log-dir", os.getenv("LOG_DIR")])
        if os.getenv("DRY_RUN", "false").lower() == "true":
            args.append("--dry-run")
        if os.getenv("NO_COORDINATOR", "false").lower() == "true":
            args.append("--no-coordinator")
        if os.getenv("NO_ORGANIZER", "false").lower() == "true":
            args.append("--no-organizer")
        if os.getenv("BATCH_MODE", "false").lower() == "true":
            args.append("--batch")
        if os.getenv("FAST_MODE", "false").lower() == "true":
            args.append("--fast")
        if os.getenv("FLUSH_CACHE", "false").lower() == "true":
            args.append("--flush-cache")
        if os.getenv("CACHE_EXPIRE_DAYS"):
            args.extend(["--cache-expire-days", os.getenv("CACHE_EXPIRE_DAYS")])
        if os.getenv("EXPORT_AUDIT_LOG"):
            args.extend(["--export-audit-log", os.getenv("EXPORT_AUDIT_LOG")])
        if os.getenv("REST_API", "false").lower() == "true":
            args.append("--rest-api")
        if os.getenv("SELF_HEAL", "false").lower() == "true":
            args.append("--self-heal")
            if os.getenv("MAX_RETRIES"):
                args.extend(["--max-retries", os.getenv("MAX_RETRIES")])
            if os.getenv("COOLDOWN"):
                args.extend(["--cooldown", os.getenv("COOLDOWN")])
        if os.getenv("DB_PATH"):
            args.extend(["--db-path", os.getenv("DB_PATH")])
        return args

    def has_new_entries(self, log_dir, cache_dir):
        log_files = find_log_files(log_dir, cache_dir)
        for log_file in log_files:
            entries = load_jsonl(log_file)
            if entries:
                return True
        return False

    def lock(self):
        try:
            fd = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write("locked")
            return True
        except FileExistsError:
            print("[INFO] Pipeline already running or ran.")
            return False
        except OSError as e:
            if e.errno == errno.EEXIST:
                print("[INFO] Pipeline already running or ran.")
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
            print(f"\n[SELF-HEAL] Attempt {attempt}...")
            exit_code = self.scan_misaligned()
            if exit_code == 0:
                print("[SELF-HEAL] Data is clean. Exiting self-heal mode.")
                return 0
            print(f"[SELF-HEAL] Misalignments found. Launching manual_correction_bot...")
            self.manual_correction(args=self.build_correction_args())
            print(f"[SELF-HEAL] Sleeping {cooldown}s before rescanning...")
            time.sleep(cooldown)
        print("[SELF-HEAL] Max retries reached. Some misalignments may remain.")
        return 2

    def run(self):
        if not self.lock():
            return
        try:
            self.last_run = datetime.now().isoformat()
            log_info(f"[PIPELINE] Starting pipeline at {self.last_run}")
            if not self.ensure_db_tables():
                return
            if not self.clean_and_migrate():
                return
            misaligned = self.scan_misaligned()
            correction_args = self.build_correction_args()
            # Only run correction if new entries exist
            if self.has_new_entries(LOG_DIR, CACHE_DIR):
                if misaligned != 0:
                    self.self_heal_loop()
                else:
                    self.manual_correction(args=correction_args)
            else:
                log_info("[PIPELINE] No new entries for manual correction. Skipping manual_correction_bot.")
                self.results['manual_correction'] = 'skipped'
            self.retrain_models()
            self.context = load_context_library()
            self.context_postprocess()
            self.run_orchestration_plugins()
            self.self_improve()
            self.print_summary()
        finally:
            self.unlock()

    def manual_correction(self, args=None):
        try:
            cmd = [sys.executable, "-m", "webapp.parser.bots.manual_correction_bot"]
            if args:
                cmd.extend(args)
            subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
            self.results['manual_correction'] = 'success'
            return True
        except Exception as e:
            log_error(f"[PIPELINE] manual_correction_bot failed: {e}")
            self.results['manual_correction'] = 'fail'
            return False

    def retrain_models(self):
        try:
            cmd = [sys.executable, "-m", "webapp.parser.bots.retrain_table_structure_models"]
            subprocess.run(cmd, check=True, cwd=os.environ.get("PROJECT_ROOT", "."))
            self.results['retrain_models'] = 'success'
            return True
        except Exception as e:
            log_error(f"[PIPELINE] retrain_table_structure_models failed: {e}")
            self.results['retrain_models'] = 'fail'
            return False

    def context_postprocess(self):
        try:
            # Example: add your context modules here
            from ..Context_Integration import context_organizer, context_coordinator, Integrity_check
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
            log_error(f"[PIPELINE] Context modules failed: {e}")
            self.results["integrity_check"] = "fail"
            self.results["context_organizer"] = "fail"
            self.results["context_coordinator"] = "fail"

    def run_orchestration_plugins(self):
        try:
            plugin_results = run_orchestration_plugins(self.context)
            self.results["orchestration_plugins"] = "success" if plugin_results else "none"
        except Exception as e:
            log_error(f"[PIPELINE] Orchestration plugins failed: {e}")
            self.results["orchestration_plugins"] = "fail"

    def self_improve(self):
        logs = summarize_logs()
        prompt = (
            "You are an AI pipeline assistant. Given the following pipeline results and logs, "
            "suggest improvements or next steps for the pipeline. "
            "Results: " + orjson.dumps(self.results).decode() +
            "\nLogs:\n" + logs[-1000:]
        )
        suggestion = None
        if openai and os.getenv("LLM_API_KEY"):
            try:
                openai.api_key = os.getenv("LLM_API_KEY")
                response = openai.ChatCompletion.create(
                    model=os.getenv("LLM_MODEL", "gpt-4-turbo"),
                    messages=[{"role": "system", "content": prompt}],
                    max_tokens=256,
                    temperature=0.2,
                )
                suggestion = response.choices[0].message.content
                log_info(f"[PIPELINE][LLM SUGGESTION]: {suggestion}")
                self.llm_suggestions.append(suggestion)
            except Exception as e:
                log_error(f"[PIPELINE][LLM] Suggestion failed: {e}")
        else:
            if self.results.get("scan_misaligned") == "misaligned":
                suggestion = "Consider running manual_correction_bot with --self-heal or retraining models."
            else:
                suggestion = "Pipeline ran clean. Monitor logs for anomalies."
            log_info(f"[PIPELINE][STATIC SUGGESTION]: {suggestion}")
            self.llm_suggestions.append(suggestion)

    def print_summary(self):
        print("\n[PIPELINE] Run Summary:")
        for k, v in self.results.items():
            print(f"  {k:<20}: {v}")
        if self.llm_suggestions:
            print("\n[PIPELINE] LLM/AI Suggestions:")
            for s in self.llm_suggestions:
                print(f"  - {s}")

if __name__ == "__main__":
    pipeline = BotPipeline()
    pipeline.run()
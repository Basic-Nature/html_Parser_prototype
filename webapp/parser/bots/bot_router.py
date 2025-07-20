import os
import sys
import time
import subprocess
import orjson
import errno
from datetime import datetime
from sqlalchemy import inspect
from pathlib import Path
from ..utils.shared_logger import SharedLogger, RichConsoleProxy
from ..bots.librarian import load_context_library
from ..utils.models import Base
from ..utils.db_utils import get_engine
from ..config import LOG_DIR, CACHE_DIR, PROJECT_ROOT
try:
    import openai
except ImportError:
    openai = None

logger = SharedLogger()
console = RichConsoleProxy()
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

def preclean_json_logs(log_dirs, required_files=None):
    """
    Clean all JSON/JSONL files in log_dirs.
    Quarantine corrupt lines, salvage valid lines, and create missing required files.
    """
    import glob
    import os
    import re
    import shutil

    # Clean all .jsonl and .json files
    for log_dir in log_dirs:
        for suf in [".jsonl", ".json"]:
            for path in glob.glob(os.path.join(log_dir, f"*{suf}")):
                valid_lines = []
                corrupt_lines = []
                with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            # Try to parse as JSON
                            import json
                            json.loads(line)
                            valid_lines.append(line)
                        except Exception as e:
                            # Try to fix common issues
                            fixed = line
                            fixed = re.sub(r",\s*$", "", fixed)
                            fixed = re.sub(r"'", '"', fixed)
                            fixed = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', fixed)
                            fixed = fixed.replace('\ufeff', '').replace('\x00', '')
                            try:
                                json.loads(fixed)
                                valid_lines.append(fixed)
                            except Exception:
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
        self.llm_suggestions = []
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

    def run_manual_correction(self, mode="enhanced", extra_args=None, retries=1, timeout=600):
        """
        Optimized wrapper for manual_correction_bot for end-of-pipeline use.
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
        context_path = Path(LOG_DIR) / "context_library.json" if not os.getenv("CONTEXT_PATH") else os.getenv("CONTEXT_PATH")
        args.extend([
            "--context", str(context_path),
            "--log-dir", str(log_dir_path)
        ])
        # Check for new entries before running
        if not self.has_new_entries(LOG_DIR, CACHE_DIR):
            logger.info("[BOT_ROUTER] No new entries for manual correction. Skipping enhanced mode and exiting gracefully.")
            self.results['manual_correction'] = 'skipped'
            return True  # Graceful exit
        # Try running with retries and timeout
        for attempt in range(1, retries + 1):
            try:
                logger.info(f"[BOT_ROUTER] Running manual_correction_bot (enhanced mode, attempt={attempt}) with args: {args}")
                cmd = [sys.executable, "-m", "webapp.parser.bots.manual_correction_bot"] + args
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=timeout)
                logger.info(f"[BOT_ROUTER] manual_correction_bot stdout:\n{result.stdout[:1000]}")
                if result.returncode == 0:
                    logger.info("[BOT_ROUTER] manual_correction_bot completed successfully.")
                    self.results['manual_correction'] = 'success'
                    return True
                else:
                    logger.warning(f"[BOT_ROUTER] manual_correction_bot failed (attempt {attempt}): {result.stderr}")
                    time.sleep(2)
            except subprocess.TimeoutExpired:
                logger.error(f"[BOT_ROUTER] manual_correction_bot timed out after {timeout} seconds (attempt {attempt}).")
            except Exception as e:
                logger.error(f"[BOT_ROUTER] manual_correction_bot exception: {e}")
        logger.error("[BOT_ROUTER] manual_correction_bot failed after all retries.")
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
                sys.executable, "-m", "webapp.parser.bots.manual_correction_bot",
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
            logger.warning(f"[SELF-HEAL] Misalignments found. Launching manual_correction_bot...")
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
                cmd = [sys.executable, "-m", "webapp.parser.bots.manual_correction_bot", "--fix-corrupt-json"]
                subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
                logger.info("[PIPELINE] Corrupted JSON files checked and fixed.")
            except Exception as e:
                logger.warning(f"[PIPELINE] Could not fix corrupted JSON files: {e}")

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
            if has_entries:
                extra_args = []
                # Dynamically add arguments based on pipeline state and env
                if os.getenv("INTEGRITY_CHECK", "false").lower() == "true":
                    extra_args.append("--integrity")
                if os.getenv("LLM_API_KEY"):
                    extra_args.extend([
                        "--llm-api-key", os.getenv("LLM_API_KEY"),
                        "--llm-provider", os.getenv("LLM_PROVIDER", "openai"),
                        "--llm-model", os.getenv("LLM_MODEL", "gpt-4-turbo")
                    ])
                if os.getenv("EXPORT_AUDIT_LOG"):
                    extra_args.extend(["--export-audit-log", os.getenv("EXPORT_AUDIT_LOG")])
                if os.getenv("FLUSH_CACHE", "false").lower() == "true":
                    extra_args.append("--flush-cache")
                if os.getenv("CACHE_EXPIRE_DAYS"):
                    extra_args.extend(["--cache-expire-days", os.getenv("CACHE_EXPIRE_DAYS")])
                # Always run in auto mode for end-of-pipeline
                logger.info("[PIPELINE] Running manual_correction_bot in auto mode for context correction.")
                self.run_manual_correction(mode="auto", extra_args=extra_args)
            else:
                logger.info("[PIPELINE] No new entries for manual correction. Skipping manual_correction_bot.")
                self.results['manual_correction'] = 'skipped'
                return

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

            # 10. Self-improvement suggestions (LLM/static)
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
            cmd = [sys.executable, "-m", "webapp.parser.bots.manual_correction_bot"]
            if args:
                cmd.extend(args)
            subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
            self.results['manual_correction'] = 'success'
            return True
        except Exception as e:
            logger.error(f"[PIPELINE] manual_correction_bot failed: {e}")
            self.results['manual_correction'] = 'fail'
            return False

    def retrain_models(self):
        try:
            cmd = [sys.executable, "-m", "webapp.parser.bots.retrain_table_structure_models"]
            subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
            self.results['retrain_models'] = 'success'
            return True
        except Exception as e:
            logger.error(f"[PIPELINE] retrain_table_structure_models failed: {e}")
            self.results['retrain_models'] = 'fail'
            return False

    def scan_misaligned(self):
        try:
            cmd = [sys.executable, "-m", "webapp.parser.bots.scan_misaligned_ner"]
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
            cmd = [sys.executable, "-m", "webapp.parser.bots.log_cache_cleaner_bot"]
            subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
            cmd = [sys.executable, "-m", "webapp.parser.bots.context_migration"]
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
        logs = logger.summarize_logs()
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
                logger.info(f"[PIPELINE][LLM SUGGESTION]: {suggestion}")
                self.llm_suggestions.append(suggestion)
            except Exception as e:
                logger.error(f"[PIPELINE][LLM] Suggestion failed: {e}")
        else:
            if self.results.get("scan_misaligned") == "misaligned":
                suggestion = "Consider running manual_correction_bot with --self-heal or retraining models."
            else:
                suggestion = "Pipeline ran clean. Monitor logs for anomalies."
            logger.info(f"[PIPELINE][STATIC SUGGESTION]: {suggestion}")
            self.llm_suggestions.append(suggestion)

    def print_summary(self):
        logger.info("\n[PIPELINE] Run Summary:")
        for k, v in self.results.items():
            logger.info(f"  {k:<20}: {v}")
        if self.llm_suggestions:
            console.print("\n[PIPELINE] LLM/AI Suggestions:")
            for s in self.llm_suggestions:
                console.print(f"  - {s}")

if __name__ == "__main__":
    pipeline = BotPipeline()
    pipeline.run()
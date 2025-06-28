import logging
import subprocess
import sys
import os
import json
import time
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from ..config import PROJECT_ROOT, BASE_DIR, POSTGRES_URL
import smtplib
from email.message import EmailMessage
import requests
from requests.auth import HTTPBasicAuth
from ..bots.log_cleaner_bot import run_log_cleaner
from ..utils.context_migration import migrate_all
from ..bots.scan_misaligned_ner import scan_misaligned
from ..Context_Integration import context_organizer, context_coordinator, Integrity_check
from ..bots.librarian import load_context_library

try:
    import openai
except ImportError:
    openai = None
    
## --- Optional arguments for manual_correction_bot ---
"""
--feedback: Enables interactive review of each new entry (user must approve/edit/remove).
--auto: Automatically accepts all new entries (no user prompt).
--enhanced: Enables advanced ML, spaCy, and LLM-based suggestions and learning.
--integrity: Runs anomaly/integrity checks on the context library.
--update-db: Writes the updated context library to the database.
--llm-api-key, --llm-provider, --llm-model: Use an external LLM (OpenAI/Anthropic) for suggestions and corrections.
--llm-system-prompt, --llm-extra-instructions: Customizes the LLM’s behavior and instructions.
--fields: Restricts processing to specific fields (e.g., only "contests" or "states").
--context, --log-dir: Custom paths for context library and logs.
--filter-context-key, --filter-value: Only process entries matching these filters.
--dry-run: Show what would change, but do not write to disk.
--no-coordinator, --no-organizer: Disable advanced context/ML integrations.
--db-path: Custom path for the DB file.
Argument Grouping Examples:

For full automation: --auto --enhanced --update-db
For manual review: --feedback --enhanced --integrity --update-db
For LLM-powered review: --feedback --enhanced --llm-api-key ... --llm-provider openai --llm-model gpt-4-turbo
For field-specific correction: --fields contests states --feedback --enhanced
"""
ORCHESTRATION_PLUGINS = []

WEBAPP_DIR = BASE_DIR
DEFAULT_LOG_DIR = os.path.join(PROJECT_ROOT, "log")

def register_orchestration_plugin(plugin_func):
    """Register a plugin function for orchestration logic."""
    ORCHESTRATION_PLUGINS.append(plugin_func)

def run_orchestration_plugins(context=None):
    """Run all registered orchestration plugins and collect bot suggestions."""
    suggestions = []
    for plugin in ORCHESTRATION_PLUGINS:
        try:
            suggestions.extend(plugin(context))
        except Exception as e:
            print(f"[BOT ROUTER][PLUGIN ERROR] {e}")
    return suggestions

BOT_MODULES = {
    "retrain_table_structure_models": "webapp.parser.bots.retrain_table_structure_models",
    "manual_correction_bot": "webapp.parser.bots.manual_correction_bot",
    "scan_misaligned_ner": "webapp.parser.bots.scan_misaligned_ner",
    
    # Add more bots here as needed
}

def check_db_connection():
    """Check if the database is available before running DB-dependent bots."""
    try:
        engine = create_engine(POSTGRES_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"[BOT ROUTER][ERROR] Database unavailable: {e}")
        return False

def run_bot_task(bot_name, args=None, context=None, self_heal=False, max_retries=3, cooldown=2):
    """
    Run a bot by name with optional arguments and self-heal mode.
    Args:
        bot_name: str, key in BOT_MODULES
        args: list of str, command-line arguments
        context: dict, optional context for future extension
        self_heal: bool, enable self-heal loop for supported bots
        max_retries: int, max attempts for self-heal loop
        cooldown: int, cooldown period between self-heal attempts (seconds)
    """
    if self_heal:
        return self_heal_loop(bot_name, args, max_retries, cooldown)
    module = BOT_MODULES.get(bot_name)
    if not module:
        print(f"[ERROR] Unknown bot: {bot_name}")
        return False
    cmd = [sys.executable, "-m", module]
    if args:
        cmd.extend(args)
    print(f"[BOT ROUTER] Running bot: {bot_name} ({' '.join(cmd)})")
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    try:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)
        return True
    except Exception as e:
        print(f"[BOT ROUTER][ERROR] Failed to run {bot_name}: {e}")
        if bot_name == "retrain_table_structure_models":
            print("[HINT] If on Windows, ensure no file explorer or editor is open on the model directory and try again.")
        return False

def run_subprocess_module(module, args=None, env=None):
    cmd = [sys.executable, "-m", module]
    if args:
        cmd.extend(args)
    print(f"[BOT ROUTER] Running: {' '.join(cmd)}")
    env = env or os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    try:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)
        return True
    except Exception as e:
        print(f"[BOT ROUTER][ERROR] Failed to run {module}: {e}")
        return False

def print_bot_summary(results):
    print("\n[BOT ROUTER] Pipeline Summary:")
    print("Bot Name                   | Status")
    print("---------------------------|--------")
    for bot, status in results.items():
        print(f"{bot:<27}| {status}")

def run_pipeline():
    """Run the main bot pipeline with DB checks and summary output."""
    results = {}
    print("[BOT ROUTER] Step 1: Cleaning logs/context and migrating to PostgreSQL...")
    try:
        run_log_cleaner()
        migrate_all()
        results["log_cleaner_bot"] = "success"
        results["context_migration"] = "success"
    except Exception as e:
        print(f"[BOT ROUTER][ERROR] Log cleaning or migration failed: {e}")
        results["log_cleaner_bot"] = "fail"
        results["context_migration"] = "fail"
        print_bot_summary(results)
        return

    # 2. Check DB before running DB-dependent bots
    if not check_db_connection():
        print("[BOT ROUTER] Skipping DB-dependent bots: database is not available.")
        results["scan_misaligned_ner"] = "skipped"
        results["manual_correction_bot"] = "skipped"
        results["retrain_table_structure_models"] = "skipped"
        print_bot_summary(results)
        return

    # 3. Scan for misaligned NER examples
    print("[BOT ROUTER] Step 2: Scanning for misaligned NER examples...")
    try:
        misaligned_exit_code = scan_misaligned()
        if misaligned_exit_code == 0:
            results["scan_misaligned_ner"] = "clean"
        else:
            results["scan_misaligned_ner"] = "misaligned"
            print("[BOT ROUTER][WARNING] Misaligned NER examples found. Running manual_correction_bot before retraining.")
    except Exception as e:
        print(f"[BOT ROUTER][ERROR] scan_misaligned_ner failed: {e}")
        results["scan_misaligned_ner"] = "fail"

    # 4. Build dynamic arguments for manual_correction_bot
    correction_args = []
    # Use enhanced ML/NER/LLM if available
    if os.getenv("ENABLE_ENHANCED", "true").lower() == "true":
        correction_args.append("--enhanced")
    # Use feedback loop if user wants review, else auto-accept
    if os.getenv("CORRECTION_MODE", "feedback").lower() == "feedback":
        correction_args.append("--feedback")
    else:
        correction_args.append("--auto")
    # Integrity check if flagged or in production
    if os.getenv("INTEGRITY_CHECK", "false").lower() == "true":
        correction_args.append("--integrity")
    # Always update DB if in production or as needed
    if os.getenv("UPDATE_DB", "true").lower() == "true":
        correction_args.append("--update-db")
    # Use LLM if API key is present
    llm_api_key = os.getenv("LLM_API_KEY")
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    llm_model = os.getenv("LLM_MODEL", "gpt-4-turbo")
    if llm_api_key:
        correction_args.extend([
            "--llm-api-key", llm_api_key,
            "--llm-provider", llm_provider,
            "--llm-model", llm_model
        ])
        # Anthropic-specific options
        if llm_provider == "anthropic" and os.getenv("ANTHROPIC_SYSTEM_PROMPT"):
            correction_args.extend(["--llm-system-prompt", os.getenv("ANTHROPIC_SYSTEM_PROMPT")])
        elif llm_provider == "gemini" and os.getenv("GEMINI_SYSTEM_PROMPT"):
            correction_args.extend(["--llm-system-prompt", os.getenv("GEMINI_SYSTEM_PROMPT")])
        elif llm_provider == "local" and os.getenv("LOCAL_LLM_PATH"):
            correction_args.extend(["--llm-model-path", os.getenv("LOCAL_LLM_PATH")])
        if os.getenv("LLM_SYSTEM_PROMPT"):
            correction_args.extend(["--llm-system-prompt", os.getenv("LLM_SYSTEM_PROMPT")])
        if os.getenv("LLM_EXTRA_INSTRUCTIONS"):
            correction_args.extend(["--llm-extra-instructions", os.getenv("LLM_EXTRA_INSTRUCTIONS")])
    # Filter by context key or value if set
    if os.getenv("FILTER_CONTEXT_KEY"):
        correction_args.extend(["--filter-context-key", os.getenv("FILTER_CONTEXT_KEY")])
    if os.getenv("FILTER_VALUE"):
        correction_args.extend(["--filter-value", os.getenv("FILTER_VALUE")])
    # Specify fields if needed
    if os.getenv("FIELDS"):
        correction_args.extend(["--fields"] + os.getenv("FIELDS").split(","))
    # Use custom context or log-dir if set
    if os.getenv("CONTEXT_PATH"):
        correction_args.extend(["--context", os.getenv("CONTEXT_PATH")])
    if os.getenv("LOG_DIR"):
        correction_args.extend(["--log-dir", os.getenv("LOG_DIR")])
    # Dry-run mode
    if os.getenv("DRY_RUN", "false").lower() == "true":
        correction_args.append("--dry-run")
    # Disable coordinator/organizer if needed
    if os.getenv("NO_COORDINATOR", "false").lower() == "true":
        correction_args.append("--no-coordinator")
    if os.getenv("NO_ORGANIZER", "false").lower() == "true":
        correction_args.append("--no-organizer")
    # Advanced features from manual_correction_bot
    if os.getenv("BATCH_MODE", "false").lower() == "true":
        correction_args.append("--batch")
    if os.getenv("FAST_MODE", "false").lower() == "true":
        correction_args.append("--fast")
    if os.getenv("FLUSH_CACHE", "false").lower() == "true":
        correction_args.append("--flush-cache")
    if os.getenv("CACHE_EXPIRE_DAYS"):
        correction_args.extend(["--cache-expire-days", os.getenv("CACHE_EXPIRE_DAYS")])
    if os.getenv("EXPORT_AUDIT_LOG"):
        correction_args.extend(["--export-audit-log", os.getenv("EXPORT_AUDIT_LOG")])
    if os.getenv("REST_API", "false").lower() == "true":
        correction_args.append("--rest-api")
    if os.getenv("SELF_HEAL", "false").lower() == "true":
        correction_args.append("--self-heal")
        if os.getenv("MAX_RETRIES"):
            correction_args.extend(["--max-retries", os.getenv("MAX_RETRIES")])
        if os.getenv("COOLDOWN"):
            correction_args.extend(["--cooldown", os.getenv("COOLDOWN")])
    if os.getenv("DB_PATH"):
        correction_args.extend(["--db-path", os.getenv("DB_PATH")])

    # 5. Run manual correction bot for misalignments or general cleanup
    correction_success = None
    if results.get("scan_misaligned_ner") == "misaligned":
        print("[BOT ROUTER] Step 3: Running manual correction bot for misalignments...")
        # Always include --fields contests for targeted misalignment cleanup
        misalignment_args = ["--fields", "contests"] + correction_args
        correction_success = run_subprocess_module(
            "webapp.parser.bots.manual_correction_bot",
            args=misalignment_args
        )
        results["manual_correction_bot"] = "success" if correction_success else "fail"
    else:
        print("[BOT ROUTER] Step 3: Running manual correction bot for general structure cleanup...")
        correction_success = run_subprocess_module(
            "webapp.parser.bots.manual_correction_bot",
            args=correction_args
        )
        results["manual_correction_bot"] = "success" if correction_success else "fail"

    # 6. Retrain table structure models (after correction)
    print("[BOT ROUTER] Step 4: Retraining table structure models...")
    retrain_success = run_subprocess_module("webapp.parser.bots.retrain_table_structure_models")
    results["retrain_table_structure_models"] = "success" if retrain_success else "fail"

    # 7. Run orchestration plugins (context_organizer, context_coordinator, integrity_check, librarian, etc.)
    print("[BOT ROUTER] Step 5: Running orchestration plugins and context modules...")
    context = load_context_library()
    print("Type of context:", type(context))
    try:
        contests = context.get("contests", [])
        if contests:
            Integrity_check.print_integrity_summary(contests)
            results["integrity_check"] = "success"
        else:
            results["integrity_check"] = "no_contests"
        organizer = context_organizer.ContextOrganizer()
        organized = organizer.organize_context(context)
        results["context_organizer"] = "success"
        coordinator = context_coordinator.ContextCoordinator()
        coordinator.organize_and_enrich(context)
        results["context_coordinator"] = "success"
    except Exception as e:
        print(f"[BOT ROUTER][ERROR] Context modules failed: {e}")
        results["integrity_check"] = "fail"
        results["context_organizer"] = "fail"
        results["context_coordinator"] = "fail"

    # 8. Run any additional orchestration plugins
    try:
        plugin_results = run_orchestration_plugins(context)
        results["orchestration_plugins"] = "success" if plugin_results else "none"
    except Exception as e:
        print(f"[BOT ROUTER][ERROR] Orchestration plugins failed: {e}")
        results["orchestration_plugins"] = "fail"

    print_bot_summary(results)
   
def scan_and_notify(context):
    """
    Scan for new results in the context and send notifications if new or important results are found.
    This function can be extended to check for specific keys, statuses, or thresholds.
    """
    logging.info("[BOT] Scanning for new results and sending notifications...")
    new_results = []
    # Example: Scan for new contests or results
    if context and "contests" in context:
        for contest in context["contests"]:
            if contest.get("status") == "new" or contest.get("notify", False):
                new_results.append(contest)
    if new_results:
        for result in new_results:
            message = f"New contest detected: {result.get('name', 'Unknown')}"
            send_notification(message, context=result)
        logging.info(f"[BOT] Notifications sent for {len(new_results)} new results.")
        return True
    else:
        logging.info("[BOT] No new results found for notification.")
        return False

def batch_status_report(context):
    """
    Generate a batch status report from the context and optionally send or log it.
    """
    logging.info("[BOT] Generating batch status report...")
    report = []
    if context and "contests" in context:
        for contest in context["contests"]:
            status = contest.get("status", "unknown")
            name = contest.get("name", "Unnamed")
            report.append(f"{name}: {status}")
    report_text = "\n".join(report)
    if report_text:
        logging.info(f"[BOT] Batch Status Report:\n{report_text}")
        # Optionally, send the report via notification
        send_notification("Batch Status Report:\n" + report_text)
        return report_text
    else:
        logging.info("[BOT] No contests found for batch status report.")
        return ""

def send_notification(message, context=None, email=None):
    """
    Send a notification. Supports email, Slack, and SMS (Twilio).
    """
    logging.info(f"[BOT] Sending notification: {message}")

    # Email notification
    if os.getenv("NOTIFY_EMAILS", "false").lower() == "true":
        email_to = email or os.getenv("NOTIFY_EMAIL")
        if email_to:
            try:
                smtp_server = os.getenv("SMTP_SERVER", "localhost")
                smtp_port = int(os.getenv("SMTP_PORT", 25))
                smtp_user = os.getenv("SMTP_USER")
                smtp_pass = os.getenv("SMTP_PASS")
                msg = EmailMessage()
                msg.set_content(message)
                msg["Subject"] = "Pipeline Notification"
                msg["From"] = os.getenv("SMTP_FROM", "noreply@example.com")
                msg["To"] = email_to
                with smtplib.SMTP(smtp_server, smtp_port) as server:
                    if smtp_user and smtp_pass:
                        server.starttls()
                        server.login(smtp_user, smtp_pass)
                    server.send_message(msg)
                logging.info(f"[BOT] Email notification sent to {email_to}")
            except Exception as e:
                logging.error(f"[BOT] Failed to send email notification: {e}")

    # Slack notification
    if os.getenv("NOTIFY_SLACK", "false").lower() == "true":
        slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
        if slack_webhook:
            try:
                slack_data = {"text": message}
                resp = requests.post(slack_webhook, json=slack_data)
                if resp.status_code == 200:
                    logging.info("[BOT] Slack notification sent.")
                else:
                    logging.error(f"[BOT] Slack notification failed: {resp.text}")
            except Exception as e:
                logging.error(f"[BOT] Failed to send Slack notification: {e}")

    # SMS notification (Twilio)
    if os.getenv("NOTIFY_SMS", "false").lower() == "true":
        twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
        twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
        twilio_from = os.getenv("TWILIO_FROM")
        twilio_to = os.getenv("TWILIO_TO")
        if twilio_sid and twilio_token and twilio_from and twilio_to:
            try:
                sms_url = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_sid}/Messages.json"
                sms_data = {
                    "From": twilio_from,
                    "To": twilio_to,
                    "Body": message
                }
                resp = requests.post(
                    sms_url,
                    data=sms_data,
                    auth=HTTPBasicAuth(twilio_sid, twilio_token)
                )
                if resp.status_code == 201:
                    logging.info("[BOT] SMS notification sent.")
                else:
                    logging.error(f"[BOT] SMS notification failed: {resp.text}")
            except Exception as e:
                logging.error(f"[BOT] Failed to send SMS notification: {e}")

    return True

def get_file_age_days(path):
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    return (datetime.now() - datetime.fromtimestamp(mtime)).days

def should_run_correction_bot(log_dir, last_run_time):
    if not os.path.isdir(log_dir):  # Only proceed if directory exists
        return False
    for fname in os.listdir(log_dir):
        if fname.endswith("_selection_log.jsonl"):
            if os.path.getmtime(os.path.join(log_dir, fname)) > last_run_time:
                return True
    return False

def summarize_logs(log_dir=DEFAULT_LOG_DIR, max_lines=100):
    """Summarize recent logs for AI context."""
    logs = []
    if not os.path.isdir(log_dir):
        return ""
    for fname in os.listdir(log_dir):
        if fname.endswith(".log") or fname.endswith(".jsonl"):
            with open(os.path.join(log_dir, fname), encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    logs.append(line.strip())
    return "\n".join(logs[-max_lines:])

def ai_suggest_bots(context=None):
    """
    Use an LLM (OpenAI) to suggest which bots to run and with what arguments.
    """
    suggestions = []
    context = context or {}
    log_dir = os.getenv("LOG_DIR", DEFAULT_LOG_DIR)
    logs_summary = summarize_logs(log_dir)
    # Gather context for the LLM
    context = context or {}
    log_dir = os.getenv("LOG_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "log"))
    logs_summary = summarize_logs(log_dir)
    model_path = os.path.join(os.path.dirname(__file__), "..", "Context_Integration", "Context_Library", "table_structure_model.pkl")
    model_age = get_file_age_days(model_path)
    env_vars = {k: v for k, v in os.environ.items() if k.startswith("LLM_") or k in [
        "ENABLE_ENHANCED", "CORRECTION_MODE", "INTEGRITY_CHECK", "UPDATE_DB", "FIELDS"
    ]}
    prompt = f"""
You are an AI assistant for election data pipeline automation. 
Given the following context, suggest which bots to run and with what arguments. 
Respond as a JSON list of objects: [{{"bot": "bot_name", "args": ["--arg1", ...]}}]

Context:
- Model file age (days): {model_age}
- Environment variables: {json.dumps(env_vars)}
- Recent logs: {logs_summary[:1000]}
- Known bots: {list(BOT_MODULES.keys())}

Rules:
- If the model is missing or older than 7 days, suggest retrain_table_structure_models.
- If ENABLE_ENHANCED is true, use --enhanced for manual_correction_bot.
- If CORRECTION_MODE is feedback, use --feedback, else --auto.
- If LLM_API_KEY is set, use LLM arguments.
- Always suggest manual_correction_bot with appropriate args.
- Suggest scan_and_notify if logs mention 'new results'.
- Suggest batch_status_report if logs mention 'batch' or 'status'.
- Only suggest bots that exist in BOT_MODULES or are implemented below.
"""
    # Use OpenAI if available and API key is set
    if openai and os.getenv("LLM_API_KEY"):
        try:
            openai.api_key = os.getenv("LLM_API_KEY")
            response = openai.ChatCompletion.create(
                model=os.getenv("LLM_MODEL", "gpt-4-turbo"),
                messages=[{"role": "system", "content": prompt}],
                max_tokens=512,
                temperature=0.2,
            )
            content = response.choices[0].message.content
            ai_suggestions = json.loads(content)
            for item in ai_suggestions:
                bot = item.get("bot")
                args = item.get("args", [])
                if bot in BOT_MODULES or bot in ("scan_and_notify", "batch_status_report"):
                    suggestions.append((bot, args))
        except Exception as e:
            print(f"[BOT ROUTER][AI] LLM suggestion failed: {e}")
    else:
        # Fallback to static logic if no LLM
        suggestions.extend(suggest_bots(context))
    return suggestions

def suggest_bots(context=None):
    """
    Suggest bots to run, with dynamic argument selection for manual_correction_bot.
    This is where you can add Auto-GPT-like logic.
    """
    suggestions = []
    # --- Example: Always suggest retrainer if model is missing or old ---
    model_path = os.path.join(os.path.dirname(__file__), "..", "Context_Integration", "Context_Library", "table_structure_model.pkl")
    model_age = get_file_age_days(model_path)
    if model_age is None or model_age > 7:
        suggestions.append(("retrain_table_structure_models", []))

    # --- Dynamic argument selection for manual_correction_bot ---
    correction_args = []

    # Use enhanced ML/NER/LLM if available
    if os.getenv("ENABLE_ENHANCED", "true").lower() == "true":
        correction_args.append("--enhanced")

    # Use feedback loop if user wants review, else auto-accept
    if os.getenv("CORRECTION_MODE", "feedback").lower() == "feedback":
        correction_args.append("--feedback")
    else:
        correction_args.append("--auto")

    # Integrity check if flagged or in production
    if os.getenv("INTEGRITY_CHECK", "false").lower() == "true":
        correction_args.append("--integrity")

    # Always update DB if in production or as needed
    if os.getenv("UPDATE_DB", "true").lower() == "true":
        correction_args.append("--update-db")

    # Use LLM if API key is present
    # Use LLM if API key is present
    llm_api_key = os.getenv("LLM_API_KEY")
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    llm_model = os.getenv("LLM_MODEL", "gpt-4-turbo")
    if llm_api_key:
        correction_args.extend([
            "--llm-api-key", llm_api_key,
            "--llm-provider", llm_provider,
            "--llm-model", llm_model
        ])
        # Anthropic-specific options
        if llm_provider == "anthropic":
            if os.getenv("ANTHROPIC_SYSTEM_PROMPT"):
                correction_args.extend(["--llm-system-prompt", os.getenv("ANTHROPIC_SYSTEM_PROMPT")])
        # Gemini-specific options
        elif llm_provider == "gemini":
            if os.getenv("GEMINI_SYSTEM_PROMPT"):
                correction_args.extend(["--llm-system-prompt", os.getenv("GEMINI_SYSTEM_PROMPT")])
        # Local LLM options
        elif llm_provider == "local":
            if os.getenv("LOCAL_LLM_PATH"):
                correction_args.extend(["--llm-model-path", os.getenv("LOCAL_LLM_PATH")])
        # General options (applies to all providers)
        if os.getenv("LLM_SYSTEM_PROMPT"):
            correction_args.extend(["--llm-system-prompt", os.getenv("LLM_SYSTEM_PROMPT")])
        if os.getenv("LLM_EXTRA_INSTRUCTIONS"):
            correction_args.extend(["--llm-extra-instructions", os.getenv("LLM_EXTRA_INSTRUCTIONS")])
            
    # Filter by context key or value if set
    if os.getenv("FILTER_CONTEXT_KEY"):
        correction_args.extend(["--filter-context-key", os.getenv("FILTER_CONTEXT_KEY")])
    if os.getenv("FILTER_VALUE"):
        correction_args.extend(["--filter-value", os.getenv("FILTER_VALUE")])

    # Specify fields if needed
    if os.getenv("FIELDS"):
        correction_args.extend(["--fields"] + os.getenv("FIELDS").split(","))

    # Use custom context or log-dir if set
    if os.getenv("CONTEXT_PATH"):
        correction_args.extend(["--context", os.getenv("CONTEXT_PATH")])
    if os.getenv("LOG_DIR"):
        correction_args.extend(["--log-dir", os.getenv("LOG_DIR")])

    suggestions.append(("manual_correction_bot", correction_args))

    # Example: If a certain log file exists, suggest a notification bot
    log_path = os.path.join(os.path.dirname(__file__), "some_log.txt")
    if os.path.exists(log_path):
        suggestions.append(("scan_and_notify", []))

    # Suggest batch_status_report if logs mention 'batch' or 'status'
    log_dir = os.getenv("LOG_DIR", DEFAULT_LOG_DIR)
    last_run_time = time.time() - 3600  # Example: last hour
    if should_run_correction_bot(log_dir, last_run_time):
        suggestions.append(("manual_correction_bot", correction_args))
    if os.path.isdir(log_dir) and should_run_correction_bot(log_dir, last_run_time):
        suggestions.append(("manual_correction_bot", correction_args))

    suggestions.extend(run_orchestration_plugins(context))
    return suggestions

# Attach both suggestion engines for flexibility
run_bot_task.suggest_bots = suggest_bots
run_bot_task.ai_suggest_bots = ai_suggest_bots

def self_heal_loop(bot_name, args=None, max_retries=3, cooldown=2):
    """Loop: scan -> correct -> rescan, until clean or max_retries reached."""
    scan_script = os.path.join(os.path.dirname(__file__), "scan_misaligned_ner.py")
    for attempt in range(1, max_retries + 1):
        print(f"\n[SELF-HEAL] Attempt {attempt}...")
        scan_cmd = [sys.executable, scan_script, "--jsonl", "log/spacy_ner_train_data.jsonl"]
        scan_result = subprocess.run(scan_cmd, check=True, cwd=PROJECT_ROOT, capture_output=True, text=True)
        if scan_result.returncode == 0:
            print("[SELF-HEAL] Data is clean. Exiting self-heal mode.")
            return 0
        print(f"[SELF-HEAL] Misalignments found. Launching {bot_name}...")
        bot_cmd = [sys.executable, "-m", f"webapp.parser.bots.{bot_name}"]
        if args:
            bot_cmd.extend(args)
        subprocess.run(bot_cmd, check=True, cwd=PROJECT_ROOT)
        print(f"[SELF-HEAL] Sleeping {cooldown}s before rescanning...")
        time.sleep(cooldown)
    print("[SELF-HEAL] Max retries reached. Some misalignments may remain.")
    return 2

# Optional: If you want to run this as a script directly
if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not __debug__:
        run_pipeline()

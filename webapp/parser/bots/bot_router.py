import logging
import subprocess
import sys
import os
import json
from datetime import datetime, time

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
    # Add more bots here as needed
}

def run_bot_task(bot_name, args=None, context=None):
    """
    Run a bot by name with optional arguments.
    Args:
        bot_name: str, key in BOT_MODULES
        args: list of str, command-line arguments
        context: dict, optional context for future extension
    """
    module = BOT_MODULES.get(bot_name)
    if not module:
        print(f"[ERROR] Unknown bot: {bot_name}")
        return
    cmd = [sys.executable, "-m", module]
    if args:
        cmd.extend(args)
    print(f"[BOT ROUTER] Running bot: {bot_name} ({' '.join(cmd)})")
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"[BOT ROUTER][ERROR] Failed to run {bot_name}: {e}")
        if bot_name == "retrain_table_structure_models":
            print("[HINT] If on Windows, ensure no file explorer or editor is open on the model directory and try again.")

def scan_and_notify(context):
    logging.info("[BOT] Scanning for new results and sending notifications (not yet implemented).")
    return True

def batch_status_report(context):
    logging.info("[BOT] Generating batch status report (not yet implemented).")
    return True

def send_notification(message, context=None):
    logging.info(f"[BOT] Sending notification: {message}")
    return True

def get_file_age_days(path):
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    return (datetime.now() - datetime.fromtimestamp(mtime)).days

def summarize_logs(log_dir, max_lines=100):
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
    def should_run_correction_bot(log_dir, last_run_time):
        for fname in os.listdir(log_dir):
            if fname.endswith("_selection_log.jsonl"):
                if os.path.getmtime(os.path.join(log_dir, fname)) > last_run_time:
                    return True
        return False
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
    log_dir = os.getenv("LOG_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "log"))
    last_run_time = time.time() - 3600  # Example: last hour
    if should_run_correction_bot(log_dir, last_run_time):
        suggestions.append(("manual_correction_bot", correction_args))


    suggestions.extend(run_orchestration_plugins(context))
    return suggestions

# Attach both suggestion engines for flexibility
run_bot_task.suggest_bots = suggest_bots
run_bot_task.ai_suggest_bots = ai_suggest_bots
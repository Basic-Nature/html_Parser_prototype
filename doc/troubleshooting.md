# 🛠️ Troubleshooting Guide

## ❗ Problem: Parser exits without processing any data

- **Possible Cause**: Missing or malformed URL in `urls.txt`, or all URLs already marked as processed.
- **Fix**:
  - Check that your `urls.txt` file contains at least one valid, full URL (not commented out).
  - Use the interactive prompt to select URLs.
  - If using `.processed_urls` caching, set `CACHE_RESET=true` in `.env` to clear the cache.

### ❗ Problem: No handler found for the URL

- **Possible Cause**: `state_router.py` failed to match the state or county.
- **Fix**:
  - Ensure the domain or state name in the URL matches what's listed in `state_router.py`.
  - Add or update a handler for the state/county.
  - Confirm fallback to `format_router.py` is working (see logs).

### ❗ Problem: User prompt not appearing or not working

- **Possible Cause**: Not using `prompt_user_input()` everywhere.
- **Fix**:
  - Ensure all user input is routed through `prompt_user_input()` from `utils/user_prompt.py`.
  - This is required for both CLI and future web UI compatibility.

### ❗ Problem: CAPTCHA triggered but no browser appears

- **Possible Cause**: Browser is running in headless mode.
- **Fix**:
  - Set `HEADLESS=false` in your `.env` file.
  - Set `SHOW_BROWSER_ON_CAPTCHA=true` in `.env`.
  - Verify you’ve installed the proper Playwright browser binaries:

    ```bash
    playwright install
    ```

### ❗ Problem: CAPTCHA page stuck or browser keeps refreshing

- **Fix**:
  - Manually refresh the page once.
  - Ensure JavaScript and cookies are enabled.
  - Try switching User-Agent (rotate via `.env` or update `user_agents.py`).
  - If using a VPN or proxy, try disabling it.

### ❗ Problem: Output file not written

- **Possible Cause**: No data returned from handler, or handler returned wrong tuple structure.
- **Fix**:
  - Confirm handler returns a `(headers, data, contest_title, metadata)` tuple.
  - Ensure `metadata` includes at least `state` and `race` to build the output path.
  - Check logs for `[WARN] No output file path returned from parser.`

### ❗ Problem: CSV headers mismatch or missing columns

- **Fix**:
  - Use utilities like `utils.table_utils.normalize_headers()` to ensure consistent naming.
  - Validate all candidate-method combinations are included.
  - Check for noisy labels or patterns interfering with contest selection.

### ❗ Problem: PDF/CSV/JSON file not found

- **Possible Cause**: Dynamic downloads failed or input folder not scanned.
- **Fix**:
  - Check that the file exists in `input/`.
  - Confirm download logic in `download_utils.py` is functioning.
  - Use `ENABLE_DOWNLOAD_DISCOVERY=true` in `.env` to allow automatic retrieval.
  - For manual override, ensure `FORCE_PARSE_INPUT_FILE=true` and `FORCE_PARSE_FORMAT` are set in `.env`.

### ❗ Problem: Manual file parsing not working

- **Possible Cause**: Wrong file extension or missing handler.
- **Fix**:
  - Place the file in `input/` with the correct extension.
  - Ensure a handler for the format exists and is registered in `format_router.py`.
  - Use the interactive prompt to select the file.

### ❗ Problem: Bot tasks not running

- **Possible Cause**: Bot integration not enabled or `bot_router.py` missing.
- **Fix**:
  - Set `ENABLE_BOT_TASKS=true` in `.env`.
  - Ensure `bots/bot_router.py` exists and exports `run_bot_task`.
  - Check logs for bot task execution.

---

## 🧪 Debugging Tips

- Use `DEBUG_MODE=true` in `.env` to enable verbose logging.
- Use `print_dom_structure()` utility in `html_scanner.py` for debugging site layout.
- Log User-Agent string to verify spoofing effectiveness.
- Manually test URLs in a normal browser before scripting.
- Check logs for `[ERROR] Handler returned unexpected structure` for tuple/return issues.
- Use `CACHE_RESET=true` to clear processed URL cache if needed.

---

## 📚 Reference & Deeper Debugging

- See [`architecture.md`](architecture.md) for data flow and module responsibilities.
- See [`handlers.md`](handlers.md) for handler development and return structure.
- See [`README.md`](../README.md) for install and usage basics.

---

## 📫 Still stuck?

- Open a GitHub issue with your traceback and `urls.txt` sample.
- Include screenshots if possible.
- Reference the docs above to verify the data flow and handler structure.

---

Happy parsing! 🗳️

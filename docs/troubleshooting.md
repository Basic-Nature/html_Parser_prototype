---
layout: default
---

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

### ❗ Problem: Selenium fallback not available

- **Possible Cause**: The optional SeleniumBase dependency is not installed.
- **Fix**:
  - Install SeleniumBase only when you need the manual CAPTCHA workflow: `pip install seleniumbase`.
  - Leave it uninstalled to keep default Playwright-only runs lighter and avoid pytest plugin hooks.
  - If you re-enable SeleniumBase, run pytest with `-p no:seleniumbase` if its plugin causes conflicts.

### ❗ Problem: Output file not written

- **Possible Cause**: No data returned from handler, or handler returned wrong tuple structure.
- **Fix**:
  - Confirm handler returns a `(headers, data, contest, metadata)` tuple.
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

### ❗ Problem: `[ERROR] pdf2image conversion failed ... Poppler missing`

- **Cause**: Poppler utilities (pdftoppm/pdftocairo) are not installed, so `pdf2image` cannot rasterize pages and the handler falls back to PyMuPDF every run.
- **Fix (Windows local development)**:
  1. Download the latest Poppler zip from [https://github.com/oschwartz10612/poppler-windows/releases](https://github.com/oschwartz10612/poppler-windows/releases) (or poppler.org) and unzip it, e.g., to `C:\poppler`.
  2. Set `POPPLER_PATH` (or `CONFIG_POPPLER_PATH` in `webapp/config.py`) to the extracted `bin` folder, for example:

     ```powershell
     setx POPPLER_PATH "C:\\poppler\\Library\\bin"
     ```

     Restart the parser process so the cached disable flag clears.
- **Fix (Linux / Azure)**:
  - Install Poppler utilities during provisioning:

    ```bash
    sudo apt-get update
    sudo apt-get install -y poppler-utils
    ```

  - Ensure the command above runs in your deployment script or container build. Once `pdftoppm` is on PATH, the handler automatically re-enables `pdf2image`.
- **Verification**:
  - Re-run the problematic PDF (e.g., the Minnesota 2016 sample) and confirm the logs show `pdf2image` succeeded or that the Poppler warning no longer appears.

### ❗ Problem: Eventlet monkey patching interferes with threads or teardown

- **Symptoms**: Background threads fail with `'NoneType' object is not callable'` during shutdown, or sessions bleed across users when running the web UI under heavy load.
- **Cause**: Aggressive eventlet monkey patching replaces native threading primitives; during tests or in environments that rely on OS threads this can lead to cleanup failures.
- **Fix**:
  - To disable patching (e.g., in unit tests), set `SMART_ELECTIONS_SKIP_EVENTLET_PATCH=1` or `SMART_ELECTIONS_FORCE_THREADING=1` before importing the web app.
  - To keep eventlet but avoid patching the threading module, leave the defaults in place or explicitly set `SMART_ELECTIONS_EVENTLET_PATCH_THREAD=0`.
  - If you need full eventlet patching, set `SMART_ELECTIONS_EVENTLET_PATCH_THREAD=1` and restart the process.
- **Diagnostics**: The web app logs the chosen async mode and patch configuration on startup. You can also inspect `webapp.Smart_Elections_Parser_Webapp.EVENTLET_STATUS` in a Python shell to verify the current settings.

### ❗ Problem: PyMuPDF emits `swigvarlink` DeprecationWarning on import

- **Root Cause**: Python 3.12 tightened validation around C-extension types that lack a `__module__` attribute. PyMuPDF wheels prior to the upstream fix still expose several SWIG-generated builtin types (`SwigPyObject`, `SwigPyPacked`, `swigvarlink`) without that metadata, triggering a warning the first time the module loads.
- **Fix options**:
  1. **Upgrade to the latest PyMuPDF release**. We pin the minimum supported version in `requirements.txt`. Run `pip install --upgrade PyMuPDF` to pick up the newest wheel once the maintainers publish the patch (the warning disappears in those builds).
  2. **Locally rebuild PyMuPDF with the patched SWIG bindings**. Clone [https://github.com/pymupdf/PyMuPDF](https://github.com/pymupdf/PyMuPDF), apply the commit that sets `__module__` on the SWIG typemaps (see PR #2945 upstream), run `python -m build`, then install the wheel with `pip install dist/PyMuPDF-*.whl`. Building requires the MuPDF toolchain—follow PyMuPDF’s `install.md` for Windows specifics.
- **Project behaviour**: Our loader records the warning once and surfaces the affected type names in handler metadata so you can confirm whether you are running a patched build.

### ❗ Problem: Manual file parsing not working

- **Possible Cause**: Wrong file extension or missing handler.
- **Fix**:
  - Place the file in `input/` with the correct extension.
  - Ensure a handler for the format exists and is registered in `format_router.py`.
  - Use the interactive prompt to select the file.

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

# docs/troubleshooting.md

# Troubleshooting Guide

This guide offers solutions to common problems encountered while using or developing the **Smart Elections Parser** project.

---

## 🔍 General Issues

### ❗ Problem: Parser exits without processing any data
- **Possible Cause**: Missing or malformed URL in `urls.txt`.
- **Fix**: 
  - Check that your `urls.txt` file contains a valid, full URL.
  - Use the interactive prompt to ensure URLs are selected.

### ❗ Problem: No handler found for the URL
- **Possible Cause**: `state_router.py` failed to match the state or county.
- **Fix**:
  - Ensure the domain or state name in the URL matches what's listed in `state_router.py`.
  - Add a fallback to `format_router.py`.

---

## 🕵️ CAPTCHA & Browser Behavior

### ❗ Problem: CAPTCHA triggered but no browser appears
- **Possible Cause**: Browser is running in headless mode.
- **Fix**:
  - Set `HEADLESS=False` in your `.env` file.
  - Set `SHOW_BROWSER_ON_CAPTCHA=True` in `.env`.
  - Verify you’ve installed the proper Playwright browser binaries:
    ```bash
    playwright install
    ```

### ❗ Problem: CAPTCHA page stuck or browser keeps refreshing
- **Fix**:
  - Manually refresh the page once.
  - Ensure JavaScript and cookies are enabled.
  - Consider switching User-Agent (rotate via `.env`).

---

## 🧾 Output Issues

### ❗ Problem: Output file not written
- **Possible Cause**: No data returned from handler.
- **Fix**:
  - Confirm handler returns a `(title, headers, rows, metadata)` tuple.
  - Ensure `metadata` includes at least `state` and `race` to build path.

### ❗ Problem: CSV headers mismatch
- **Fix**:
  - Use utilities like `utils.table_utils.normalize_headers()` to ensure consistent naming.
  - Validate all candidate-method combinations are included.

---

## 📁 File Handling Issues

### ❗ Problem: PDF/CSV/JSON file not found
- **Possible Cause**: Dynamic downloads failed or input folder not scanned.
- **Fix**:
  - Check that the file exists in `input/`.
  - Confirm download logic in `download_utils.py` is functioning.
  - Use `ENABLE_DOWNLOAD_DISCOVERY=True` in `.env` to allow automatic retrieval.

---

## 🧪 Debugging Tips

- Use `DEBUG_MODE=True` in `.env` to enable verbose logging.
- Use `print_dom_structure()` utility in `html_scanner.py` for debugging site layout.
- Log User-Agent string to verify spoofing effectiveness.
- Manually test URLs in a normal browser before scripting.

---

## 📫 Still stuck?

- Open a GitHub issue with your traceback and `urls.txt` sample.
- Include screenshots if possible.
- Reference `docs/architecture.md` to verify the data flow.

Happy parsing! 🗳️

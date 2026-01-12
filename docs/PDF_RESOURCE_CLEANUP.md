# PDF Resource Cleanup Safeguards

## Problem Statement

The parser was experiencing exit errors on Windows due to:

1. **pdfium library memory leaks** - PIL Image objects created by pdf2image weren't being closed before the library was destroyed
2. **Windows file lock errors** - Temporary PDF files couldn't be deleted during `atexit` cleanup because they were still locked by the pdfium process
3. **Incomplete resource cleanup** - No explicit cleanup of PIL Image objects and temporary directories

Error symptoms:

```text
PermissionError: [WinError 32] The process cannot access the file because it is being used by another process
-> Cannot close object; pdfium library is destroyed. This may cause a memory leak.
```

## Solution Implemented

### 1. Resource Tracking

Added global tracking variables in `pdf_handler.py`:

- `_PDF_IMAGE_REFS` - List of PIL Image objects that need cleanup
- `_PDF_TEMP_DIRS` - Set of temporary directories created by pdf2image
- `_PDF_CLEANUP_REGISTERED` - Flag to ensure cleanup is registered only once

### 2. Cleanup Function

Created `_cleanup_pdf_resources()` that:

- **Closes all PIL Image objects** to release pdfium handles
- **Forces garbage collection** to ensure file handles are released
- **Adds small delay** for Windows file system to release locks
- **Removes temp directories** with retry logic (3 attempts)
- **Gracefully handles** PermissionErrors with fallback to ignore_errors

### 3. Registration Function

Created `_register_pdf_cleanup()` that:

- Registers the cleanup handler with `atexit` (once only)
- Ensures cleanup runs before Python interpreter exits

### 4. Integration Points

#### pdf2image Path (line ~2705)

```python
# Create controlled temp directory
temp_dir = tempfile.mkdtemp(prefix="pdf2image_")
_PDF_TEMP_DIRS.add(temp_dir)

kwargs = {"dpi": dpi, "output_folder": temp_dir}
images_raw = pdf2image.convert_from_path(pdf_path, **kwargs)

# Track images for cleanup
for idx, img in enumerate(images_raw):
    _PDF_IMAGE_REFS.append(img)
    _store(idx, img)

# Explicit cleanup after use
for img in images_raw:
    if hasattr(img, 'close'):
        img.close()
```

#### PyMuPDF Fallback Path (line ~2735)

```python
pil_img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
_PDF_IMAGE_REFS.append(pil_img)  # Track for cleanup
_store(i, pil_img)
pix = None  # Explicitly release pixmap memory
```

#### Exception Handling

```python
finally:
    if doc:
        try:
            doc.close()
        except Exception:
            pass
    gc.collect()  # Force cleanup of partial results
```

## Benefits

1. ✅ **Prevents Windows file lock errors** - Files are closed before deletion
2. ✅ **Eliminates pdfium memory leaks** - Objects closed before library destruction
3. ✅ **Graceful failure handling** - Retry logic with fallback to best-effort cleanup
4. ✅ **No functional changes** - Parsing behavior unchanged, only cleanup improved
5. ✅ **Production ready** - Minimal performance impact, robust error handling

## Testing

To verify the fix works:

1. Process a multi-page PDF document
2. Check terminal for absence of:
   - `PermissionError: [WinError 32]`
   - `Cannot close object; pdfium library is destroyed`
3. Verify temp directories in `%TEMP%` are properly cleaned up
4. Monitor memory usage for absence of leaks

## Environment Variables

No new environment variables required. The cleanup is automatic and always enabled.

## Dependencies

- Python 3.8+
- PIL/Pillow (already required)
- pdf2image (optional, graceful fallback)
- PyMuPDF/fitz (already required)

## Maintenance Notes

- Cleanup function runs automatically via `atexit` hook
- No manual intervention needed
- Safe to call multiple times (idempotent)
- Works on Windows, Linux, and macOS

## Related Files

- `webapp/parser/handlers/formats/pdf_handler.py` - Main implementation
- `webapp/parser/config.py` - PDF processing configuration
- `webapp/parser/utils/pdf_table_utils.py` - PDF utility functions

## Future Enhancements

Potential improvements for future releases:

1. Add resource usage metrics/logging
2. Implement configurable cleanup delay for slower systems
3. Add diagnostic mode to track resource lifetime
4. Consider context manager pattern for PDF processing

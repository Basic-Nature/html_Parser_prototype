# Flask Routes Security Patch
# Apply these changes to webapp/Smart_Elections_Parser_Webapp.py

## Add imports at top (after existing imports from shared_logic)
```python
from webapp.parser.utils.shared_logic import (
    safe_get,
    safe_is_set,
    safe_lower,
    safe_rsplit,
    safe_sid,
    safe_split,
    safe_strip,
    # NEW: Add these path security imports
    safe_filename,
    safe_resolve_path,
    safe_join_path,
    is_path_safe,
)
```

## Add security validation functions (before route definitions)
```python
# Define allowed root directories for file operations (SECURITY)
ALLOWED_FS_ROOTS = {
    'input': INPUT_DIR,
    'output': OUTPUT_DIR,
    'uploads': UPLOADS_DIR,
}

def validate_fs_root(root: str) -> tuple[bool, str]:
    """
    Validate that a root parameter is an allowed file system root.
    
    Args:
        root: Root identifier (input, output, uploads)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not root:
        return False, "Root parameter is required"
    
    root_lower = safe_lower(root).strip()
    if root_lower not in ALLOWED_FS_ROOTS:
        logger.warning({
            "level": "WARNING",
            "type": "security",
            "message": f"[SECURITY] Invalid root parameter: {root}",
        })
        return False, f"Invalid root: {root}"
    
    return True, ""

def validate_fs_path(root: str, subpath: str = "", name: str = "") -> tuple[bool, str, str]:
    """
    Validate file system path parameters and return safe resolved path.
    
    Args:
        root: Root directory identifier
        subpath: Subdirectory path
        name: Filename
        
    Returns:
        Tuple of (is_valid, error_message, safe_path)
    """
    # Validate root
    is_valid, error = validate_fs_root(root)
    if not is_valid:
        return False, error, ""
    
    base_dir = ALLOWED_FS_ROOTS[safe_lower(root).strip()]
    
    try:
        # Sanitize path components
        safe_components = []
        
        if subpath:
            # Split subpath and sanitize each component
            subpath_clean = subpath.replace('\\', '/').strip('/')
            if subpath_clean:
                for component in subpath_clean.split('/'):
                    if component and component != '.':
                        safe_comp = safe_filename(component, strict_mode=True)
                        safe_components.append(safe_comp)
        
        if name:
            safe_name = safe_filename(name, strict_mode=True)
            safe_components.append(safe_name)
        
        # Construct and validate path
        if safe_components:
            resolved_path = safe_join_path(base_dir, *safe_components)
        else:
            resolved_path = safe_resolve_path(base_dir, base_dir)
        
        # Final security check
        if not is_path_safe(resolved_path, [base_dir]):
            logger.error({
                "level": "ERROR",
                "type": "security",
                "message": f"[SECURITY] Path traversal blocked: {subpath}/{name}",
                "root": root,
            })
            return False, "Path traversal detected", ""
        
        return True, "", str(resolved_path)
        
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "security",
            "message": f"[SECURITY] Path validation failed: {e}",
            "root": root,
            "subpath": subpath,
            "name": name,
        })
        return False, f"Invalid path: {e}", ""
```

## Replace /api/fs/list route (lines ~1087-1119)
```python
@app.route("/api/fs/list", methods=["GET"])
def api_fs_list():
    """List directory contents with strict path validation."""
    root = (request.args.get("root") or "").strip()
    subpath = (request.args.get("path") or "").strip().replace("\\", "/")
    
    # Validate parameters
    is_valid, error, safe_path = validate_fs_path(root, subpath)
    if not is_valid:
        return jsonify({"root": root, "path": subpath, "entries": [], "error": error}), 400
    
    # Check path exists and is directory
    if not os.path.isdir(safe_path):
        return jsonify({"root": root, "path": subpath, "entries": []})

    entries = []
    try:
        with os.scandir(safe_path) as it:
            for de in it:
                try:
                    st = de.stat(follow_symlinks=False)
                    entries.append({
                        "name": de.name,
                        "type": "dir" if de.is_dir(follow_symlinks=False) else "file",
                        "size": None if de.is_dir(follow_symlinks=False) else int(st.st_size),
                        "modified": int(st.st_mtime * 1000)
                    })
                except Exception:
                    entries.append({
                        "name": de.name,
                        "type": "dir" if de.is_dir(follow_symlinks=False) else "file",
                        "size": None,
                        "modified": None
                    })
        entries.sort(key=lambda e: (e["type"] != "dir", e["name"].lower()))
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "browser",
            "message": f"Failed to list dir {root}:{subpath} -> {e}",
            "session_id": None
        })
        entries = []
    return jsonify({"root": root, "path": subpath, "entries": entries})
```

## Replace /api/fs/mkdir route (lines ~1124-1146)
```python
@app.route("/api/fs/mkdir", methods=["POST"])
def api_fs_mkdir():
    """Create directory with strict path validation."""
    data = request.get_json(force=True) or {}
    root = (data.get("root") or "").strip()
    subpath = (data.get("path") or "").strip().replace("\\", "/")
    name = (data.get("name") or "").strip()
    
    # Validate name doesn't contain path separators
    if not name or "/" in name or "\\" in name or name in {'.', '..'}:
        return jsonify({"success": False, "error": "Invalid folder name."}), 400
    
    # Validate path
    is_valid, error, parent_path = validate_fs_path(root, subpath)
    if not is_valid:
        return jsonify({"success": False, "error": error}), 400
    
    # Validate target path
    is_valid, error, target_path = validate_fs_path(root, subpath, name)
    if not is_valid:
        return jsonify({"success": False, "error": error}), 400
    
    try:
        os.makedirs(target_path, exist_ok=False)
        logger.info({
            "level": "INFO",
            "type": "browser",
            "message": f"Created directory: {target_path}",
            "session_id": None
        })
        return jsonify({"success": True})
    except FileExistsError:
        return jsonify({"success": False, "error": "Folder already exists."}), 409
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "browser",
            "message": f"mkdir failed: {e}",
            "session_id": None
        })
        return jsonify({"success": False, "error": str(e)}), 500
```

## Replace /api/fs/delete route (lines ~1148-1176)
```python
@app.route("/api/fs/delete", methods=["POST"])
def api_fs_delete():
    """Delete file or directory with strict path validation."""
    data = request.get_json(force=True) or {}
    root = (data.get("root") or "").strip()
    subpath = (data.get("path") or "").strip().replace("\\", "/")
    name = (data.get("name") or "").strip()
    recursive = bool(data.get("recursive"))
    
    # Validate name
    if not name or name in {'.', '..'}:
        return jsonify({"success": False, "error": "Invalid name."}), 400
    
    # Validate target path
    is_valid, error, target_path = validate_fs_path(root, subpath, name)
    if not is_valid:
        return jsonify({"success": False, "error": error}), 400
    
    if not os.path.exists(target_path):
        return jsonify({"success": False, "error": "Not found."}), 404
    
    try:
        if os.path.isfile(target_path):
            os.remove(target_path)
            logger.info({
                "level": "INFO",
                "type": "browser",
                "message": f"Deleted file: {target_path}",
                "session_id": session_id
            })
        elif os.path.isdir(target_path):
            if recursive:
                shutil.rmtree(target_path)
                logger.info({
                    "level": "INFO",
                    "type": "browser",
                    "message": f"Deleted directory (recursive): {target_path}",
                    "session_id": None
                })
            else:
                os.rmdir(target_path)  # only if empty
                logger.info({
                    "level": "INFO",
                    "type": "browser",
                    "message": f"Deleted empty directory: {target_path}",
                    "session_id": None
                })
        else:
            return jsonify({"success": False, "error": "Unsupported type."}), 400
        return jsonify({"success": True})
    except OSError as e:
        logger.error({
            "level": "ERROR",
            "type": "browser",
            "message": f"Delete failed: {e}",
            "session_id": None
        })
        return jsonify({"success": False, "error": str(e)}), 500
```

## Replace /download_fs route (lines ~1178-1194)
```python
@app.route("/download_fs")
def download_fs():
    """Download file with strict path validation."""
    root = (request.args.get("root") or "").strip()
    subpath = (request.args.get("path") or "").strip().replace("\\", "/")
    name = request.args.get("name") or ""
    
    # Validate parameters
    if not name:
        raise NotFound()
    
    is_valid, error, target_path = validate_fs_path(root, subpath, name)
    if not is_valid:
        logger.warning({
            "level": "WARNING",
            "type": "security",
            "message": f"[SECURITY] Download blocked: {error}",
            "root": root,
            "subpath": subpath,
            "name": name,
        })
        raise NotFound()
    
    if not os.path.isfile(target_path):
        raise NotFound()
    
    return send_file(target_path, as_attachment=True)
```

## Security Benefits
1. **Path Traversal Prevention**: All paths validated before use
2. **Strict Sanitization**: Filenames sanitized with strict_mode=True
3. **Whitelist Approach**: Only allowed roots accepted
4. **Security Logging**: All violations logged
5. **Defense in Depth**: Multiple validation layers
6. **Fail Secure**: Errors result in access denial

## Testing Checklist
- [ ] Normal file listing works
- [ ] Directory creation works
- [ ] File deletion works
- [ ] Download works
- [ ] Path traversal attempts blocked (../../etc/passwd)
- [ ] Null byte injection blocked
- [ ] Invalid root rejected
- [ ] Path escape attempts logged
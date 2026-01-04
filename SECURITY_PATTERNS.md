# Security Patterns & Best Practices Guide

## Overview

This document establishes the security patterns and best practices for the Smart Elections Parser project, specifically focusing on path traversal prevention and file operation security.

## Table of Contents

1. [Security Principles](#security-principles)
2. [Path Security Patterns](#path-security-patterns)
3. [Module-Specific Implementations](#module-specific-implementations)
4. [Testing Requirements](#testing-requirements)
5. [Code Review Checklist](#code-review-checklist)

---

## Security Principles

### Defense in Depth

All file operations MUST implement multiple layers of security:

1. **Input Sanitization**: Clean user-provided paths
2. **Path Validation**: Verify paths are within allowed boundaries
3. **Resolution Checking**: Resolve symlinks and relative paths before validation
4. **Boundary Enforcement**: Maintain strict ALLOWED_ROOTS boundaries

### Zero Trust

- **Never trust user input** - All paths from user input, URLs, or external sources must be validated
- **Validate before use** - Check paths BEFORE any file system operation
- **Fail securely** - Raise `ValueError` on suspicious paths rather than allowing operation

---

## Path Security Patterns

### 1. Define Security Boundaries

Every module that performs file operations MUST define `ALLOWED_ROOTS`:

```python
from pathlib import Path

# Define allowed root directories
LOG_DIR_PATH = Path(LOG_DIR).resolve()
CONTEXT_LIBRARY_DIR = Path(CONTEXT_LIBRARY_PATH).parent.resolve()
PROJECT_ROOT_PATH = Path(PROJECT_ROOT).resolve()

ALLOWED_ROOTS = [LOG_DIR_PATH, CONTEXT_LIBRARY_DIR, PROJECT_ROOT_PATH]
```

**Rules:**

- Use `Path().resolve()` to get absolute paths
- Include only directories your module legitimately needs to access
- Document why each root is needed

### 2. Path Validation Function

Every module MUST implement or import `safe_path()`:

```python
def safe_path(path, allowed_roots=None):
    """
    Validate that a path is within allowed directories.
    
    Args:
        path: Path to validate
        allowed_roots: List of allowed root directories (defaults to ALLOWED_ROOTS)
    
    Returns:
        Resolved Path object if valid
    
    Raises:
        ValueError: If path is outside allowed directories
    """
    if allowed_roots is None:
        allowed_roots = ALLOWED_ROOTS
    
    path = Path(path).resolve()
    
    for root in allowed_roots:
        root = Path(root).resolve()
        try:
            path.relative_to(root)
            return path
        except ValueError:
            continue
    
    raise ValueError(
        f"Path traversal detected: {path} is not within "
        f"allowed directories {allowed_roots}"
    )
```

**Rules:**

- ALWAYS call before file operations
- ALWAYS resolve() both path and roots
- ALWAYS raise ValueError on failure
- NEVER silently return a "safe" alternative

### 3. File Operation Patterns

#### Reading Files

```python
def load_data(file_path):
    """Load data from file with path validation."""
    # SECURITY: Validate path first
    validated_path = safe_path(file_path, ALLOWED_ROOTS)
    
    if not validated_path.exists():
        raise FileNotFoundError(f"File not found: {validated_path}")
    
    with open(validated_path, 'r') as f:
        return f.read()
```

#### Writing Files

```python
def save_data(file_path, data):
    """Save data to file with path validation."""
    # SECURITY: Validate path first
    validated_path = safe_path(file_path, ALLOWED_ROOTS)
    
    # Ensure parent directory exists
    validated_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(validated_path, 'w') as f:
        f.write(data)
```

#### Atomic Operations

```python
def atomic_write(file_path, data):
    """Write file atomically with security."""
    # SECURITY: Validate all paths
    validated_path = safe_path(file_path, ALLOWED_ROOTS)
    
    tmp_path = validated_path.with_suffix(validated_path.suffix + ".tmp")
    backup_path = validated_path.with_suffix(validated_path.suffix + ".bak")
    
    # SECURITY: Validate derived paths
    tmp_path = safe_path(tmp_path, ALLOWED_ROOTS)
    backup_path = safe_path(backup_path, ALLOWED_ROOTS)
    
    # Write to temp file
    with open(tmp_path, 'w') as f:
        f.write(data)
    
    # Backup existing
    if validated_path.exists():
        shutil.copy2(validated_path, backup_path)
    
    # Atomic move
    shutil.move(tmp_path, validated_path)
```

#### Deleting Files

```python
def delete_file(file_path):
    """Delete file with path validation."""
    # SECURITY: Validate path first
    validated_path = safe_path(file_path, ALLOWED_ROOTS)
    
    if validated_path.exists():
        validated_path.unlink()
```

#### Subprocess Execution

```python
def run_script(script_path, *args):
    """Run script with path validation."""
    # SECURITY: Validate script path
    validated_script = safe_path(script_path, ALLOWED_ROOTS)
    
    if not validated_script.exists():
        raise FileNotFoundError(f"Script not found: {validated_script}")
    
    # SECURITY: Validate working directory
    validated_cwd = safe_path(PROJECT_ROOT, ALLOWED_ROOTS)
    
    subprocess.run(
        [sys.executable, str(validated_script), *args],
        cwd=str(validated_cwd),
        check=True
    )
```

### 4. Path Construction Patterns

#### User Input Sanitization

```python
from webapp.parser.utils.shared_logic import safe_filename

def build_output_path(user_contest, user_state, user_county):
    """Build output path from user input safely."""
    # SECURITY: Sanitize all user inputs
    safe_contest = safe_filename(user_contest, strict_mode=True)
    safe_state = safe_filename(user_state, strict_mode=True)
    safe_county = safe_filename(user_county, strict_mode=True)
    
    # Build path
    output_path = OUTPUT_DIR / safe_state / safe_county / safe_contest
    
    # SECURITY: Validate final path
    output_path = safe_path(output_path, [OUTPUT_DIR])
    
    return output_path
```

#### Path Joining

```python
def join_paths_safely(base, *components):
    """Join path components safely."""
    # SECURITY: Validate base
    base = safe_path(base, ALLOWED_ROOTS)
    
    # Sanitize components
    safe_components = [
        safe_filename(comp, strict_mode=True) 
        for comp in components
    ]
    
    # Build path
    result = base / Path(*safe_components)
    
    # SECURITY: Validate result
    result = safe_path(result, ALLOWED_ROOTS)
    
    return result
```

---

## Module-Specific Implementations

### manual_correction_bot.py

**Security Boundary:**

```python
ALLOWED_ROOTS = [LOG_DIR, CONTEXT_LIBRARY_DIR, CACHE_DIR, PROJECT_ROOT]
```

**Critical Functions:**

- `load_jsonl()` - Validates before reading
- `save_jsonl()` - Validates main and temp paths
- `atomic_write_json()` - Validates all paths (main, backup, temp)
- `find_log_files()` - Validates directories and returned files
- `check_and_fix_json_files()` - Validates all directories and quarantine paths
- `export_correction_session()` - Validates source and destination
- `import_correction_session()` - Validates import and destination

**Testing:**

See `webapp/tests/test_manual_correction_security.py`

### librarian.py

**Security Boundary:**

```python
ALLOWED_ROOTS = [LOG_DIR_PATH, CONTEXT_LIBRARY_DIR, PROJECT_ROOT_PATH, BASE_DIR_PATH]
```

**Critical Functions:**

- `get_safe_log_path()` - Sanitizes filename and validates path
- `atomic_write_json()` - Validates all derived paths
- `load_context_library()` - Validates library path and backup paths
- `save_context_library()` - Validates save path and temp files
- `backup_context_library()` - Validates all backup paths
- `_get_log_path()` - Sanitizes and validates log paths
- `_deduplicate_jsonl_log()` - Validates log file paths
- `log_unknown_tag()` / `log_unknown_attr()` - Validate log file paths
- `self_heal_context_library()` - Validates script paths before subprocess

**Testing:**

See `webapp/tests/test_librarian_security.py`

### context_coordinator.py

**Security Boundary:**

```python
# Uses librarian.py's ALLOWED_ROOTS via atomic_write_json import
```

**Critical Functions:**

- `_log_jsonl()` - Should validate log_path parameter
- `_log_enrichment_snapshot()` - Constructs paths for logging

**Recommended Patch:**

```python
def _log_jsonl(self, log_path, log_entry):
    """Centralized JSONL logging with path validation."""
    from .librarian import safe_path, ALLOWED_ROOTS
    
    # SECURITY: Validate log_path
    log_path = safe_path(log_path, ALLOWED_ROOTS)
    
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "ab") as f:
        f.write(orjson.dumps(clean_for_json(log_entry)) + b"\n")
```

---

## Testing Requirements

### Unit Tests Required

Every secured module MUST have:

1. **Path Validation Tests**
   - Valid paths within allowed roots
   - Invalid paths outside allowed roots
   - Traversal attempts (`../../../etc/passwd`)
   - Symlink escapes (Unix only)

2. **File Operation Tests**
   - Reading from allowed directories
   - Reading from forbidden directories (should fail)
   - Writing to allowed directories
   - Writing to forbidden directories (should fail)

3. **Edge Case Tests**
   - Empty paths
   - Null bytes
   - URL-encoded traversal
   - Unicode traversal attempts
   - Mixed path separators
   - Windows reserved names

4. **Integration Tests**
   - Realistic attack scenarios
   - Multiple security layers working together
   - Proper error propagation

### Test Template

```python
class TestModuleSecurity:
    """Security tests for module_name"""
    
    def test_safe_path_validates(self, tmp_path):
        """Test that safe_path validates correctly"""
        allowed = tmp_path / "allowed"
        forbidden = tmp_path / "forbidden"
        
        for d in [allowed, forbidden]:
            d.mkdir()
        
        # Should accept allowed
        result = safe_path(allowed / "file.txt", [allowed])
        assert result.is_relative_to(allowed)
        
        # Should reject forbidden
        with pytest.raises(ValueError, match="Path traversal"):
            safe_path(forbidden / "file.txt", [allowed])
    
    def test_file_operation_validates(self, tmp_path):
        """Test that file operations validate paths"""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        
        with patch('module.ALLOWED_ROOTS', [allowed]):
            # Valid operation should succeed
            result = module.some_operation(allowed / "file.txt")
            
            # Invalid operation should fail
            with pytest.raises(ValueError):
                module.some_operation(tmp_path / "forbidden" / "file.txt")
```

---

## Code Review Checklist

### For New Code

- [ ] Does it perform file operations?
- [ ] Are all file paths validated with `safe_path()`?
- [ ] Is `ALLOWED_ROOTS` defined and appropriate?
- [ ] Are user inputs sanitized with `safe_filename()`?
- [ ] Are derived paths (temp, backup) also validated?
- [ ] Are subprocess execution paths validated?
- [ ] Are tests included for security validation?

### For Existing Code

- [ ] Audit all `open()` calls
- [ ] Audit all `os.path` operations
- [ ] Audit all `Path` operations
- [ ] Audit all `shutil` operations
- [ ] Audit all `subprocess` calls
- [ ] Add `safe_path()` before operations
- [ ] Add security tests

### Red Flags

?? **STOP and fix immediately:**

- Any `open(user_input)` without validation
- Any `os.path.join(base, user_input)` without validation
- Any `subprocess.run()` with user-controlled paths
- Any path construction from URL parameters
- Any file operations without `safe_path()` call
- Any `os.remove()` or `shutil.rmtree()` without validation

---

## Examples of Vulnerable Code

### ? WRONG - No Validation

```python
def load_file(filename):
    # VULNERABLE: No validation!
    with open(filename, 'r') as f:
        return f.read()
```

### ? CORRECT - With Validation

```python
def load_file(filename):
    # SECURE: Validate first
    validated = safe_path(filename, ALLOWED_ROOTS)
    with open(validated, 'r') as f:
        return f.read()
```

### ? WRONG - Partial Validation

```python
def save_file(base_dir, filename, data):
    # VULNERABLE: Filename could contain ../
    path = os.path.join(base_dir, filename)
    with open(path, 'w') as f:
        f.write(data)
```

### ? CORRECT - Full Validation

```python
def save_file(base_dir, filename, data):
    # SECURE: Sanitize and validate
    safe_name = safe_filename(filename, strict_mode=True)
    path = Path(base_dir) / safe_name
    path = safe_path(path, ALLOWED_ROOTS)
    with open(path, 'w') as f:
        f.write(data)
```

---

## Maintenance

### When Adding New File Operations

1. **Always** use the security patterns above
2. **Always** add security tests
3. **Always** update this document if introducing new patterns
4. **Run** `security_audit.py` before committing

### When Modifying Existing Code

1. **Review** if change affects file operations
2. **Ensure** security patterns are maintained
3. **Update** tests if security behavior changes
4. **Run** full test suite including security tests

---

## Tools

### Security Audit Tool

Run security audit regularly:

```bash
# Scan entire parser directory
python security_audit.py

# Scan specific directory
python security_audit.py --dir webapp/parser/health

# Generate detailed report
python security_audit.py --output security_report.txt
```

### Test Suite

Run security tests:

```bash
# All security tests
pytest webapp/tests/test_*_security.py -v

# Specific module
pytest webapp/tests/test_manual_correction_security.py -v

# With coverage
pytest webapp/tests/test_*_security.py --cov=webapp.parser --cov-report=html
```

---

## References

- [OWASP Path Traversal](https://owasp.org/www-community/attacks/Path_Traversal)
- [CWE-22: Path Traversal](https://cwe.mitre.org/data/definitions/22.html)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)

---

**Last Updated:** 2025-12-31
**Version:** 1.0.0
**Maintained By:** Security Team

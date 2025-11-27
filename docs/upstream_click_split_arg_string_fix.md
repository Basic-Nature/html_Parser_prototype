---
layout: default
---

# Proposed Upstream Fix: Click `split_arg_string` Deprecation

spaCy (>=3.8) and its dependency weasel still import `split_arg_string` from
`click.parser`, which triggers a `DeprecationWarning` in Click 8.2+. The project
currently applies a `sitecustomize.py` shim to monkey-patch the attribute before
import. This section documents a patch that upstream projects can adopt so that
our shim can be removed once new releases are published.

## Recommended Changes

### weasel (`weasel/util/config.py`)

```diff
@@
-from click.parser import split_arg_string
+try:  # Click < 8.2 exposed split_arg_string on click.parser and warned otherwise
+    from click.shell_completion import split_arg_string  # Click >= 8.2
+except ImportError:  # pragma: no cover - backwards compatibility
+    from click.parser import split_arg_string  # type: ignore F401
```

*Prefer the new location first,* falling back to the old module to preserve
compatibility with older Click releases.

### spaCy (`spacy/cli/_util.py`)

```diff
@@
-from click.parser import split_arg_string
+try:  # Click < 8.2 exposed split_arg_string on click.parser and warned otherwise
+    from click.shell_completion import split_arg_string  # Click >= 8.2
+except ImportError:  # pragma: no cover - backwards compatibility
+    from click.parser import split_arg_string  # type: ignore F401
```

spaCy mirrors the same import pattern as weasel, so the identical fix applies.

## Test Surface

Once either project merges the change, we recommend running:

```bash
python -m pytest webapp/tests/test_table_builder_e2e.py \
    webapp/tests/test_table_builder_noise.py \
    webapp/tests/test_table_core_panels.py
```

The table-builder suite is our sentinel for CLI noise: it interacts with the
pipeline that imports spaCy/weasel and fails if the DeprecationWarning resurfaces.

## Removal Plan for the Shim

1. Upgrade to the released spaCy/weasel versions containing the fix (tracked in
   `requirements.txt`).
2. Delete `sitecustomize.py`.
3. Re-run the pytest bundle above to confirm the warnings remain gone.

Documented here so we can link to the exact change when filing upstream PRs or
issues.

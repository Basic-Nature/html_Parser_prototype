---
layout: default
title: TODO System Overview
---

## Development TODO System

⚠️ **This page contains auto-generated documentation**. While this overview is manually maintained, the TODO lists below are automatically generated from your codebase. See [Auto-Generated Files](#auto-generated-files) for details.

## Overview

This development documentation category tracks all outstanding work items, enhancements, and technical debt across the Smart Elections Parser project. TODOs are automatically extracted from the codebase and categorized by priority level.

## How TODOs Work

### Code Markers

The TODO system automatically scans Python and JavaScript files for the following markers:

| Marker | Usage | Example |
| -------- | ------- | --------- |
| `TODO` | General improvements and future work | `# TODO: Refactor event loop` |
| `FIXME` | Known bugs or issues to fix | `# FIXME: Race condition in auth` |
| `HACK` | Temporary workarounds that need cleanup | `# HACK: Suppress type error` |
| `XXX` | Dangerous code requiring attention | `# XXX: SQL injection risk here` |

### Priority Levels

TODOs are automatically categorized based on context keywords:

- **HIGH**: Critical issues, security concerns, blocking work
  - Keywords: `critical`, `security`, `urgent`, `blocking`, `regression`
  - Impact: Affects core functionality or user safety

- **MEDIUM**: Improvements and technical debt
  - Keywords: `improve`, `refactor`, `optimize`, `cleanup`, `tech-debt`
  - Impact: Enhances quality or maintainability

- **LOW**: Nice-to-haves and polish items
  - Keywords: `nice`, `polish`, `cosmetic`, `future`, `research`
  - Impact: Enhances UX or provides context

### Example Markers

```python
# TODO: HIGH - Validate user input before parsing
def parse_ballot_data(ballot):
    # ... implementation
    pass

# FIXME: Medium priority - Race condition on concurrent uploads
def upload_handler():
    # ... implementation
    pass

# HACK: Low - Suppress mypy error, needs proper type annotation
cast(Dict, user_data)

# XXX: CRITICAL - SQL injection vulnerability, sanitize input!
query = f"SELECT * FROM {table_name}"
```

## Auto-Generated Files

The TODO system generates four markdown files automatically on each build:

- **[todos.md](/html_Parser_prototype/development/todos/)** - Complete list of all TODOs
- **[todos_high.md](/html_Parser_prototype/development/todos-high/)** - High priority items only (critical & urgent)
- **[todos_medium.md](/html_Parser_prototype/development/todos-medium/)** - Medium priority items (improvements, refactoring)
- **[todos_low.md](/html_Parser_prototype/development/todos-low/)** - Low priority items (future, nice-to-have)

Additionally, comprehensive project reference documentation is generated:

- **[project_audit.md](/html_Parser_prototype/development/project-audit/)** - Complete module audit with Mermaid diagrams showing all 90+ modules and their dependencies
- **[pipeline_map.md](/html_Parser_prototype/development/pipeline-map/)** - Detailed pipeline connection map with interactive Mermaid visualizations and module contexts

### Generation Script

The script `scripts/generate_todo_index.py` scans all Python and JavaScript source files and:

1. Extracts all TODO/FIXME/HACK/XXX markers with context
2. Parses file location (module, function, line number)
3. Determines priority level from keywords
4. Generates markdown files with statistics and cross-references
5. Outputs files to: `docs/DEVELOPMENT/todos_*.md`

**Configuration**:

```python
TODO_PATTERNS = {
    'high': ['critical', 'security', 'urgent', 'blocking', 'regression'],
    'medium': ['improve', 'refactor', 'optimize', 'cleanup', 'tech-debt'],
    'low': ['nice', 'polish', 'cosmetic', 'future', 'research']
}
```

## Using TODOs in Your Development

### Adding a New TODO

1. Navigate to the relevant source file
2. Add a comment with the marker and brief description:

   ```python
   # TODO: Improve error handling in retry logic
   ```

3. Optionally add priority context:

   ```python
   # TODO: CRITICAL - Validate certificate before auth check
   ```

4. Run the generation script to update documentation:

   ```bash
   python scripts/generate_todo_index.py
   ```

### Completing a TODO

1. Implement the work and test thoroughly
2. Remove the TODO marker from the source code
3. Commit the change with reference to the TODO:

   ```bash
   git commit -m "fix: Address HIGH TODO - validate certificates (#42)"
   ```

4. The next generation automatically removes it from TODO lists

### Finding TODOs by Category

- **Looking for unfinished work?** Check [todos.md](/html_Parser_prototype/development/todos/)
- **Focused on critical items?** See [todos_high.md](/html_Parser_prototype/development/todos-high/)
- **Planning improvements?** Review [todos_medium.md](/html_Parser_prototype/development/todos-medium/)
- **Future research items?** Check [todos_low.md](/html_Parser_prototype/development/todos-low/)

## Integration with CI/CD

The TODO system is integrated into the automated build pipeline via:

```bash
python scripts/generate_todo_index.py [--root webapp] [--root scripts] [--root docs]
```

**Automation targets**:

- Scans: `webapp/`, `scripts/`, `docs/` directories
- Output: `docs/DEVELOPMENT/todos*.md` (4 files)
- Frequency: Runs after each commit to keep documentation in sync
- Git ignore: Auto-generated files not tracked (see `.gitignore`)

## Key Statistics

The TODO files automatically include:

- Total count of items by priority
- File locations and line numbers
- Function/class context
- Keyword prevalence
- Recommendations for priority redistribution

## Guidelines for TODO Contributors

### ✅ DO

- Use clear, actionable descriptions
- Include context or affected component
- Add priority keywords when appropriate
- Remove TODOs immediately when work is complete
- Reference issue numbers when applicable: `TODO: Fix #42 - ...`

### ❌ DON'T

- Leave completed TODOs in code
- Create obvious TODOs that could be refactored away
- Use TODOs as replacement for issue tracking
- Commit with "TODO: test this" type placeholders
- Mix multiple unrelated TODOs in one marker

## Maintenance

### Regular Review

- Weekly: Check [todos_high.md](/html_Parser_prototype/development/todos-high/) for blocking items
- Monthly: Review all TODOs to identify stale/obsolete items
- Quarterly: Assess tech-debt backlog and re-prioritize

### Cleanup

Run the generation script to refresh documentation:

```bash
python scripts/generate_todo_index.py --root webapp --root scripts --root docs
```

### Troubleshooting

**TODOs not appearing in generated files?**

- Ensure marker format is correct: `# TODO: Description` or `# FIXME: Description`
- Check that file is in scanned directories (webapp/, scripts/, docs/)
- Verify file has correct extension (.py, .js, .ts)
- Run script with verbose output: `python scripts/generate_todo_index.py --verbose`

**Generated files showing old TODOs?**

- Delete `docs/DEVELOPMENT/todos*.md` files
- Re-run script to regenerate from scratch
- Verify old files don't exist in other locations

## See Also

- [System Architecture](/html_Parser_prototype/core/architecture/)
- [Development Standards](/html_Parser_prototype/features/guides/)
- [Integrity Monitoring](/html_Parser_prototype/governance/governance/)

---

**Last Generated**: Auto-generated documentation system active since consolidation integration
**Maintained By**: Smart Elections Parser Development Team

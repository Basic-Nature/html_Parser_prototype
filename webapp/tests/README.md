# Smart Elections Parser - Unit Tests

Comprehensive unit test suite for the Smart Elections Parser project.

## Test Structure

```tree
webapp/tests/
├── __init__.py                    # Test package initialization
├── conftest.py                    # Shared pytest fixtures
├── test_shared_logic.py          # Tests for utils/shared_logic.py
├── test_detect.py                # Tests for utils/detect.py
├── test_table_builder.py         # Tests for utils/table_builder.py
├── test_csv_handler.py           # Tests for CSV handler
├── test_context_coordinator.py   # Tests for context coordinator
├── test_session_manager.py       # Tests for session manager
├── test_librarian.py             # Tests for librarian functions
├── test_models.py                # Tests for database models
├── test_batch_processor.py       # Tests for batch processor
```

## Warning Suppression

**Automatic on Localhost:** When running tests on localhost (detected via `POSTGRES_HOST` set to `localhost` or `127.0.0.1`), warnings are automatically suppressed for cleaner test output. This includes:

- DeprecationWarning
- PendingDeprecationWarning
- FutureWarning
- eventlet and socketio warnings

This behavior is configured in [`conftest.py`](conftest.py) and helps reduce noise during local development while keeping production environments fully verbose.

## Running Tests

### Run all tests

```bash
python run_tests.py
```

### Run specific test module

```bash
python run_tests.py --module test_shared_logic
```

### Run with verbose output

```bash
python run_tests.py -v
```

### Run with coverage report

```bash
python run_tests.py --coverage
```

### Run only unit tests

```bash
python run_tests.py --markers unit
```

### Run tests excluding slow tests

```bash
pytest -m "not slow" webapp/tests
```

## Test Fixtures

Shared fixtures are defined in `conftest.py`:

- **`test_db_engine`**: In-memory SQLite database for testing
- **`db_session`**: Transactional database session
- **`temp_output_dir`**: Temporary directory for test outputs
- **`sample_html_content`**: Sample HTML election results
- **`sample_csv_data`**: Sample CSV election data
- **`sample_contest_data`**: Sample contest metadata
- **`mock_coordinator`**: Mock ContextCoordinator
- **`mock_page`**: Mock Playwright page object

## Writing New Tests

### Test File Naming

- Test files must start with `test_`
- Test classes must start with `Test`
- Test functions must start with `test_`

### Example Test Structure

```python
"""Tests for my_module.py"""
import pytest
from webapp.parser.my_module import my_function


class TestMyFunction:
    """Tests for my_function."""
    
    def test_basic_case(self):
        """Test basic functionality."""
        result = my_function("input")
        assert result == "expected"
    
    def test_edge_case(self, mock_coordinator):
        """Test edge case with fixture."""
        result = my_function("edge", coordinator=mock_coordinator)
        assert result is not None
```

## Test Categories

### Unit Tests

Pure unit tests with no external dependencies. Mark with `@pytest.mark.unit`

### Integration Tests

Tests requiring external resources (database, files, network). Mark with `@pytest.mark.integration`

### Slow Tests

Long-running tests. Mark with `@pytest.mark.slow`

## Coverage Reports

After running tests with `--coverage`, view the HTML coverage report:

```bash
# Generate coverage report
python run_tests.py --coverage

# Open HTML report (Windows)
start htmlcov\index.html

# Open HTML report (Linux/Mac)
open htmlcov/index.html
```

## Configuration

Test configuration is in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["webapp/tests"]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests requiring external resources",
    "unit: marks pure unit tests",
]
```

## Debugging Tests

### Run a single test

```bash
pytest webapp/tests/test_shared_logic.py::TestSafeFilename::test_basic_filename -v
```

### Run tests with debugger

```bash
pytest --pdb webapp/tests/test_shared_logic.py
```

### Show print statements

```bash
pytest -s webapp/tests/test_shared_logic.py
```

## Best Practices

1. **Isolation**: Each test should be independent
2. **Fixtures**: Use fixtures for shared setup
3. **Mocking**: Mock external dependencies
4. **Clarity**: Test names should describe what is being tested
5. **Coverage**: Aim for critical path coverage, not 100%
6. **Fast**: Keep unit tests fast; use markers for slow tests

## Dependencies

Required packages for testing:

- pytest
- pytest-cov (for coverage)
- pytest-mock (for mocking)

Install with:

```bash
pip install pytest pytest-cov pytest-mock
```

## ?? Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Python unittest.mock](https://docs.python.org/3/library/unittest.mock.html)

## Experimental scratch tests

The repository-root `tests/` directory is gitignored scratch space for temporary validation, experiments, and agent-generated probes. Files there are not part of the permanent regression suite and should be cleaned up after their intent is either migrated or retired.

Permanent Python regression tests belong in `webapp/tests/`. Development diagnostics or smoke tooling may live under `tools/` or `scripts/` when their behavior is intentionally manual. Never place secrets, credentials, or other sensitive data in scratch tests.

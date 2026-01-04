# Test Suite Setup Summary

## ? Files Created

All test files have been successfully created in the `webapp/tests/` directory:

### Core Test Infrastructure

- ? `webapp/tests/__init__.py` - Test package initialization
- ? `webapp/tests/conftest.py` - Shared pytest fixtures and configuration
- ? `webapp/tests/README.md` - Comprehensive testing documentation

### Test Modules

- ? `webapp/tests/test_shared_logic.py` - Tests for utils/shared_logic.py (127 lines)
- ? `webapp/tests/test_detect.py` - Tests for utils/detect.py (136 lines)
- ? `webapp/tests/test_table_builder.py` - Tests for table building (66 lines)
- ? `webapp/tests/test_csv_handler.py` - Tests for CSV handler (49 lines)
- ? `webapp/tests/test_context_coordinator.py` - Tests for context coordinator (51 lines)
- ? `webapp/tests/test_session_manager.py` - Tests for session manager (52 lines)
- ? `webapp/tests/test_librarian.py` - Tests for librarian functions (32 lines)
- ? `webapp/tests/test_models.py` - Tests for database models (38 lines)
- ? `webapp/tests/test_batch_processor.py` - Tests for batch processor (23 lines)

### Test Utilities

- ? `run_tests.py` - Test runner script with command-line options
- ? `validate_tests.py` - Setup validation script
- ? `pyproject.toml` - Updated with pytest configuration

## ?? Test Coverage

The test suite covers:

1. **Shared Logic** (test_shared_logic.py)
   - Filename sanitization
   - Slug generation
   - Safe accessor functions
   - Location normalization
2. **Detection & Parsing** (test_detect.py)
   - Text and header normalization
   - Location header detection
   - Candidate column detection
   - Data harmonization
   - Numeric parsing
3. **Table Building** (test_table_builder.py)
   - Simple table building
   - Table pivoting
   - Robust table extraction

4. **Format Handlers** (test_csv_handler.py)
   - Basic CSV parsing
   - CSV with contest columns

5. **Context Integration** (test_context_coordinator.py)
   - Coordinator initialization
   - Entity extraction
   - Semantic scoring
   - State/county detection

6. **Session Management** (test_session_manager.py)
   - Session creation
   - State transitions
   - Manual source management
   - Session cleanup

7. **Data Models** (test_models.py)
   - Contest model
   - Party model
   - State-County relationships

8. **Batch Processing** (test_batch_processor.py)
   - Batch processor initialization

## ?? Running Tests

### Quick Start

```bash
# Activate virtual environment (if not already active)
.venv\Scripts\Activate.ps1

# Run all tests
.venv\Scripts\python.exe run_tests.py

# Run with verbose output
.venv\Scripts\python.exe run_tests.py -v

# Run specific test module
.venv\Scripts\python.exe run_tests.py --module test_shared_logic

# Run with coverage report
.venv\Scripts\python.exe run_tests.py --coverage
```

### Direct pytest Commands

```bash
# Run all tests
.venv\Scripts\python.exe -m pytest webapp/tests

# Run specific test file
.venv\Scripts\python.exe -m pytest webapp/tests/test_shared_logic.py

# Run specific test class
.venv\Scripts\python.exe -m pytest webapp/tests/test_shared_logic.py::TestSafeFilename

# Run specific test function
.venv\Scripts\python.exe -m pytest webapp/tests/test_shared_logic.py::TestSafeFilename::test_basic_filename
```

## ?? Helper Scripts

- `scripts/run_tests.sh` / `scripts/run_tests.ps1`: Run ruff, mypy (formats + tests), then pytest. Use `SKIP_RUFF=1` or `SKIP_MYPY=1` to bypass linters; pass pytest args through.
- `scripts/run_pipeline_smoke.sh`: Fast smoke (context/detect/librarian tests) plus `validate_tests.py` and `run_statement_test.py --dry-run` when present.
- `scripts/ci_verify.sh`: CI-friendly gate running ruff, mypy (formats + tests), and pytest with coverage + JUnit output to `./artifacts` (override via `ARTIFACTS_DIR`).
- `scripts/run_webapp.sh` / `scripts/run_webapp.ps1`: Safeguard launcher for `python -m webapp.Smart_Elections_Parser_Webapp`; loads `.env`, verifies `FLASK_SECRET_KEY` and DB env vars, creates runtime dirs, then starts the server. Override env file via `ENV_FILE=...`.

### Quick Start (Unix)

```bash
./scripts/run_tests.sh -q
./scripts/run_pipeline_smoke.sh
./scripts/ci_verify.sh
```

### Quick Start (Windows)

```powershell
./scripts/run_tests.ps1 -q
```

## ?? Next Steps

1. **Install Test Dependencies** (if not already installed):

   ```bash
   .venv\Scripts\python.exe -m pip install pytest pytest-cov pytest-mock
   ```

2. **Validate Setup**:

   ```bash
   .venv\Scripts\python.exe validate_tests.py
   ```

3. **Run Tests**:

   ```bash
   .venv\Scripts\python.exe run_tests.py
   ```

4. **Add More Tests**: Extend the test suite by adding more test modules following the existing patterns.

## ?? Test Fixtures Available

All test modules can use these fixtures from `conftest.py`:

- `test_db_engine` - In-memory SQLite database
- `db_session` - Transactional database session
- `temp_output_dir` - Temporary directory for test outputs
- `sample_html_content` - Sample HTML election results
- `sample_csv_data` - Sample CSV data
- `sample_contest_data` - Sample contest metadata
- `mock_coordinator` - Mock ContextCoordinator
- `mock_page` - Mock Playwright page object

## ?? Test Markers

Tests can be marked with:

- `@pytest.mark.unit` - Pure unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow-running tests

Run specific markers:

```bash
.venv\Scripts\python.exe -m pytest -m unit webapp/tests
.venv\Scripts\python.exe -m pytest -m "not slow" webapp/tests
```

## ?? Configuration

Test configuration is in `pyproject.toml`:

- Test discovery patterns
- Coverage settings
- Warning filters
- pytest options

## ?? Documentation

See `webapp/tests/README.md` for:

- Detailed usage instructions
- Writing new tests
- Best practices
- Debugging tips
- Coverage reports

## ? Summary

? Complete test infrastructure is set up
? 9 test modules created covering major components
? Comprehensive fixtures for mocking and test data
? Test runner with multiple options
? Coverage reporting configured
? Documentation provided

The test suite is ready to use! Start by running the validation script, then run the tests.

# Quick Start Guide - Running Tests

## Prerequisites Check

1. **Verify Python is available**:
   ```powershell
   .venv\Scripts\python.exe --version
   ```
   Should show Python 3.11 or higher.

2. **Install test dependencies**:
   ```powershell
   .venv\Scripts\python.exe -m pip install pytest pytest-cov pytest-mock
   ```

3. **Validate setup**:
   ```powershell
   .venv\Scripts\python.exe validate_tests.py
   ```

## Running Tests - Common Commands

### Run All Tests
```powershell
.venv\Scripts\python.exe run_tests.py
```

### Run with Verbose Output
```powershell
.venv\Scripts\python.exe run_tests.py -v
```

### Run Specific Test Module
```powershell
# Test shared logic utilities
.venv\Scripts\python.exe run_tests.py --module test_shared_logic

# Test detection functions
.venv\Scripts\python.exe run_tests.py --module test_detect

# Test CSV handler
.venv\Scripts\python.exe run_tests.py --module test_csv_handler
```

### Run with Coverage Report
```powershell
.venv\Scripts\python.exe run_tests.py --coverage
```

Then open the coverage report:
```powershell
start htmlcov\index.html
```

## Alternative: Direct pytest Commands

If you prefer using pytest directly:

```powershell
# Run all tests
.venv\Scripts\python.exe -m pytest webapp/tests -v

# Run specific file
.venv\Scripts\python.exe -m pytest webapp/tests/test_shared_logic.py -v

# Run specific test class
.venv\Scripts\python.exe -m pytest webapp/tests/test_shared_logic.py::TestSafeFilename -v

# Run specific test function
.venv\Scripts\python.exe -m pytest webapp/tests/test_shared_logic.py::TestSafeFilename::test_basic_filename -v
```

## Understanding Test Output

### Success
```
webapp/tests/test_shared_logic.py::TestSafeFilename::test_basic_filename PASSED
```

### Failure
```
webapp/tests/test_shared_logic.py::TestSafeFilename::test_basic_filename FAILED
```
The output will show:
- What was expected
- What was actually received
- The line where the assertion failed

### Example Test Run
```
======================== test session starts ========================
collected 45 items

webapp/tests/test_shared_logic.py ............    [ 26%]
webapp/tests/test_detect.py ..............        [ 57%]
webapp/tests/test_table_builder.py ..             [ 61%]
webapp/tests/test_csv_handler.py ..               [ 66%]
webapp/tests/test_context_coordinator.py ....     [ 75%]
webapp/tests/test_session_manager.py ....         [ 84%]
webapp/tests/test_librarian.py ..                 [ 88%]
webapp/tests/test_models.py ...                   [ 95%]
webapp/tests/test_batch_processor.py .            [100%]

======================== 45 passed in 2.45s =========================
```

## Troubleshooting

### Import Errors
If you see import errors:
```powershell
# Make sure you're in the project root
cd C:\Users\olivi\html_Parser_prototype

# Verify the path
.venv\Scripts\python.exe -c "import sys; print('\n'.join(sys.path))"
```

### Missing Dependencies
```powershell
# Install all test dependencies
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install pytest pytest-cov pytest-mock
```

### Tests Not Found
```powershell
# Verify test files exist
Get-ChildItem -Path webapp\tests -Filter "test_*.py"
```

## Next Steps

1. ? Run validation: `.venv\Scripts\python.exe validate_tests.py`
2. ? Run all tests: `.venv\Scripts\python.exe run_tests.py -v`
3. ? Check coverage: `.venv\Scripts\python.exe run_tests.py --coverage`
4. ? Add more tests as needed

## Adding New Tests

1. Create a new file: `webapp/tests/test_mymodule.py`
2. Use the existing tests as templates
3. Import the module you want to test
4. Write test classes and functions
5. Run your new tests

Example:
```python
"""Tests for my new module"""
import pytest
from webapp.parser.my_module import my_function

class TestMyFunction:
    """Tests for my_function."""
    
    def test_basic_case(self):
        """Test basic functionality."""
        result = my_function("input")
        assert result == "expected"
```

## Getting Help

- See `webapp/tests/README.md` for detailed documentation
- Check existing tests for examples
- pytest documentation: https://docs.pytest.org/

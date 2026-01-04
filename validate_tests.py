#!/usr/bin/env python3
"""
Validate test setup and configuration.
Run this before running tests to ensure everything is configured correctly.
"""
import sys
from pathlib import Path

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def check_file(path: Path, required: bool = True) -> bool:
    """Check if a file exists."""
    if path.exists():
        print(f"{GREEN}?{RESET} Found: {path}")
        return True
    else:
        symbol = f"{RED}?{RESET}" if required else f"{YELLOW}?{RESET}"
        status = "REQUIRED" if required else "OPTIONAL"
        print(f"{symbol} Missing ({status}): {path}")
        return not required

def check_import(module_name: str, package: str = None) -> bool:
    """Check if a Python module can be imported."""
    try:
        if package:
            __import__(package)
        else:
            __import__(module_name)
        print(f"{GREEN}?{RESET} Can import: {module_name}")
        return True
    except ImportError as e:
        print(f"{RED}?{RESET} Cannot import: {module_name} ({e})")
        return False

def main():
    """Validate the test setup."""
    print("\n" + "="*60)
    print("Smart Elections Parser - Test Setup Validation")
    print("="*60 + "\n")
    
    all_ok = True
    
    # Check test directory structure
    print("Checking test directory structure...")
    test_dir = Path("webapp/tests")
    required_files = [
        test_dir / "__init__.py",
        test_dir / "conftest.py",
        test_dir / "README.md",
    ]
    
    for file_path in required_files:
        if not check_file(file_path, required=True):
            all_ok = False
    
    print()
    
    # Check test files
    print("Checking test files...")
    test_files = [
        "test_shared_logic.py",
        "test_detect.py",
        "test_table_builder.py",
        "test_csv_handler.py",
        "test_context_coordinator.py",
        "test_session_manager.py",
        "test_librarian.py",
        "test_models.py",
        "test_batch_processor.py",
    ]
    
    for test_file in test_files:
        if not check_file(test_dir / test_file, required=False):
            pass  # Optional test files
    
    print()
    
    # Check required Python packages
    print("Checking Python packages...")
    required_packages = [
        ("pytest", "pytest"),
        ("pytest-cov", "pytest_cov"),
        ("sqlalchemy", "sqlalchemy"),
    ]
    
    for package_name, import_name in required_packages:
        if not check_import(import_name):
            print(f"  Install with: pip install {package_name}")
            all_ok = False
    
    print()
    
    # Check project structure
    print("Checking project structure...")
    project_files = [
        Path("pyproject.toml"),
        Path("run_tests.py"),
        Path("webapp/parser/config.py"),
        Path("webapp/parser/utils/shared_logic.py"),
    ]
    
    for file_path in project_files:
        if not check_file(file_path, required=True):
            all_ok = False
    
    print()
    
    # Summary
    print("="*60)
    if all_ok:
        print(f"{GREEN}? All checks passed! You're ready to run tests.{RESET}")
        print(f"\nRun tests with: python run_tests.py")
        return 0
    else:
        print(f"{RED}? Some checks failed. Please fix the issues above.{RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

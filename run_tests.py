#!/usr/bin/env python3
"""
Test runner for Smart Elections Parser unit tests.
Usage: python run_tests.py [--module <module_name>] [--verbose]
"""
import argparse
import sys
import pytest


def main():
    parser = argparse.ArgumentParser(description="Run unit tests for Smart Elections Parser")
    parser.add_argument("--module", help="Specific test module to run (e.g., test_shared_logic)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", "-c", action="store_true", help="Generate coverage report")
    parser.add_argument("--markers", "-m", help="Run tests with specific marker (e.g., slow, integration)")
    
    args = parser.parse_args()
    
    pytest_args = ["webapp/tests"]
    
    if args.module:
        pytest_args = [f"webapp/tests/{args.module}.py"]
    
    if args.verbose:
        pytest_args.append("-v")
    
    if args.coverage:
        pytest_args.extend(["--cov=webapp/parser", "--cov-report=html", "--cov-report=term"])
    
    if args.markers:
        pytest_args.extend(["-m", args.markers])
    
    # Add standard options
    pytest_args.extend([
        "--tb=short",  # Shorter traceback format
        "--strict-markers",  # Strict marker validation
        "-ra",  # Show all test outcomes
    ])
    
    exit_code = pytest.main(pytest_args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

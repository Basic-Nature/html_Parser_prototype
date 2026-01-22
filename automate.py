#!/usr/bin/env python3
"""
Central automation script for Smart Elections Parser.

Runs all automated tasks:
- Generates comprehensive pipeline audit map
- Runs health bots and integrity checks
- Performs web asset linting and type checking
- Executes automated tests
- Validates webapp startup

Usage: python automate.py [--skip-web] [--skip-health] [--skip-tests]
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

from webapp.parser.utils.shared_logic import generate_pipeline_map
from webapp.parser.health.health_router import BotPipeline
from webapp.parser.utils.logger_singleton import logger


def run_pipeline_audit():
    """Generate the comprehensive pipeline audit map."""
    print("[AUTOMATE] Generating pipeline audit map...")
    logger.info("[AUTOMATE] Generating pipeline audit map...")
    success = generate_pipeline_map(project_root=str(project_root))
    if success:
        print("[AUTOMATE] Pipeline audit map generated successfully.")
        logger.info("[AUTOMATE] Pipeline audit map generated successfully.")
    else:
        print("[AUTOMATE] Failed to generate pipeline audit map.")
        logger.error("[AUTOMATE] Failed to generate pipeline audit map.")
    return success


def run_health_bots():
    """Run all health bots and integrity checks."""
    print("[AUTOMATE] Running health bots and integrity checks...")
    logger.info("[AUTOMATE] Running health bots and integrity checks...")
    try:
        pipeline = BotPipeline()
        pipeline.run()
        print("[AUTOMATE] Health bots completed successfully.")
        logger.info("[AUTOMATE] Health bots completed successfully.")
        return True
    except Exception as e:
        print(f"[AUTOMATE] Health bots failed: {e}")
        logger.error(f"[AUTOMATE] Health bots failed: {e}")
        return False


def run_web_checks():
    """Run linting and type checking for web assets (JS, CSS, HTML)."""
    print("[AUTOMATE] Running web asset checks (linting, type checking)...")
    logger.info("[AUTOMATE] Running web asset checks (linting, type checking)...")
    try:
        # Run npm verify:all which includes JS lint, TS check, and Python checks
        result = subprocess.run(
            ["npm", "run", "verify:all"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        if result.returncode == 0:
            print("[AUTOMATE] Web checks passed.")
            logger.info("[AUTOMATE] Web checks passed.")
            logger.debug(f"[AUTOMATE] Web check output: {result.stdout}")
        else:
            print(f"[AUTOMATE] Web checks failed with code {result.returncode}")
            logger.error(f"[AUTOMATE] Web checks failed with code {result.returncode}")
            logger.error(f"[AUTOMATE] STDERR: {result.stderr}")
            logger.error(f"[AUTOMATE] STDOUT: {result.stdout}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("[AUTOMATE] Web checks timed out.")
        logger.error("[AUTOMATE] Web checks timed out.")
        return False
    except FileNotFoundError:
        print("[AUTOMATE] npm not found. Install Node.js and npm to run web checks.")
        logger.error("[AUTOMATE] npm not found. Install Node.js and npm to run web checks.")
        return False
    except Exception as e:
        print(f"[AUTOMATE] Web checks failed: {e}")
        logger.error(f"[AUTOMATE] Web checks failed: {e}")
        return False


def run_automated_tests():
    """Run automated tests."""
    logger.info("[AUTOMATE] Running automated tests...")
    try:
        # Run the PDF statement test as an example
        result = subprocess.run(
            [sys.executable, "run_statement_test.py"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            logger.info("[AUTOMATE] Automated tests passed.")
            logger.debug(f"[AUTOMATE] Test output: {result.stdout}")
        else:
            logger.error(f"[AUTOMATE] Tests failed with code {result.returncode}")
            logger.error(f"[AUTOMATE] STDERR: {result.stderr}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.error("[AUTOMATE] Tests timed out.")
        return False
    except Exception as e:
        logger.error(f"[AUTOMATE] Tests failed: {e}")
        return False


def validate_webapp_startup():
    """Quick validation that the webapp can start (doesn't run full server)."""
    logger.info("[AUTOMATE] Validating webapp startup...")
    try:
        # Import the webapp module to check for import errors
        import webapp.Smart_Elections_Parser_Webapp
        logger.info("[AUTOMATE] Webapp import successful.")
        return True
    except Exception as e:
        logger.error(f"[AUTOMATE] Webapp validation failed: {e}")
        return False


def run_self_check():
    """Run the headless CI self-check script and return True on success."""
    logger.info("[AUTOMATE] Running headless self-check (tools/ci_headless_check.py)...")
    try:
        result = subprocess.run(
            [sys.executable, "tools/ci_headless_check.py"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=180
        )
        logger.debug(f"[AUTOMATE] Self-check stdout: {result.stdout}")
        logger.debug(f"[AUTOMATE] Self-check stderr: {result.stderr}")
        if result.returncode == 0:
            logger.info("[AUTOMATE] Self-check passed.")
            return True
        else:
            logger.error(f"[AUTOMATE] Self-check failed with code {result.returncode}")
            print(result.stdout)
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        logger.error("[AUTOMATE] Self-check timed out.")
        return False
    except Exception as e:
        logger.error(f"[AUTOMATE] Self-check failed: {e}")
        return False


def run_ballot_lens_check():
    """Run the Playwright Ballot Lens visibility check (tools/pw_check_ballot_lens.py)."""
    logger.info("[AUTOMATE] Running Ballot Lens headless check (tools/pw_check_ballot_lens.py)...")
    try:
        result = subprocess.run(
            [sys.executable, "tools/pw_check_ballot_lens.py"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=120,
        )
        logger.debug(f"[AUTOMATE] Ballot Lens stdout: {result.stdout}")
        logger.debug(f"[AUTOMATE] Ballot Lens stderr: {result.stderr}")
        if result.returncode == 0:
            logger.info("[AUTOMATE] Ballot Lens check passed.")
            return True
        logger.error(f"[AUTOMATE] Ballot Lens check failed with code {result.returncode}")
        print(result.stdout)
        print(result.stderr)
        return False
    except subprocess.TimeoutExpired:
        logger.error("[AUTOMATE] Ballot Lens check timed out.")
        return False
    except Exception as e:
        logger.error(f"[AUTOMATE] Ballot Lens check failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all automated scripts for Smart Elections Parser.")
    parser.add_argument("--skip-web", action="store_true", help="Skip web asset checks")
    parser.add_argument("--skip-health", action="store_true", help="Skip health bots")
    parser.add_argument("--skip-tests", action="store_true", help="Skip automated tests")
    parser.add_argument("--skip-webapp-check", action="store_true", help="Skip webapp startup validation")
    parser.add_argument("--self-check", action="store_true", help="Run headless self-check (tools/ci_headless_check.py) after other checks")
    parser.add_argument("--ballot-lens-check", action="store_true", help="Run Ballot Lens Playwright visibility check (tools/pw_check_ballot_lens.py)")

    args = parser.parse_args()

    print("[AUTOMATE] Starting comprehensive automation run...")
    logger.info("[AUTOMATE] Starting comprehensive automation run...")

    results = {}

    # Always run pipeline audit
    results["pipeline_audit"] = run_pipeline_audit()

    # Run health bots unless skipped
    if not args.skip_health:
        results["health_bots"] = run_health_bots()
    else:
        print("[AUTOMATE] Skipping health bots.")
        logger.info("[AUTOMATE] Skipping health bots.")
        results["health_bots"] = None

    # Run web checks unless skipped
    if not args.skip_web:
        results["web_checks"] = run_web_checks()
    else:
        print("[AUTOMATE] Skipping web checks.")
        logger.info("[AUTOMATE] Skipping web checks.")
        results["web_checks"] = None

    # Run tests unless skipped
    if not args.skip_tests:
        results["tests"] = run_automated_tests()
    else:
        print("[AUTOMATE] Skipping automated tests.")
        logger.info("[AUTOMATE] Skipping automated tests.")
        results["tests"] = None

    # Optional headless self-check
    if args.self_check:
        results['self_check'] = run_self_check()
    else:
        results['self_check'] = None

    # Optional Ballot Lens check
    if args.ballot_lens_check:
        results['ballot_lens_check'] = run_ballot_lens_check()
    else:
        results['ballot_lens_check'] = None

    # Validate webapp unless skipped
    if not args.skip_webapp_check:
        results["webapp_validation"] = validate_webapp_startup()
    else:
        print("[AUTOMATE] Skipping webapp validation.")
        logger.info("[AUTOMATE] Skipping webapp validation.")
        results["webapp_validation"] = None

    # Summary
    print("[AUTOMATE] Automation run complete. Summary:")
    logger.info("[AUTOMATE] Automation run complete. Summary:")
    for task, success in results.items():
        status = "PASSED" if success else ("SKIPPED" if success is None else "FAILED")
        print(f"  {task:<20}: {status}")
        logger.info(f"  {task:<20}: {status}")

    # Exit with failure if any critical task failed
    critical_failures = [k for k, v in results.items() if v is False and k in ["pipeline_audit", "web_checks"]]
    # Optional self-check failure handling: if --self-check was requested, treat it as critical
    if args.self_check:
        sc = results.get('self_check')
        if sc is False:
            critical_failures.append('self_check')
    # Treat Ballot Lens check as critical only when requested
    if args.ballot_lens_check and results.get('ballot_lens_check') is False:
        critical_failures.append('ballot_lens_check')
    if critical_failures:
        print(f"[AUTOMATE] Critical failures in: {', '.join(critical_failures)}")
        logger.error(f"[AUTOMATE] Critical failures in: {', '.join(critical_failures)}")
        sys.exit(1)
    else:
        print("[AUTOMATE] All critical tasks passed!")
        logger.info("[AUTOMATE] All critical tasks passed!")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
PyCoMSIA Test Runner

This script runs the PyCoMSIA test suite with appropriate configuration.
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_tests(test_type="all", verbose=False, coverage=False):
    """Run PyCoMSIA tests with specified options."""
    
    # Base command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    
    # Add coverage if requested
    if coverage:
        cmd.extend(["--cov=pycomsia", "--cov-report=html", "--cov-report=term"])
    
    # Test selection
    if test_type == "unit":
        cmd.extend(["-m", "not integration"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "regression":
        cmd.extend(["-m", "regression"])
    elif test_type == "fast":
        cmd.extend(["-m", "not slow"])
    # "all" runs everything
    
    # Specify test directory
    cmd.append("pycomsia/tests")
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run PyCoMSIA test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Types:
  all         Run all tests (default)
  unit        Run only unit tests (no integration tests)
  integration Run only integration tests
  regression  Run only regression tests
  fast        Run only fast tests (exclude slow tests)

Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --type unit        # Run only unit tests
  python run_tests.py --type fast -v     # Run fast tests with verbose output
  python run_tests.py --coverage         # Run all tests with coverage report
        """
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=["all", "unit", "integration", "regression", "fast"],
        default="all",
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Run with coverage reporting"
    )
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not Path("pycomsia").exists():
        print("Error: Must be run from the PyCoMSIA root directory")
        sys.exit(1)
    
    # Run tests
    return_code = run_tests(args.type, args.verbose, args.coverage)
    sys.exit(return_code)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test runner for the Legal Document Processing Pipeline

Runs all unit tests and provides coverage reporting.
"""

import unittest
import sys
import os
import subprocess
from pathlib import Path

def discover_and_run_tests():
    """Discover and run all tests in the tests directory."""
    # Add project root to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    # Discover tests
    loader = unittest.TestLoader()
    start_dir = project_root / 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return success/failure
    return result.wasSuccessful()

def run_with_coverage():
    """Run tests with coverage reporting if coverage is available."""
    try:
        import coverage
        cov = coverage.Coverage()
        cov.start()

        success = discover_and_run_tests()

        cov.stop()
        cov.save()

        print("\n" + "="*50)
        print("Coverage Report")
        print("="*50)
        cov.report(show_missing=True)

        return success

    except ImportError:
        print("Coverage not available. Install with: pip install coverage")
        return discover_and_run_tests()

def main():
    """Main entry point."""
    print("Legal Document Processing Pipeline - Test Suite")
    print("="*50)

    # Check if coverage should be used
    use_coverage = '--coverage' in sys.argv or '-c' in sys.argv

    if use_coverage:
        success = run_with_coverage()
    else:
        success = discover_and_run_tests()

    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()
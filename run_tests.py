#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all tests using pytest."""
    print("Running Algorithmic Trading Strategy Tests")
    print("=" * 50)
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("ERROR: pytest is not installed. Please install it with:")
        print("pip install pytest")
        return False
    
    # Run tests
    try:
        # Run pytest with coverage if available
        try:
            import pytest_cov
            cmd = ["python", "-m", "pytest", "tests/", "--cov=src", "--cov-report=term-missing"]
        except ImportError:
            cmd = ["python", "-m", "pytest", "tests/", "-v"]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n" + "=" * 50)
            print("ALL TESTS PASSED! ✅")
            return True
        else:
            print("\n" + "=" * 50)
            print("SOME TESTS FAILED! ❌")
            return False
            
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


def run_specific_test(test_file):
    """Run a specific test file."""
    test_path = Path("tests") / test_file
    if not test_path.exists():
        print(f"Test file not found: {test_path}")
        return False
    
    try:
        cmd = ["python", "-m", "pytest", str(test_path), "-v"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running test: {e}")
        return False


def main():
    """Main function."""
    if len(sys.argv) > 1:
        # Run specific test file
        test_file = sys.argv[1]
        success = run_specific_test(test_file)
    else:
        # Run all tests
        success = run_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

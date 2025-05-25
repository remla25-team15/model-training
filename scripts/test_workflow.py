#!/usr/bin/env python3
"""
Complete ML Testing Workflow Validator

Tests the complete ML Testing assignment workflow including:
1. Running tests with coverage
2. Calculating ML Test Score
3. Updating README badges
4. Validating all metrics are properly generated

This simulates what the GitHub Actions workflow would do.
"""

import json
import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a shell command and return success status."""
    print(f"🔄 {description}...")
    try:
        # Use bash explicitly for commands that need source
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            executable='/bin/bash'
        )
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - EXCEPTION: {e}")
        return False


def validate_file_exists(file_path: Path, description: str) -> bool:
    """Validate that a file exists."""
    if file_path.exists():
        print(f"✅ {description} - EXISTS")
        return True
    else:
        print(f"❌ {description} - MISSING")
        return False


def validate_json_content(file_path: Path, required_keys: list, description: str) -> bool:
    """Validate that a JSON file contains required keys."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            print(f"❌ {description} - MISSING KEYS: {missing_keys}")
            return False
        else:
            print(f"✅ {description} - VALID JSON")
            return True
    except Exception as e:
        print(f"❌ {description} - JSON ERROR: {e}")
        return False


def main():
    """Run complete workflow validation."""
    print("🚀 Starting ML Testing Workflow Validation\n")

    base_dir = Path(__file__).parent.parent
    success_count = 0
    total_checks = 0

    # Step 1: Create test reports directory
    total_checks += 1
    test_reports_dir = base_dir / "test_reports"
    test_reports_dir.mkdir(exist_ok=True)
    print("✅ Test reports directory created")
    success_count += 1

    # Step 2: Run tests with coverage
    total_checks += 1
    cmd = (
        "python -m pytest tests/ --cov=src --cov-report=json:test_reports/coverage.json "
        "--junit-xml=test_reports/junit.xml -v"
    )
    if run_command(cmd, "Running tests with coverage"):
        success_count += 1

    # Step 3: Validate coverage.json
    total_checks += 1
    coverage_file = base_dir / "test_reports" / "coverage.json"
    if validate_file_exists(coverage_file, "Coverage JSON file") and \
       validate_json_content(coverage_file, ["totals"], "Coverage JSON content"):
        success_count += 1

    # Step 4: Validate junit.xml
    total_checks += 1
    junit_file = base_dir / "test_reports" / "junit.xml"
    if validate_file_exists(junit_file, "JUnit XML file"):
        success_count += 1

    # Step 5: Calculate ML Test Score
    total_checks += 1
    cmd = "python scripts/ml_test_score.py test_reports/junit.xml test_reports/ml_test_score.json"
    if run_command(cmd, "Calculating ML Test Score"):
        success_count += 1

    # Step 6: Validate ml_test_score.json
    total_checks += 1
    ml_score_file = base_dir / "test_reports" / "ml_test_score.json"
    required_keys = [
        "overall_score", "metamorphic_score", "total_tests",
        "passed_tests", "category_breakdown"
    ]
    if validate_file_exists(ml_score_file, "ML Test Score JSON file") and \
       validate_json_content(ml_score_file, required_keys, "ML Test Score JSON content"):
        success_count += 1

    # Step 7: Update README badges
    total_checks += 1
    cmd = "python scripts/update_readme.py"
    if run_command(cmd, "Updating README badges"):
        success_count += 1

    # Step 8: Validate README was updated
    total_checks += 1
    readme_file = base_dir / "README.md"
    if validate_file_exists(readme_file, "README.md file"):
        try:
            with open(readme_file, 'r') as f:
                content = f.read()
            if "<!-- AUTOMATED-BADGES -->" in content and "ML%20Test%20Score" in content:
                print("✅ README badges updated - VALID")
                success_count += 1
            else:
                print("❌ README badges updated - BADGES MISSING")
        except Exception as e:
            print(f"❌ README validation - ERROR: {e}")

    # Final summary
    print("\n📊 WORKFLOW VALIDATION RESULTS")
    print(f"✅ Successful checks: {success_count}/{total_checks}")
    print(f"❌ Failed checks: {total_checks - success_count}/{total_checks}")

    if success_count == total_checks:
        print("\n🎉 ALL CHECKS PASSED - ML Testing workflow is working correctly!")
        return 0
    else:
        print(f"\n⚠️ {total_checks - success_count} CHECKS FAILED - Please review the issues above")
        return 1


if __name__ == "__main__":
    sys.exit(main())

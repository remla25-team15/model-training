#!/usr/bin/env python3
"""
ML Test Score Calculator

Analyzes pytest results and calculates an ML Test Score based on:
- Feature and Data Integrity tests
- Model Development tests
- ML Infrastructure tests
- Monitoring tests
- Metamorphic tests

Based on the ML Test Score methodology from Google.
"""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


class MLTestScoreCalculator:
    """Calculate ML Test Score from pytest results."""

    def __init__(self):
        self.test_categories = {
            "data": {
                "weight": 30,
                "description": "Feature and Data Integrity",
                "keywords": ["data", "feature", "schema", "dataset"],
            },
            "model": {
                "weight": 25,
                "description": "Model Development",
                "keywords": ["model", "negative_keywords", "robustness", "slice"],
            },
            "infra": {
                "weight": 20,
                "description": "ML Infrastructure",
                "keywords": ["infra", "serving", "validation", "quality"],
            },
            "monitor": {
                "weight": 15,
                "description": "Monitoring",
                "keywords": [
                    "monitor",
                    "memory",
                    "latency",
                    "throughput",
                    "performance",
                ],
            },
            "metamorphic": {
                "weight": 10,
                "description": "Metamorphic Testing",
                "keywords": ["mutamorphic", "metamorphic", "swap", "invariant"],
            },
        }

    def parse_junit_xml(self, xml_path: Path) -> dict:
        """Parse JUnit XML test results."""
        if not xml_path.exists():
            raise FileNotFoundError(f"JUnit XML file not found: {xml_path}")

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Try to find testsuite element (can be root or child)
        testsuite = root.find(".//testsuite")
        if testsuite is None:
            testsuite = root

        results = {
            "total_tests": int(testsuite.get("tests", 0)),
            "failures": int(testsuite.get("failures", 0)),
            "errors": int(testsuite.get("errors", 0)),
            "skipped": int(testsuite.get("skipped", 0)),
            "test_cases": [],
        }

        # Parse individual test cases
        for testcase in root.findall(".//testcase"):
            test_name = testcase.get("name", "")
            test_class = testcase.get("classname", "")
            test_time = float(testcase.get("time", 0))

            # Determine test status
            status = "passed"
            error_msg = None

            if testcase.find("failure") is not None:
                status = "failed"
                error_msg = testcase.find("failure").text
            elif testcase.find("error") is not None:
                status = "error"
                error_msg = testcase.find("error").text
            elif testcase.find("skipped") is not None:
                status = "skipped"
                error_msg = testcase.find("skipped").text

            results["test_cases"].append(
                {
                    "name": test_name,
                    "class": test_class,
                    "time": test_time,
                    "status": status,
                    "error": error_msg,
                }
            )

        results["passed"] = (
            results["total_tests"] - results["failures"] - results["errors"]
        )

        return results

    def categorize_test(self, test_name: str, test_class: str) -> str:
        """Categorize a test based on its name and class."""
        full_test_id = f"{test_class}.{test_name}".lower()

        # Specific class-based categorization first (more accurate)
        if "test_infra" in test_class:
            return "infra"
        elif "test_mutamorphic" in test_class:
            return "metamorphic"
        elif "test_monitor" in test_class:
            return "monitor"
        elif "test_model" in test_class:
            return "model"
        elif "test_data" in test_class:
            return "data"
        elif "test_get_data" in test_class:
            return "data"
        elif "test_utils" in test_class:
            return "data"  # Utils tests are typically data pipeline related

        # Check each category for keyword matches in test names
        for category, config in self.test_categories.items():
            for keyword in config["keywords"]:
                if keyword in full_test_id:
                    return category

        # Additional specific mappings for better categorization
        if "serving" in test_name or "validation" in test_name:
            return "infra"
        elif (
            "robustness" in test_name or "slice" in test_name or "negative" in test_name
        ):
            return "model"
        elif (
            "memory" in test_name or "latency" in test_name or "throughput" in test_name
        ):
            return "monitor"
        elif (
            "mutamorphic" in test_name
            or "metamorphic" in test_name
            or "swap" in test_name
        ):
            return "metamorphic"
        elif "schema" in test_name or "dataset" in test_name or "feature" in test_name:
            return "data"

        # Default category if no match found
        return "data"  # Most tests are data-related by default

    def calculate_category_scores(self, test_results: dict) -> dict:
        """Calculate scores for each test category."""
        category_stats = {}

        # Initialize category stats
        for category in self.test_categories:
            category_stats[category] = {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "score": 0.0,
            }

        # Categorize and count tests
        for test_case in test_results["test_cases"]:
            category = self.categorize_test(test_case["name"], test_case["class"])
            category_stats[category]["total"] += 1

            if test_case["status"] == "passed":
                category_stats[category]["passed"] += 1
            else:
                category_stats[category]["failed"] += 1

        # Calculate scores for each category (0-100)
        for category, stats in category_stats.items():
            if stats["total"] > 0:
                stats["score"] = (stats["passed"] / stats["total"]) * 100
            else:
                stats["score"] = 0.0  # No tests in this category

        return category_stats

    def calculate_overall_score(self, category_stats: dict) -> float:
        """Calculate weighted overall ML Test Score."""
        total_weighted_score = 0.0
        total_weight = 0.0

        for category, config in self.test_categories.items():
            weight = config["weight"]
            score = category_stats[category]["score"]

            # Only include categories that have tests
            if category_stats[category]["total"] > 0:
                total_weighted_score += weight * score
                total_weight += weight

        if total_weight > 0:
            return total_weighted_score / total_weight
        else:
            return 0.0

    def calculate_metamorphic_score(self, category_stats: dict) -> float:
        """Calculate specific score for metamorphic testing."""
        metamorphic_stats = category_stats.get("metamorphic", {"score": 0.0})
        return metamorphic_stats["score"]

    def generate_recommendations(self, category_stats: dict) -> list:
        """Generate recommendations based on test results."""
        recommendations = []

        for category, stats in category_stats.items():
            if stats["total"] == 0:
                config = self.test_categories[category]
                recommendations.append(
                    f"⚠️ No {config['description']} tests found. "
                    f"Consider adding tests with keywords: {', '.join(config['keywords'])}"
                )
            elif stats["score"] < 70:
                config = self.test_categories[category]
                recommendations.append(
                    f"❌ {config['description']} score is low ({stats['score']:.1f}%). "
                    f"Review and fix failing tests."
                )
            elif stats["score"] < 90:
                config = self.test_categories[category]
                recommendations.append(
                    f"⚠️ {config['description']} score could be improved ({stats['score']:.1f}%). "
                    f"Consider adding more comprehensive tests."
                )

        if not recommendations:
            recommendations.append("✅ All test categories are performing well!")

        return recommendations

    def save_results(
        self,
        output_path: Path,
        test_results: dict,
        category_stats: dict,
        overall_score: float,
        metamorphic_score: float,
        recommendations: list,
    ) -> None:
        """Save ML Test Score results to JSON."""

        results = {
            "overall_score": overall_score,
            "metamorphic_score": metamorphic_score,
            "total_tests": test_results["total_tests"],
            "passed_tests": test_results["passed"],
            "failed_tests": test_results["failures"] + test_results["errors"],
            "skipped_tests": test_results["skipped"],
            "category_breakdown": {},
            "recommendations": recommendations,
            "methodology": "ML Test Score based on Google's methodology",
            "categories": {},
        }

        # Add category details
        for category, config in self.test_categories.items():
            stats = category_stats[category]
            results["category_breakdown"][category] = {
                "name": config["description"],
                "weight": config["weight"],
                "total_tests": stats["total"],
                "passed_tests": stats["passed"],
                "failed_tests": stats["failed"],
                "score": stats["score"],
            }
            results["categories"][category] = stats["score"]

        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"✅ ML Test Score results saved to: {output_path}")


def main():
    """Main function."""
    if len(sys.argv) != 3:
        print("Usage: python ml_test_score.py <junit_xml_path> <output_path>")
        sys.exit(1)

    xml_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    try:
        calculator = MLTestScoreCalculator()

        # Parse test results
        print(f"📊 Parsing test results from: {xml_path}")
        test_results = calculator.parse_junit_xml(xml_path)

        # Calculate category scores
        print("🔍 Categorizing and scoring tests...")
        category_stats = calculator.calculate_category_scores(test_results)

        # Calculate overall scores
        overall_score = calculator.calculate_overall_score(category_stats)
        metamorphic_score = calculator.calculate_metamorphic_score(category_stats)

        # Generate recommendations
        recommendations = calculator.generate_recommendations(category_stats)

        # Save results
        calculator.save_results(
            output_path,
            test_results,
            category_stats,
            overall_score,
            metamorphic_score,
            recommendations,
        )

        # Print summary
        print("\n📈 ML Test Score Results:")
        print(f"Overall Score: {overall_score:.1f}/100")
        print(f"Metamorphic Testing Score: {metamorphic_score:.1f}/100")
        print(f"Total Tests: {test_results['total_tests']}")
        print(f"Passed: {test_results['passed']}")
        print(f"Failed: {test_results['failures'] + test_results['errors']}")
        print(f"Skipped: {test_results['skipped']}")

        print("\n📋 Category Breakdown:")
        for category, config in calculator.test_categories.items():
            stats = category_stats[category]
            print(
                f"  {config['description']}: {stats['score']:.1f}% "
                f"({stats['passed']}/{stats['total']} tests)"
            )

        print("\n💡 Recommendations:")
        for rec in recommendations:
            print(f"  {rec}")

        # Set exit code based on score
        if overall_score < 50:
            print(f"\n❌ ML Test Score too low: {overall_score:.1f}/100")
            sys.exit(1)
        elif overall_score < 75:
            print(f"\n⚠️ ML Test Score could be improved: {overall_score:.1f}/100")
        else:
            print(f"\n✅ Good ML Test Score: {overall_score:.1f}/100")

    except Exception as e:
        print(f"❌ Error calculating ML Test Score: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

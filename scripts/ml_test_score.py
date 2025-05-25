# ...existing code...

from typing import Dict, List


class MLTestScoreCalculator:
    """Calculate ML Test Score from pytest results."""

    # ...existing methods...


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
            output_path, test_results, category_stats,
            overall_score, metamorphic_score, recommendations
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
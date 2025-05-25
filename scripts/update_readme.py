#!/usr/bin/env python3
"""
README Badge Updater

Updates README.md with test coverage, ML test score, and pylint score badges.
"""

import json
import sys
from pathlib import Path
from typing import Dict


class ReadmeUpdater:
    """Updates README.md with automated badges and metrics."""

    def __init__(self, readme_path: Path):
        self.readme_path = readme_path
        self.badge_section_marker = "<!-- AUTOMATED-BADGES -->"
        self.badge_end_marker = "<!-- END-AUTOMATED-BADGES -->"

    def create_badge(self, label: str, message: str, color: str) -> str:
        """Create a shields.io badge."""
        # URL encode the message
        message_encoded = message.replace(" ", "%20").replace("-", "--")
        return f"![{label}](https://img.shields.io/badge/{label}-{message_encoded}-{color})"

    def get_color_for_score(self, score: float, thresholds: Dict[str, float] = None) -> str:
        """Get color based on score thresholds."""
        if thresholds is None:
            thresholds = {'green': 80, 'yellow': 60, 'red': 40}
        if score >= thresholds['green']:
            return 'brightgreen'
        elif score >= thresholds['yellow']:
            return 'yellow'
        elif score >= thresholds['red']:
            return 'orange'
        else:
            return 'red'

    def generate_badges(self, metrics: Dict) -> str:
        """Generate badge section content."""
        badges = []
        # Test Coverage Badge
        if 'coverage' in metrics:
            coverage = metrics['coverage']
            color = self.get_color_for_score(coverage)
            badges.append(self.create_badge("Test%20Coverage", f"{coverage:.1f}%25", color))
        # ML Test Score Badge
        if 'ml_test_score' in metrics:
            ml_score = metrics['ml_test_score']
            color = self.get_color_for_score(ml_score)
            badges.append(self.create_badge("ML%20Test%20Score", f"{ml_score:.1f}/100", color))
        # Metamorphic Testing Badge
        if 'metamorphic_score' in metrics:
            meta_score = metrics['metamorphic_score']
            color = self.get_color_for_score(meta_score)
            badges.append(self.create_badge("Metamorphic%20Tests", f"{meta_score:.1f}%25", color))
        # Pylint Score Badge
        if 'pylint_score' in metrics:
            pylint_score = metrics['pylint_score']
            color = self.get_color_for_score(pylint_score * 10)  # Convert to 0-100 scale
            badges.append(self.create_badge("Pylint%20Score", f"{pylint_score:.2f}/10", color))
        # Code Quality Badge (overall)
        if 'pylint_score' in metrics and 'coverage' in metrics:
            quality_score = (metrics['pylint_score'] * 10 + metrics['coverage']) / 2
            color = self.get_color_for_score(quality_score)
            badges.append(
                self.create_badge(
                    "Code%20Quality",
                    f"{quality_score:.1f}%25",
                    color
                )
            )
        # Test Status Badge
        if 'tests_passed' in metrics and 'total_tests' in metrics:
            passed = metrics['tests_passed']
            total = metrics['total_tests']
            if total > 0:
                color = 'brightgreen' if passed == total else 'red'
                badges.append(self.create_badge("Tests", f"{passed}/{total}%20passed", color))
        badge_content = f"{self.badge_section_marker}\n"
        badge_content += "\n".join(badges)
        badge_content += f"\n{self.badge_end_marker}"
        return badge_content

    def update_readme(self, metrics: Dict) -> bool:
        """Update README.md with new badges."""
        if not self.readme_path.exists():
            print(f"README.md not found at {self.readme_path}")
            return False
        # Read current README
        with open(self.readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Generate new badge section
        new_badges = self.generate_badges(metrics)
        # Check if badge section exists
        if self.badge_section_marker in content:
            # Replace existing badge section
            start_idx = content.find(self.badge_section_marker)
            end_idx = content.find(self.badge_end_marker) + len(self.badge_end_marker)
            if end_idx > start_idx:
                updated_content = content[:start_idx] + new_badges + content[end_idx:]
            else:
                print("Warning: End marker not found, appending badges at the marker")
                updated_content = content.replace(self.badge_section_marker, new_badges)
        else:
            # Add badge section after title
            lines = content.split('\n')
            title_line = -1
            for i, line in enumerate(lines):
                if line.startswith('#') and 'model-training' in line.lower():
                    title_line = i
                    break
            if title_line >= 0:
                lines.insert(title_line + 2, new_badges)
                updated_content = '\n'.join(lines)
            else:
                # Fallback: add at the beginning
                updated_content = new_badges + '\n\n' + content
        # Write updated README
        with open(self.readme_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        return True

    def generate_summary_section(self, metrics: Dict) -> str:
        """Generate a detailed metrics summary section."""
        summary = "\n## 📊 Automated Quality Metrics\n\n"
        if 'coverage' in metrics:
            summary += f"- **Test Coverage**: {metrics['coverage']:.1f}%\n"
        if 'ml_test_score' in metrics:
            summary += f"- **ML Test Score**: {metrics['ml_test_score']:.1f}/100\n"
        if 'metamorphic_score' in metrics:
            summary += f"- **Metamorphic Testing**: {metrics['metamorphic_score']:.1f}%\n"
        if 'pylint_score' in metrics:
            summary += f"- **Code Quality (Pylint)**: {metrics['pylint_score']:.2f}/10\n"
        if 'tests_passed' in metrics and 'total_tests' in metrics:
            summary += f"- **Test Results**: {metrics['tests_passed']}/{metrics['total_tests']} passed\n"
        summary += "\n*Last updated: Automatically by GitHub Actions*\n"
        return summary


def load_metrics() -> Dict:
    """Load metrics from various sources."""
    metrics = {}
    base_dir = Path(__file__).parent.parent
    test_reports_dir = base_dir / "test_reports"
    # Load test coverage
    coverage_file = test_reports_dir / 'coverage.json'
    if coverage_file.exists():
        try:
            with open(coverage_file, 'r') as f:
                coverage_data = json.load(f)
            metrics['coverage'] = coverage_data.get('totals', {}).get('percent_covered', 0)
            print(f"✅ Loaded coverage: {metrics['coverage']:.1f}%")
        except Exception as e:
            print(f"⚠️ Error loading coverage: {e}")
    # Load ML test score
    ml_score_file = test_reports_dir / 'ml_test_score.json'
    if ml_score_file.exists():
        try:
            with open(ml_score_file, 'r') as f:
                ml_data = json.load(f)
            metrics['ml_test_score'] = ml_data.get('overall_score', 0)
            metrics['metamorphic_score'] = ml_data.get('metamorphic_score', 0)
            metrics['tests_passed'] = ml_data.get('passed_tests', 0)
            metrics['total_tests'] = ml_data.get('total_tests', 0)
            print(f"✅ Loaded ML Test Score: {metrics['ml_test_score']:.1f}/100")
        except Exception as e:
            print(f"⚠️ Error loading ML test score: {e}")
    # Load pylint score
    pylint_file = test_reports_dir / 'pylint_score.json'
    if pylint_file.exists():
        try:
            with open(pylint_file, 'r') as f:
                pylint_data = json.load(f)
            metrics['pylint_score'] = pylint_data.get('score', 0)
            print(f"✅ Loaded Pylint Score: {metrics['pylint_score']:.2f}/10")
        except Exception as e:
            print(f"⚠️ Error loading pylint score: {e}")
    return metrics


def main():
    """Main function."""
    readme_path = Path('README.md')
    if len(sys.argv) > 1:
        readme_path = Path(sys.argv[1])
    # Load metrics
    metrics = load_metrics()
    if not metrics:
        print("⚠️ No metrics found to update README")
        return 1
    # Update README
    updater = ReadmeUpdater(readme_path)
    if updater.update_readme(metrics):
        print(f"✅ README updated successfully with {len(metrics)} metrics")
        # Print summary of what was updated
        print("\n📊 Updated metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'score' in key:
                    print(f"  - {key}: {value:.1f}")
                else:
                    print(f"  - {key}: {value:.1f}%")
            else:
                print(f"  - {key}: {value}")
        return 0
    else:
        print("❌ Failed to update README")
        return 1


if __name__ == "__main__":
    sys.exit(main())

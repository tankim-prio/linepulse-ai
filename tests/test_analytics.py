from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from linepulse.analytics.eda import generate_eda_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "linepulse"
    / "modeling"
    / "daily_line_ml.csv"
)


class AnalyticsTests(unittest.TestCase):
    def test_eda_report_contains_required_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            summary = generate_eda_report(INPUT_PATH, output_dir)

            required_files = {
                "dataset_summary.json",
                "missing_values.csv",
                "numeric_summary.csv",
                "categorical_summary.csv",
                "split_summary.csv",
                "target_balance.png",
                "split_volume.png",
                "line_miss_rate.png",
                "target_correlations.png",
            }
            actual_files = {path.name for path in output_dir.iterdir()}

            self.assertEqual(summary["rows"], 1_224)
            self.assertEqual(summary["missing_cells"], 0)
            self.assertTrue(summary["synthetic_only"])
            self.assertTrue(required_files.issubset(actual_files))
            for filename in required_files:
                self.assertGreater((output_dir / filename).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()


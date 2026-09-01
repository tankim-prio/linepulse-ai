from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from linepulse.data.validator import DataValidator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "linepulse"


class DataValidatorTests(unittest.TestCase):
    def test_bundled_dataset_passes(self) -> None:
        report = DataValidator(DATA_DIR).run()

        failures = [
            f"{check.name}: {check.detail}"
            for check in report.checks
            if not check.passed and check.severity == "error"
        ]
        self.assertTrue(report.passed, "\n".join(failures))
        self.assertEqual(report.dataset_count, 21)
        self.assertEqual(report.row_count, 25_277)

    def test_missing_declared_file_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            copied = Path(temp_dir) / "linepulse"
            shutil.copytree(DATA_DIR, copied)
            (copied / "relational" / "factories.csv").unlink()

            report = DataValidator(copied).run()
            check = self._find_check(report, "inventory:factories")

            self.assertFalse(report.passed)
            self.assertFalse(check.passed)
            self.assertIn("Missing declared file", check.detail)

    def test_duplicate_primary_key_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            copied = Path(temp_dir) / "linepulse"
            shutil.copytree(DATA_DIR, copied)
            path = copied / "relational" / "factories.csv"
            frame = pd.read_csv(path, encoding="utf-8-sig")
            frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
            frame.to_csv(path, index=False, encoding="utf-8-sig")

            report = DataValidator(copied).run()
            check = self._find_check(report, "primary_key:factories")

            self.assertFalse(report.passed)
            self.assertFalse(check.passed)
            self.assertIn("duplicate rows=2", check.detail)

    @staticmethod
    def _find_check(report, name):
        return next(check for check in report.checks if check.name == name)


if __name__ == "__main__":
    unittest.main()


from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from linepulse.ingestion import CsvIngestionService
from linepulse.settings import DEFAULT_DATA_DIR


class CsvIngestionTests(unittest.TestCase):

    def setUp(self):
        self.service = CsvIngestionService(DEFAULT_DATA_DIR)

    def test_valid_csv_can_be_staged(self):
        source = DEFAULT_DATA_DIR / "relational" / "factories.csv"

        with tempfile.TemporaryDirectory() as temp:
            result = self.service.validate_and_stage(
                "factories",
                source,
                Path(temp) / "staging",
            )

            self.assertTrue(result.accepted)
            self.assertTrue(Path(result.staged_path).is_file())
            self.assertFalse(result.already_staged)

    def test_repeated_upload_is_idempotent(self):
        source = DEFAULT_DATA_DIR / "relational" / "factories.csv"

        with tempfile.TemporaryDirectory() as temp:
            staging = Path(temp) / "staging"

            first = self.service.validate_and_stage(
                "factories",
                source,
                staging,
            )

            second = self.service.validate_and_stage(
                "factories",
                source,
                staging,
            )

            self.assertEqual(first.staged_path, second.staged_path)
            self.assertTrue(second.already_staged)

    def test_undeclared_dataset_is_rejected(self):
        source = DEFAULT_DATA_DIR / "relational" / "factories.csv"

        result = self.service.validate_csv(
            "fake_dataset",
            source,
        )

        self.assertFalse(result.accepted)
        self.assertEqual(
            result.issues[0].code,
            "undeclared_dataset",
        )

    def test_missing_required_column_is_rejected(self):
        source = DEFAULT_DATA_DIR / "relational" / "factories.csv"
        frame = pd.read_csv(source)
        frame = frame.drop(columns=["factory_id"])

        with tempfile.TemporaryDirectory() as temp:
            broken = Path(temp) / "factories.csv"
            frame.to_csv(broken, index=False)

            result = self.service.validate_csv(
                "factories",
                broken,
            )

        self.assertFalse(result.accepted)
        self.assertIn(
            "missing_columns",
            {issue.code for issue in result.issues},
        )

    def test_duplicate_primary_key_is_rejected(self):
        source = DEFAULT_DATA_DIR / "relational" / "factories.csv"
        frame = pd.read_csv(source)

        frame = pd.concat(
            [frame, frame.iloc[[0]]],
            ignore_index=True,
        )

        with tempfile.TemporaryDirectory() as temp:
            broken = Path(temp) / "factories.csv"
            frame.to_csv(broken, index=False)

            result = self.service.validate_csv(
                "factories",
                broken,
            )

        self.assertFalse(result.accepted)
        self.assertIn(
            "duplicate_primary_key",
            {issue.code for issue in result.issues},
        )


if __name__ == "__main__":
    unittest.main()

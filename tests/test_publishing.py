from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from linepulse.ingestion import CsvIngestionService
from linepulse.publishing import DatasetPublisher
from linepulse.settings import DEFAULT_DATA_DIR


class DatasetPublisherTests(unittest.TestCase):

    def setUp(self):
        self.ingestion = CsvIngestionService(DEFAULT_DATA_DIR)
        self.publisher = DatasetPublisher()

    def test_staged_csv_can_be_published(self):
        source = (
            DEFAULT_DATA_DIR
            / "relational"
            / "factories.csv"
        )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            staged = self.ingestion.validate_and_stage(
                "factories",
                source,
                root / "staging",
            )

            result = self.publisher.publish(
                "factories",
                staged.staged_path,
                root / "published",
            )

            self.assertTrue(
                Path(result.published_path).is_file()
            )

            pointer = Path(result.pointer_path)

            self.assertTrue(pointer.is_file())

            self.assertEqual(
                pointer.read_text(
                    encoding="utf-8"
                ).strip(),
                Path(result.published_path).name,
            )

            self.assertFalse(result.already_published)

    def test_repeated_publish_is_idempotent(self):
        source = (
            DEFAULT_DATA_DIR
            / "relational"
            / "factories.csv"
        )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            staged = self.ingestion.validate_and_stage(
                "factories",
                source,
                root / "staging",
            )

            first = self.publisher.publish(
                "factories",
                staged.staged_path,
                root / "published",
            )

            second = self.publisher.publish(
                "factories",
                staged.staged_path,
                root / "published",
            )

            self.assertEqual(
                first.published_path,
                second.published_path,
            )

            self.assertTrue(
                second.already_published
            )

    def test_tampered_staged_name_is_rejected(self):
        source = (
            DEFAULT_DATA_DIR
            / "relational"
            / "factories.csv"
        )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            fake_stage = (
                root
                / ("factories-" + "0" * 64 + ".csv")
            )

            fake_stage.write_bytes(
                source.read_bytes()
            )

            with self.assertRaises(ValueError):
                self.publisher.publish(
                    "factories",
                    fake_stage,
                    root / "published",
                )


if __name__ == "__main__":
    unittest.main()

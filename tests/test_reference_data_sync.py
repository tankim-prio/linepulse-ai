from __future__ import annotations

import unittest

import pandas as pd

from linepulse.database.reference_repository import (
    ReferenceDataSyncResult,
)
from linepulse.database.reference_sync import (
    FACTORY_COLUMNS,
    PRODUCTION_LINE_COLUMNS,
    ReferenceDataSyncService,
    _parse_bool,
    _validate_reference_frames,
)
from linepulse.settings import DEFAULT_DATA_DIR


class FakeReferenceRepository:
    def __init__(self):
        self.factories = []
        self.production_lines = []

    def sync(
        self,
        factories,
        production_lines,
    ):
        self.factories = list(
            factories
        )

        self.production_lines = list(
            production_lines
        )

        return ReferenceDataSyncResult(
            factories_processed=len(
                self.factories
            ),
            production_lines_processed=len(
                self.production_lines
            ),
        )


class ReferenceDataSyncTests(unittest.TestCase):
    def test_bundled_reference_data_is_loaded(
        self,
    ):
        repository = FakeReferenceRepository()

        service = ReferenceDataSyncService(
            DEFAULT_DATA_DIR,
            repository=repository,
        )

        result = service.sync()

        self.assertEqual(
            result.factories_processed,
            1,
        )

        self.assertEqual(
            result.production_lines_processed,
            8,
        )

        self.assertEqual(
            repository.factories[0][
                "factory_id"
            ],
            "FAC-001",
        )

        self.assertEqual(
            repository.factories[0][
                "factory_name"
            ],
            "Demo Apparel Works",
        )

        self.assertEqual(
            repository.production_lines[0][
                "line_id"
            ],
            "LINE-01",
        )

        self.assertEqual(
            repository.production_lines[0][
                "factory_id"
            ],
            "FAC-001",
        )

        self.assertIs(
            repository.production_lines[0][
                "active"
            ],
            True,
        )

    def test_unknown_factory_reference_is_rejected(
        self,
    ):
        factories = pd.DataFrame(
            [
                {
                    "factory_id": "FAC-001",
                    "factory_name": "Factory",
                    "country": "Bangladesh",
                    "timezone": "Asia/Dhaka",
                    "weekly_closure_day": "Friday",
                    "dataset_provenance": "synthetic",
                }
            ],
            columns=FACTORY_COLUMNS,
        )

        lines = pd.DataFrame(
            [
                {
                    "line_id": "LINE-01",
                    "factory_id": "FAC-MISSING",
                    "line_name": "Line",
                    "specialization": "knit",
                    "standard_operator_capacity": 48,
                    "planning_efficiency": 0.71,
                    "active": True,
                    "dataset_provenance": "synthetic",
                }
            ],
            columns=PRODUCTION_LINE_COLUMNS,
        )

        with self.assertRaisesRegex(
            ValueError,
            "unknown factory_id",
        ):
            _validate_reference_frames(
                factories,
                lines,
            )

    def test_duplicate_line_id_is_rejected(
        self,
    ):
        factories = pd.DataFrame(
            [
                {
                    "factory_id": "FAC-001",
                    "factory_name": "Factory",
                    "country": "Bangladesh",
                    "timezone": "Asia/Dhaka",
                    "weekly_closure_day": "Friday",
                    "dataset_provenance": "synthetic",
                }
            ],
            columns=FACTORY_COLUMNS,
        )

        line = {
            "line_id": "LINE-01",
            "factory_id": "FAC-001",
            "line_name": "Line",
            "specialization": "knit",
            "standard_operator_capacity": 48,
            "planning_efficiency": 0.71,
            "active": True,
            "dataset_provenance": "synthetic",
        }

        lines = pd.DataFrame(
            [
                line,
                line.copy(),
            ],
            columns=PRODUCTION_LINE_COLUMNS,
        )

        with self.assertRaisesRegex(
            ValueError,
            "duplicate line_id",
        ):
            _validate_reference_frames(
                factories,
                lines,
            )

    def test_boolean_parser_is_strict(
        self,
    ):
        self.assertTrue(
            _parse_bool("true")
        )

        self.assertFalse(
            _parse_bool("false")
        )

        with self.assertRaises(
            ValueError
        ):
            _parse_bool(
                "maybe"
            )


if __name__ == "__main__":
    unittest.main()

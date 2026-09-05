from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from linepulse.risk import RiskEventStore
from linepulse.settings import DEFAULT_DATA_DIR
from linepulse.workflows import VerticalSliceWorkflow


class VerticalSliceWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = VerticalSliceWorkflow(
            DEFAULT_DATA_DIR,
            risk_event_store_factory=RiskEventStore,
        )

    def make_small_hourly_csv(
        self,
        directory: Path,
    ) -> Path:
        source = self.workflow._dataset_path(
            "hourly_production"
        )

        destination = (
            directory
            / "hourly_production.csv"
        )

        pd.read_csv(source).head(5).to_csv(
            destination,
            index=False,
        )

        return destination

    def test_valid_upload_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            source = self.make_small_hourly_csv(
                root
            )

            result = (
                self.workflow
                .run_hourly_production_csv(
                    source,
                    root / "work",
                )
            )

            self.assertTrue(result.accepted)
            self.assertGreater(
                result.snapshot_count,
                0,
            )
            self.assertGreater(
                result.events_created,
                0,
            )
            self.assertTrue(
                Path(
                    result.published_path
                ).is_file()
            )
            self.assertTrue(
                Path(
                    result.risk_event_path
                ).is_file()
            )

    def test_repeated_run_reuses_risk_events(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            source = self.make_small_hourly_csv(
                root
            )

            first = (
                self.workflow
                .run_hourly_production_csv(
                    source,
                    root / "work",
                )
            )

            second = (
                self.workflow
                .run_hourly_production_csv(
                    source,
                    root / "work",
                )
            )

            self.assertTrue(first.accepted)
            self.assertTrue(second.accepted)

            self.assertGreater(
                first.events_created,
                0,
            )

            self.assertEqual(
                second.events_created,
                0,
            )

            self.assertEqual(
                second.events_reused,
                second.snapshot_count,
            )

    def test_invalid_upload_stops_before_publish(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            source = self.make_small_hourly_csv(
                root
            )

            frame = pd.read_csv(
                source
            ).drop(
                columns=["production_id"]
            )

            frame.to_csv(
                source,
                index=False,
            )

            result = (
                self.workflow
                .run_hourly_production_csv(
                    source,
                    root / "work",
                )
            )

            self.assertFalse(result.accepted)
            self.assertIsNone(
                result.published_path
            )
            self.assertIsNone(
                result.risk_event_path
            )


if __name__ == "__main__":
    unittest.main()

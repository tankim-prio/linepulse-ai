"""Tests for leakage-safe point-in-time daily snapshots."""

from __future__ import annotations

import unittest

import pandas as pd

from linepulse.data.contracts import FORBIDDEN_DAILY_RISK_FEATURES
from linepulse.features.daily_snapshot import (
    EXACT_RECONSTRUCTED_FEATURES,
    build_daily_snapshot_features,
    build_daily_snapshot_features_from_directory,
)
from linepulse.settings import DEFAULT_DATA_DIR


class DailySnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = pd.read_csv(
            DEFAULT_DATA_DIR / "modeling" / "daily_line_ml.csv"
        )
        cls.reference["work_date"] = pd.to_datetime(
            cls.reference["work_date"]
        )

        cls.built = build_daily_snapshot_features_from_directory()

    def test_confirmed_features_reconstruct_reference(self):
        keys = ["work_date", "line_id", "order_id"]

        merged = self.reference[
            keys + list(EXACT_RECONSTRUCTED_FEATURES)
        ].merge(
            self.built[
                keys + list(EXACT_RECONSTRUCTED_FEATURES)
            ],
            on=keys,
            suffixes=("_reference", "_built"),
            validate="one_to_one",
        )

        self.assertEqual(len(merged), 1224)

        categorical = {
            "complexity_band",
            "material_status",
            "root_cause_signal",
        }

        for feature in EXACT_RECONSTRUCTED_FEATURES:
            expected = merged[f"{feature}_reference"]
            actual = merged[f"{feature}_built"]

            if feature in categorical:
                self.assertTrue(
                    expected.astype(str).equals(
                        actual.astype(str)
                    ),
                    msg=f"{feature} does not reconstruct exactly.",
                )
                continue

            difference = (
                pd.to_numeric(expected, errors="coerce")
                - pd.to_numeric(actual, errors="coerce")
            ).abs()

            self.assertLessEqual(
                float(difference.max()),
                0.0001001,
                msg=(
                    f"{feature} maximum difference "
                    f"is {difference.max()}."
                ),
            )

    def test_post_snapshot_hour_is_ignored(self):
        hourly = pd.read_csv(
            DEFAULT_DATA_DIR / "relational" / "hourly_production.csv"
        )
        attendance = pd.read_csv(
            DEFAULT_DATA_DIR
            / "relational"
            / "attendance_summaries.csv"
        )
        notes = pd.read_csv(
            DEFAULT_DATA_DIR / "relational" / "supervisor_notes.csv"
        )
        styles = pd.read_csv(
            DEFAULT_DATA_DIR / "relational" / "styles.csv"
        )

        before = build_daily_snapshot_features(
            hourly,
            attendance,
            notes,
            styles,
        )

        hour_number = (
            hourly["hour_start"]
            .astype(str)
            .str.slice(0, 2)
            .astype(int)
        )

        post_index = hourly.index[hour_number >= 13][0]

        key = {
            "work_date": pd.to_datetime(
                hourly.loc[post_index, "work_date"]
            ),
            "line_id": hourly.loc[post_index, "line_id"],
            "order_id": hourly.loc[post_index, "order_id"],
        }

        modified = hourly.copy()

        modified.loc[post_index, "actual_output"] += 999999
        modified.loc[post_index, "downtime_minutes"] += 999999
        modified.loc[post_index, "inspected_qty"] += 999999
        modified.loc[post_index, "defect_count"] += 999999
        modified.loc[post_index, "material_status"] = "future_only_value"

        after = build_daily_snapshot_features(
            modified,
            attendance,
            notes,
            styles,
        )

        def select(frame):
            return frame[
                (frame["work_date"] == key["work_date"])
                & (frame["line_id"] == key["line_id"])
                & (frame["order_id"] == key["order_id"])
            ].iloc[0]

        before_row = select(before)
        after_row = select(after)

        point_in_time_fields = [
            "target_to_snapshot",
            "output_to_snapshot",
            "progress_ratio",
            "downtime_to_snapshot",
            "inspected_to_snapshot",
            "defects_to_snapshot",
            "defect_rate_to_snapshot",
            "material_status",
            "root_cause_signal",
        ]

        for field in point_in_time_fields:
            self.assertEqual(
                before_row[field],
                after_row[field],
                msg=f"Post-snapshot data leaked into {field}.",
            )

    def test_forbidden_future_fields_are_absent(self):
        leaked = (
            set(self.built.columns)
            & FORBIDDEN_DAILY_RISK_FEATURES
        )
        self.assertEqual(leaked, set())


if __name__ == "__main__":
    unittest.main()

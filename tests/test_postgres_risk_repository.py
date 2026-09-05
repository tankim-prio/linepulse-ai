from __future__ import annotations

import unittest

from linepulse.database.risk_repository import (
    _parse_snapshot_at,
)


class PostgresRiskRepositoryTests(
    unittest.TestCase
):
    def test_naive_snapshot_timestamp_is_rejected(
        self,
    ):
        with self.assertRaises(ValueError):
            _parse_snapshot_at(
                "2026-09-06T13:00:00"
            )

    def test_timezone_aware_snapshot_is_accepted(
        self,
    ):
        result = _parse_snapshot_at(
            "2026-09-06T13:00:00+00:00"
        )

        self.assertIsNotNone(
            result.tzinfo
        )


if __name__ == "__main__":
    unittest.main()

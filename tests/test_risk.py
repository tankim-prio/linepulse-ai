from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from linepulse.risk import (
    RiskEventStore,
    build_risk_event,
    evaluate_rule_risk,
)


class RuleRiskTests(unittest.TestCase):

    def sample_snapshot(self):
        return {
            "factory_id": "FAC-001",
            "line_id": "LINE-01",
            "order_id": "ORD-001",
            "snapshot_at": "2025-01-04T13:00:00+00:00",
            "progress_ratio": 0.60,
            "planned_workers": 50,
            "present_workers": 45,
            "downtime_to_snapshot": 20,
            "defects_to_snapshot": 4,
            "material_status": "available",
            "root_cause_signal": "machine",
        }

    def test_progress_gap_score_is_deterministic(self):
        result = evaluate_rule_risk(
            self.sample_snapshot()
        )

        self.assertEqual(
            result.risk_score,
            0.40,
        )

    def test_adverse_signals_are_explained(self):
        result = evaluate_rule_risk(
            self.sample_snapshot()
        )

        self.assertIn(
            "progress_below_snapshot_target",
            result.factors,
        )

        self.assertIn(
            "attendance_shortfall",
            result.factors,
        )

        self.assertIn(
            "downtime_observed",
            result.factors,
        )

        self.assertIn(
            "root_cause:machine",
            result.factors,
        )

    def test_event_id_is_stable(self):
        snapshot = self.sample_snapshot()
        assessment = evaluate_rule_risk(snapshot)

        first = build_risk_event(
            snapshot,
            assessment,
        )

        second = build_risk_event(
            snapshot,
            assessment,
        )

        self.assertEqual(
            first.event_id,
            second.event_id,
        )

    def test_event_store_is_idempotent(self):
        snapshot = self.sample_snapshot()
        assessment = evaluate_rule_risk(snapshot)

        event = build_risk_event(
            snapshot,
            assessment,
        )

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "risk_events.jsonl"
            store = RiskEventStore(path)

            first = store.append(event)
            second = store.append(event)

            self.assertTrue(first)
            self.assertFalse(second)

            lines = path.read_text(
                encoding="utf-8"
            ).splitlines()

            self.assertEqual(len(lines), 1)


if __name__ == "__main__":
    unittest.main()

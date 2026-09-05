from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from linepulse.database.analytics_repository import (
    AnalyticsOverviewRecord,
    LineRiskSummaryRecord,
)
from linepulse.main import (
    app,
    get_analytics_repository,
)


class FakeAnalyticsRepository:
    def get_overview(
        self,
    ):
        return AnalyticsOverviewRecord(
            factory_count=1,
            production_line_count=8,
            risk_event_count=1224,
            average_risk_score=0.1154,
            maximum_risk_score=0.6826,
            latest_snapshot_at=(
                "2025-06-30T19:00:00+06:00"
            ),
        )

    def list_line_summaries(
        self,
        *,
        factory_id=None,
    ):
        records = [
            LineRiskSummaryRecord(
                factory_id="FAC-001",
                line_id="LINE-01",
                line_name="Knit Line 1",
                specialization="knit",
                active=True,
                event_count=153,
                average_risk_score=0.1131,
                maximum_risk_score=0.6699,
                latest_snapshot_at=(
                    "2025-06-30T19:00:00+06:00"
                ),
            ),
            LineRiskSummaryRecord(
                factory_id="FAC-001",
                line_id="LINE-02",
                line_name="Knit Line 2",
                specialization="knit",
                active=True,
                event_count=153,
                average_risk_score=0.1184,
                maximum_risk_score=0.6826,
                latest_snapshot_at=(
                    "2025-06-30T19:00:00+06:00"
                ),
            ),
        ]

        if (
            factory_id is not None
            and factory_id != "FAC-001"
        ):
            return []

        return records


class AnalyticsApiTests(unittest.TestCase):
    def setUp(self):
        self.repository = (
            FakeAnalyticsRepository()
        )

        app.dependency_overrides[
            get_analytics_repository
        ] = lambda: self.repository

        self.client = TestClient(
            app
        )

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_overview_returns_metrics(
        self,
    ):
        response = self.client.get(
            "/api/analytics/overview"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        body = response.json()

        self.assertEqual(
            body["factory_count"],
            1,
        )

        self.assertEqual(
            body["production_line_count"],
            8,
        )

        self.assertEqual(
            body["risk_event_count"],
            1224,
        )

        self.assertEqual(
            body["maximum_risk_score"],
            0.6826,
        )

    def test_line_summaries_return_metrics(
        self,
    ):
        response = self.client.get(
            "/api/analytics/lines"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        body = response.json()

        self.assertEqual(
            len(body),
            2,
        )

        self.assertEqual(
            body[0]["line_id"],
            "LINE-01",
        )

        self.assertEqual(
            body[0]["event_count"],
            153,
        )

    def test_line_summary_factory_filter(
        self,
    ):
        response = self.client.get(
            "/api/analytics/lines",
            params={
                "factory_id": "FAC-MISSING",
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertEqual(
            response.json(),
            [],
        )


if __name__ == "__main__":
    unittest.main()

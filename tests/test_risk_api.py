from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from linepulse.main import (
    app,
    get_risk_event_repository,
)
from linepulse.risk import RiskEvent


class FakeRiskRepository:
    """In-memory repository used only by API unit tests."""

    def __init__(self):
        self.event = RiskEvent(
            event_id="event-001",
            factory_id="FAC-001",
            line_id="LINE-01",
            order_id="ORD-001",
            snapshot_at="2026-09-06T13:00:00+00:00",
            rule_version="progress-gap-v1",
            risk_score=0.4,
            factors=(
                "progress_below_snapshot_target",
                "attendance_shortfall",
            ),
        )

    def list_recent(
        self,
        *,
        limit=100,
        factory_id=None,
        line_id=None,
    ):
        events = [self.event]

        if (
            factory_id is not None
            and factory_id != self.event.factory_id
        ):
            return []

        if (
            line_id is not None
            and line_id != self.event.line_id
        ):
            return []

        return events[:limit]

    def get_by_event_id(
        self,
        event_id,
    ):
        if event_id == self.event.event_id:
            return self.event

        return None


class RiskEventApiTests(unittest.TestCase):
    def setUp(self):
        self.repository = FakeRiskRepository()

        app.dependency_overrides[
            get_risk_event_repository
        ] = lambda: self.repository

        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_list_risk_events_returns_200(self):
        response = self.client.get(
            "/api/risk-events"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        body = response.json()

        self.assertEqual(
            len(body),
            1,
        )

        self.assertEqual(
            body[0]["event_id"],
            "event-001",
        )

        self.assertEqual(
            body[0]["factors"],
            [
                "progress_below_snapshot_target",
                "attendance_shortfall",
            ],
        )

    def test_list_risk_events_supports_line_filter(
        self,
    ):
        response = self.client.get(
            "/api/risk-events",
            params={
                "line_id": "OTHER-LINE",
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

    def test_get_risk_event_returns_200(self):
        response = self.client.get(
            "/api/risk-events/event-001"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        body = response.json()

        self.assertEqual(
            body["factory_id"],
            "FAC-001",
        )

        self.assertEqual(
            body["line_id"],
            "LINE-01",
        )

    def test_get_missing_event_returns_404(self):
        response = self.client.get(
            "/api/risk-events/missing-event"
        )

        self.assertEqual(
            response.status_code,
            404,
        )

        self.assertEqual(
            response.json(),
            {
                "detail": "Risk event not found."
            },
        )

    def test_invalid_limit_returns_422(self):
        response = self.client.get(
            "/api/risk-events",
            params={
                "limit": 0,
            },
        )

        self.assertEqual(
            response.status_code,
            422,
        )


if __name__ == "__main__":
    unittest.main()

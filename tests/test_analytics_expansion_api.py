from __future__ import annotations

from fastapi.testclient import TestClient

from linepulse.database.analytics_repository import (
    FactorSummaryRecord,
    RiskTrendPointRecord,
)
from linepulse.main import (
    app,
    get_analytics_repository,
)


class FakeExpandedAnalyticsRepository:
    def __init__(self) -> None:
        self.trend_filters = None
        self.factor_filters = None

    def list_risk_trend(
        self,
        *,
        factory_id=None,
        line_id=None,
        rule_version=None,
    ):
        self.trend_filters = (
            factory_id,
            line_id,
            rule_version,
        )

        if factory_id == "FAC-MISSING":
            return []

        return [
            RiskTrendPointRecord(
                rule_version="progress-gap-v1",
                snapshot_at=(
                    "2025-06-29T19:00:00+06:00"
                ),
                event_count=8,
                average_risk_score=0.12,
                maximum_risk_score=0.61,
            ),
            RiskTrendPointRecord(
                rule_version="progress-gap-v1",
                snapshot_at=(
                    "2025-06-30T19:00:00+06:00"
                ),
                event_count=8,
                average_risk_score=0.14,
                maximum_risk_score=0.68,
            ),
        ]

    def list_factor_summaries(
        self,
        *,
        factory_id=None,
        line_id=None,
        rule_version=None,
    ):
        self.factor_filters = (
            factory_id,
            line_id,
            rule_version,
        )

        if line_id == "LINE-MISSING":
            return []

        return [
            FactorSummaryRecord(
                rule_version="progress-gap-v1",
                factor=(
                    "progress_below_snapshot_target"
                ),
                occurrence_count=900,
            ),
            FactorSummaryRecord(
                rule_version="progress-gap-v1",
                factor="attendance_shortfall",
                occurrence_count=700,
            ),
        ]


def test_risk_trend_endpoint():
    repository = (
        FakeExpandedAnalyticsRepository()
    )

    app.dependency_overrides[
        get_analytics_repository
    ] = lambda: repository

    try:
        client = TestClient(app)

        response = client.get(
            "/api/analytics/risk-trend",
            params={
                "factory_id": "FAC-001",
                "line_id": "LINE-01",
                "rule_version": "progress-gap-v1",
            },
        )

        assert response.status_code == 200

        body = response.json()

        assert len(body) == 2

        assert body[0] == {
            "rule_version": "progress-gap-v1",
            "snapshot_at": (
                "2025-06-29T19:00:00+06:00"
            ),
            "event_count": 8,
            "average_risk_score": 0.12,
            "maximum_risk_score": 0.61,
        }

        assert repository.trend_filters == (
            "FAC-001",
            "LINE-01",
            "progress-gap-v1",
        )

    finally:
        app.dependency_overrides.clear()


def test_risk_trend_missing_factory():
    repository = (
        FakeExpandedAnalyticsRepository()
    )

    app.dependency_overrides[
        get_analytics_repository
    ] = lambda: repository

    try:
        client = TestClient(app)

        response = client.get(
            "/api/analytics/risk-trend",
            params={
                "factory_id": "FAC-MISSING",
            },
        )

        assert response.status_code == 200
        assert response.json() == []

    finally:
        app.dependency_overrides.clear()


def test_factor_summary_endpoint():
    repository = (
        FakeExpandedAnalyticsRepository()
    )

    app.dependency_overrides[
        get_analytics_repository
    ] = lambda: repository

    try:
        client = TestClient(app)

        response = client.get(
            "/api/analytics/factors",
            params={
                "factory_id": "FAC-001",
                "line_id": "LINE-01",
                "rule_version": "progress-gap-v1",
            },
        )

        assert response.status_code == 200

        body = response.json()

        assert len(body) == 2

        assert body[0] == {
            "rule_version": "progress-gap-v1",
            "factor": (
                "progress_below_snapshot_target"
            ),
            "occurrence_count": 900,
        }

        assert repository.factor_filters == (
            "FAC-001",
            "LINE-01",
            "progress-gap-v1",
        )

    finally:
        app.dependency_overrides.clear()


def test_factor_summary_missing_line():
    repository = (
        FakeExpandedAnalyticsRepository()
    )

    app.dependency_overrides[
        get_analytics_repository
    ] = lambda: repository

    try:
        client = TestClient(app)

        response = client.get(
            "/api/analytics/factors",
            params={
                "line_id": "LINE-MISSING",
            },
        )

        assert response.status_code == 200
        assert response.json() == []

    finally:
        app.dependency_overrides.clear()

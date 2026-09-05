"""PostgreSQL analytics queries for LinePulse dashboard data."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session, sessionmaker

from linepulse.database.connection import SessionLocal
from linepulse.database.models import (
    Factory,
    ProductionLine,
    RiskEventModel,
)


@dataclass(frozen=True)
class AnalyticsOverviewRecord:
    """Factory-wide persisted risk overview."""

    factory_count: int
    production_line_count: int
    risk_event_count: int
    average_risk_score: float | None
    maximum_risk_score: float | None
    latest_snapshot_at: str | None


@dataclass(frozen=True)
class LineRiskSummaryRecord:
    """Persisted risk summary for one production line."""

    factory_id: str
    line_id: str
    line_name: str
    specialization: str
    active: bool
    event_count: int
    average_risk_score: float | None
    maximum_risk_score: float | None
    latest_snapshot_at: str | None


class PostgresAnalyticsRepository:
    """Read-only PostgreSQL analytics repository."""

    def __init__(
        self,
        session_factory: sessionmaker[Session] = SessionLocal,
    ) -> None:
        self.session_factory = session_factory

    def get_overview(
        self,
    ) -> AnalyticsOverviewRecord:
        """Return persisted platform-wide summary metrics."""

        with self.session_factory() as session:
            factory_count = session.scalar(
                select(
                    func.count(Factory.id)
                )
            )

            line_count = session.scalar(
                select(
                    func.count(
                        ProductionLine.id
                    )
                )
            )

            stats = session.execute(
                select(
                    func.count(
                        RiskEventModel.id
                    ),
                    func.avg(
                        RiskEventModel.risk_score
                    ),
                    func.max(
                        RiskEventModel.risk_score
                    ),
                    func.max(
                        RiskEventModel.snapshot_at
                    ),
                )
            ).one()

        event_count = int(
            stats[0] or 0
        )

        average_risk = (
            float(stats[1])
            if stats[1] is not None
            else None
        )

        maximum_risk = (
            float(stats[2])
            if stats[2] is not None
            else None
        )

        latest_snapshot = (
            stats[3].isoformat()
            if stats[3] is not None
            else None
        )

        return AnalyticsOverviewRecord(
            factory_count=int(
                factory_count or 0
            ),
            production_line_count=int(
                line_count or 0
            ),
            risk_event_count=event_count,
            average_risk_score=average_risk,
            maximum_risk_score=maximum_risk,
            latest_snapshot_at=latest_snapshot,
        )

    def list_line_summaries(
        self,
        *,
        factory_id: str | None = None,
    ) -> list[LineRiskSummaryRecord]:
        """Return risk metrics grouped by production line."""

        statement = (
            select(
                ProductionLine.factory_id.label(
                    "factory_id"
                ),
                ProductionLine.line_id.label(
                    "line_id"
                ),
                ProductionLine.line_name.label(
                    "line_name"
                ),
                ProductionLine.specialization.label(
                    "specialization"
                ),
                ProductionLine.active.label(
                    "active"
                ),
                func.count(
                    RiskEventModel.id
                ).label(
                    "event_count"
                ),
                func.avg(
                    RiskEventModel.risk_score
                ).label(
                    "average_risk_score"
                ),
                func.max(
                    RiskEventModel.risk_score
                ).label(
                    "maximum_risk_score"
                ),
                func.max(
                    RiskEventModel.snapshot_at
                ).label(
                    "latest_snapshot_at"
                ),
            )
            .select_from(
                ProductionLine
            )
            .outerjoin(
                RiskEventModel,
                and_(
                    RiskEventModel.line_id
                    == ProductionLine.line_id,
                    RiskEventModel.factory_id
                    == ProductionLine.factory_id,
                ),
            )
        )

        if factory_id is not None:
            statement = statement.where(
                ProductionLine.factory_id
                == factory_id
            )

        statement = (
            statement
            .group_by(
                ProductionLine.factory_id,
                ProductionLine.line_id,
                ProductionLine.line_name,
                ProductionLine.specialization,
                ProductionLine.active,
            )
            .order_by(
                ProductionLine.line_id
            )
        )

        with self.session_factory() as session:
            rows = session.execute(
                statement
            ).mappings().all()

        results: list[
            LineRiskSummaryRecord
        ] = []

        for row in rows:
            latest_snapshot = (
                row["latest_snapshot_at"].isoformat()
                if row["latest_snapshot_at"]
                is not None
                else None
            )

            average_risk = (
                float(
                    row[
                        "average_risk_score"
                    ]
                )
                if row[
                    "average_risk_score"
                ] is not None
                else None
            )

            maximum_risk = (
                float(
                    row[
                        "maximum_risk_score"
                    ]
                )
                if row[
                    "maximum_risk_score"
                ] is not None
                else None
            )

            results.append(
                LineRiskSummaryRecord(
                    factory_id=str(
                        row["factory_id"]
                    ),
                    line_id=str(
                        row["line_id"]
                    ),
                    line_name=str(
                        row["line_name"]
                    ),
                    specialization=str(
                        row["specialization"]
                    ),
                    active=bool(
                        row["active"]
                    ),
                    event_count=int(
                        row["event_count"]
                    ),
                    average_risk_score=(
                        average_risk
                    ),
                    maximum_risk_score=(
                        maximum_risk
                    ),
                    latest_snapshot_at=(
                        latest_snapshot
                    ),
                )
            )

        return results

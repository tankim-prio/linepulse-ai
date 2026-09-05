"""PostgreSQL persistence for LinePulse risk events."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from linepulse.database.connection import SessionLocal
from linepulse.database.models import RiskEventModel
from linepulse.risk.events import RiskEvent


def _parse_snapshot_at(value: str) -> datetime:
    """Convert an ISO timestamp to an aware datetime."""

    parsed = datetime.fromisoformat(value)

    if parsed.tzinfo is None:
        raise ValueError(
            "RiskEvent snapshot_at must include timezone information."
        )

    return parsed


def _to_risk_event(
    model: RiskEventModel,
) -> RiskEvent:
    """Convert a database row to the domain RiskEvent."""

    return RiskEvent(
        event_id=model.event_id,
        factory_id=model.factory_id,
        line_id=model.line_id,
        order_id=model.order_id,
        snapshot_at=model.snapshot_at.isoformat(),
        rule_version=model.rule_version,
        risk_score=model.risk_score,
        factors=tuple(model.factors),
    )


class PostgresRiskEventRepository:
    """PostgreSQL repository for LinePulse RiskEvent objects."""

    def __init__(
        self,
        session_factory: sessionmaker[Session] = SessionLocal,
    ) -> None:
        self.session_factory = session_factory

    def append(
        self,
        event: RiskEvent,
    ) -> bool:
        """Insert once; return False when event_id already exists."""

        statement = (
            insert(RiskEventModel)
            .values(
                event_id=event.event_id,
                factory_id=event.factory_id,
                line_id=event.line_id,
                order_id=event.order_id,
                snapshot_at=_parse_snapshot_at(
                    event.snapshot_at
                ),
                rule_version=event.rule_version,
                risk_score=event.risk_score,
                factors=list(event.factors),
            )
            .on_conflict_do_nothing(
                index_elements=[
                    RiskEventModel.event_id
                ],
            )
        )

        with self.session_factory() as session:
            result = session.execute(statement)
            session.commit()

            return result.rowcount == 1

    def get_by_event_id(
        self,
        event_id: str,
    ) -> RiskEvent | None:
        """Return one event by stable event ID."""

        statement = select(
            RiskEventModel
        ).where(
            RiskEventModel.event_id == event_id
        )

        with self.session_factory() as session:
            model = session.scalar(statement)

            if model is None:
                return None

            return _to_risk_event(model)

    def list_recent(
        self,
        *,
        limit: int = 100,
        factory_id: str | None = None,
        line_id: str | None = None,
    ) -> list[RiskEvent]:
        """Return recent risk events with optional filters."""

        if limit < 1 or limit > 500:
            raise ValueError(
                "limit must be between 1 and 500."
            )

        statement = select(
            RiskEventModel
        )

        if factory_id is not None:
            statement = statement.where(
                RiskEventModel.factory_id
                == factory_id
            )

        if line_id is not None:
            statement = statement.where(
                RiskEventModel.line_id
                == line_id
            )

        statement = statement.order_by(
            RiskEventModel.snapshot_at.desc(),
            RiskEventModel.id.desc(),
        ).limit(limit)

        with self.session_factory() as session:
            models = session.scalars(
                statement
            ).all()

            return [
                _to_risk_event(model)
                for model in models
            ]

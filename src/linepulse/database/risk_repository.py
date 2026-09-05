"""PostgreSQL persistence for LinePulse risk events."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from linepulse.database.connection import SessionLocal
from linepulse.database.models import RiskEventModel
from linepulse.risk.events import RiskEvent


def _parse_snapshot_at(value: str) -> datetime:
    """Convert the RiskEvent ISO timestamp to an aware datetime."""

    parsed = datetime.fromisoformat(value)

    if parsed.tzinfo is None:
        raise ValueError(
            "RiskEvent snapshot_at must include timezone information."
        )

    return parsed


class PostgresRiskEventRepository:
    """Idempotent PostgreSQL repository for RiskEvent objects."""

    def __init__(
        self,
        session_factory: sessionmaker[Session] = SessionLocal,
    ) -> None:
        self.session_factory = session_factory

    def append(self, event: RiskEvent) -> bool:
        """
        Insert a risk event once.

        Returns True when a new row is created.
        Returns False when event_id already exists.
        """

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

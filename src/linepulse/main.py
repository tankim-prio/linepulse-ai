"""FastAPI entry point for LinePulse AI."""

from __future__ import annotations

from typing import Annotated

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
)
from pydantic import BaseModel
from sqlalchemy import text

from linepulse.database.connection import engine
from linepulse.database.risk_repository import (
    PostgresRiskEventRepository,
)
from linepulse.risk import RiskEvent


app = FastAPI(
    title="LinePulse AI API",
    version="0.1.0",
    description=(
        "Production risk intelligence API "
        "for LinePulse AI."
    ),
)


class RiskEventResponse(BaseModel):
    event_id: str
    factory_id: str
    line_id: str
    order_id: str
    snapshot_at: str
    rule_version: str
    risk_score: float
    factors: list[str]


def _response_from_event(
    event: RiskEvent,
) -> RiskEventResponse:
    return RiskEventResponse(
        event_id=event.event_id,
        factory_id=event.factory_id,
        line_id=event.line_id,
        order_id=event.order_id,
        snapshot_at=event.snapshot_at,
        rule_version=event.rule_version,
        risk_score=event.risk_score,
        factors=list(event.factors),
    )


def get_risk_event_repository(
) -> PostgresRiskEventRepository:
    """Create the API repository dependency."""

    return PostgresRiskEventRepository()


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "LinePulse AI",
        "status": "running",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
    }


@app.get("/health/database")
def database_health() -> dict[str, str]:
    with engine.connect() as connection:
        connection.execute(
            text("SELECT 1")
        )

    return {
        "status": "ok",
        "database": "connected",
    }


@app.get(
    "/api/risk-events",
    response_model=list[RiskEventResponse],
)
def list_risk_events(
    limit: Annotated[
        int,
        Query(ge=1, le=500),
    ] = 100,
    factory_id: str | None = None,
    line_id: str | None = None,
    repository: PostgresRiskEventRepository = Depends(
        get_risk_event_repository
    ),
) -> list[RiskEventResponse]:
    """Return recent persisted risk events."""

    events = repository.list_recent(
        limit=limit,
        factory_id=factory_id,
        line_id=line_id,
    )

    return [
        _response_from_event(event)
        for event in events
    ]


@app.get(
    "/api/risk-events/{event_id}",
    response_model=RiskEventResponse,
)
def get_risk_event(
    event_id: str,
    repository: PostgresRiskEventRepository = Depends(
        get_risk_event_repository
    ),
) -> RiskEventResponse:
    """Return one persisted risk event."""

    event = repository.get_by_event_id(
        event_id
    )

    if event is None:
        raise HTTPException(
            status_code=404,
            detail="Risk event not found.",
        )

    return _response_from_event(event)

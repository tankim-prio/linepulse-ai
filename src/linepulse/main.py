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
from linepulse.database.analytics_repository import (
    AnalyticsOverviewRecord,
    LineRiskSummaryRecord,
    PostgresAnalyticsRepository,
)
from linepulse.database.reference_repository import (
    FactoryRecord,
    PostgresReferenceDataRepository,
    ProductionLineRecord,
)
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


class FactoryResponse(BaseModel):
    factory_id: str
    factory_name: str
    country: str
    timezone: str
    weekly_closure_day: str
    dataset_provenance: str


class ProductionLineResponse(BaseModel):
    line_id: str
    factory_id: str
    line_name: str
    specialization: str
    standard_operator_capacity: int
    planning_efficiency: float
    active: bool
    dataset_provenance: str



class AnalyticsOverviewResponse(BaseModel):
    factory_count: int
    production_line_count: int
    risk_event_count: int
    average_risk_score: float | None
    maximum_risk_score: float | None
    latest_snapshot_at: str | None


class LineRiskSummaryResponse(BaseModel):
    factory_id: str
    line_id: str
    line_name: str
    specialization: str
    active: bool
    event_count: int
    average_risk_score: float | None
    maximum_risk_score: float | None
    latest_snapshot_at: str | None

def _risk_response(
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


def _factory_response(
    record: FactoryRecord,
) -> FactoryResponse:
    return FactoryResponse(
        factory_id=record.factory_id,
        factory_name=record.factory_name,
        country=record.country,
        timezone=record.timezone,
        weekly_closure_day=record.weekly_closure_day,
        dataset_provenance=record.dataset_provenance,
    )


def _production_line_response(
    record: ProductionLineRecord,
) -> ProductionLineResponse:
    return ProductionLineResponse(
        line_id=record.line_id,
        factory_id=record.factory_id,
        line_name=record.line_name,
        specialization=record.specialization,
        standard_operator_capacity=(
            record.standard_operator_capacity
        ),
        planning_efficiency=(
            record.planning_efficiency
        ),
        active=record.active,
        dataset_provenance=(
            record.dataset_provenance
        ),
    )



def _analytics_overview_response(
    record: AnalyticsOverviewRecord,
) -> AnalyticsOverviewResponse:
    return AnalyticsOverviewResponse(
        factory_count=record.factory_count,
        production_line_count=(
            record.production_line_count
        ),
        risk_event_count=(
            record.risk_event_count
        ),
        average_risk_score=(
            record.average_risk_score
        ),
        maximum_risk_score=(
            record.maximum_risk_score
        ),
        latest_snapshot_at=(
            record.latest_snapshot_at
        ),
    )


def _line_risk_summary_response(
    record: LineRiskSummaryRecord,
) -> LineRiskSummaryResponse:
    return LineRiskSummaryResponse(
        factory_id=record.factory_id,
        line_id=record.line_id,
        line_name=record.line_name,
        specialization=record.specialization,
        active=record.active,
        event_count=record.event_count,
        average_risk_score=(
            record.average_risk_score
        ),
        maximum_risk_score=(
            record.maximum_risk_score
        ),
        latest_snapshot_at=(
            record.latest_snapshot_at
        ),
    )

def get_risk_event_repository(
) -> PostgresRiskEventRepository:
    return PostgresRiskEventRepository()



def get_analytics_repository(
) -> PostgresAnalyticsRepository:
    return PostgresAnalyticsRepository()

def get_reference_data_repository(
) -> PostgresReferenceDataRepository:
    return PostgresReferenceDataRepository()


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
    events = repository.list_recent(
        limit=limit,
        factory_id=factory_id,
        line_id=line_id,
    )

    return [
        _risk_response(event)
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
    event = repository.get_by_event_id(
        event_id
    )

    if event is None:
        raise HTTPException(
            status_code=404,
            detail="Risk event not found.",
        )

    return _risk_response(
        event
    )


@app.get(
    "/api/factories",
    response_model=list[FactoryResponse],
)
def list_factories(
    repository: PostgresReferenceDataRepository = Depends(
        get_reference_data_repository
    ),
) -> list[FactoryResponse]:
    records = repository.list_factories()

    return [
        _factory_response(record)
        for record in records
    ]


@app.get(
    "/api/factories/{factory_id}",
    response_model=FactoryResponse,
)
def get_factory(
    factory_id: str,
    repository: PostgresReferenceDataRepository = Depends(
        get_reference_data_repository
    ),
) -> FactoryResponse:
    record = repository.get_factory(
        factory_id
    )

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Factory not found.",
        )

    return _factory_response(
        record
    )


@app.get(
    "/api/production-lines",
    response_model=list[ProductionLineResponse],
)
def list_production_lines(
    factory_id: str | None = None,
    active: bool | None = None,
    repository: PostgresReferenceDataRepository = Depends(
        get_reference_data_repository
    ),
) -> list[ProductionLineResponse]:
    records = repository.list_production_lines(
        factory_id=factory_id,
        active=active,
    )

    return [
        _production_line_response(
            record
        )
        for record in records
    ]


@app.get(
    "/api/production-lines/{line_id}",
    response_model=ProductionLineResponse,
)
def get_production_line(
    line_id: str,
    repository: PostgresReferenceDataRepository = Depends(
        get_reference_data_repository
    ),
) -> ProductionLineResponse:
    record = repository.get_production_line(
        line_id
    )

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Production line not found.",
        )

    return _production_line_response(
        record
    )


@app.get(
    "/api/analytics/overview",
    response_model=AnalyticsOverviewResponse,
)
def get_analytics_overview(
    repository: PostgresAnalyticsRepository = Depends(
        get_analytics_repository
    ),
) -> AnalyticsOverviewResponse:
    record = repository.get_overview()

    return _analytics_overview_response(
        record
    )


@app.get(
    "/api/analytics/lines",
    response_model=list[LineRiskSummaryResponse],
)
def list_line_risk_summaries(
    factory_id: str | None = None,
    repository: PostgresAnalyticsRepository = Depends(
        get_analytics_repository
    ),
) -> list[LineRiskSummaryResponse]:
    records = repository.list_line_summaries(
        factory_id=factory_id
    )

    return [
        _line_risk_summary_response(
            record
        )
        for record in records
    ]

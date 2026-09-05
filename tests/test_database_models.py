"""Tests for the LinePulse database schema."""

from sqlalchemy import DateTime, JSON

from linepulse.database import models  # noqa: F401
from linepulse.database.connection import Base


def test_reference_models_match_source_schema() -> None:
    factories = Base.metadata.tables["factories"]
    production_lines = Base.metadata.tables["production_lines"]

    assert set(factories.columns.keys()) == {
        "id",
        "factory_id",
        "factory_name",
        "country",
        "timezone",
        "weekly_closure_day",
        "dataset_provenance",
    }

    assert set(production_lines.columns.keys()) == {
        "id",
        "line_id",
        "factory_id",
        "line_name",
        "specialization",
        "standard_operator_capacity",
        "planning_efficiency",
        "active",
        "dataset_provenance",
    }

    foreign_keys = {
        fk.target_fullname
        for fk in production_lines.c.factory_id.foreign_keys
    }

    assert foreign_keys == {"factories.factory_id"}


def test_risk_event_model_preserves_event_schema() -> None:
    risk_events = Base.metadata.tables["risk_events"]

    assert set(risk_events.columns.keys()) == {
        "id",
        "event_id",
        "factory_id",
        "line_id",
        "order_id",
        "snapshot_at",
        "rule_version",
        "risk_score",
        "factors",
        "created_at",
    }

    assert isinstance(risk_events.c.factors.type, JSON)
    assert isinstance(risk_events.c.snapshot_at.type, DateTime)
    assert risk_events.c.snapshot_at.type.timezone is True
    assert risk_events.c.created_at.type.timezone is True
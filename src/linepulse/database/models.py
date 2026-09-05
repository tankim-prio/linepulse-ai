"""LinePulse AI database models."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from linepulse.database.connection import Base


class Factory(Base):
    """Factory reference data."""

    __tablename__ = "factories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    factory_id: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        nullable=False,
    )

    factory_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    country: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    timezone: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    weekly_closure_day: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    dataset_provenance: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )


class ProductionLine(Base):
    """Production line reference data."""

    __tablename__ = "production_lines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    line_id: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        nullable=False,
    )

    factory_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("factories.factory_id"),
        nullable=False,
        index=True,
    )

    line_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    specialization: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    standard_operator_capacity: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    planning_efficiency: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )

    dataset_provenance: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )


class RiskEventModel(Base):
    """Persisted LinePulse rule-based risk event."""

    __tablename__ = "risk_events"

    __table_args__ = (
        Index(
            "ix_risk_events_line_snapshot",
            "line_id",
            "snapshot_at",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    event_id: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
    )

    factory_id: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    line_id: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    order_id: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    snapshot_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    rule_version: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    risk_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    factors: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

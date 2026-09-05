"""PostgreSQL persistence for LinePulse reference data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from linepulse.database.connection import SessionLocal
from linepulse.database.models import (
    Factory,
    ProductionLine,
)


@dataclass(frozen=True)
class ReferenceDataSyncResult:
    """Summary of one reference-data synchronization."""

    factories_processed: int
    production_lines_processed: int


@dataclass(frozen=True)
class FactoryRecord:
    """Read-safe factory reference record."""

    factory_id: str
    factory_name: str
    country: str
    timezone: str
    weekly_closure_day: str
    dataset_provenance: str


@dataclass(frozen=True)
class ProductionLineRecord:
    """Read-safe production-line reference record."""

    line_id: str
    factory_id: str
    line_name: str
    specialization: str
    standard_operator_capacity: int
    planning_efficiency: float
    active: bool
    dataset_provenance: str


def _to_factory_record(
    model: Factory,
) -> FactoryRecord:
    return FactoryRecord(
        factory_id=model.factory_id,
        factory_name=model.factory_name,
        country=model.country,
        timezone=model.timezone,
        weekly_closure_day=model.weekly_closure_day,
        dataset_provenance=model.dataset_provenance,
    )


def _to_production_line_record(
    model: ProductionLine,
) -> ProductionLineRecord:
    return ProductionLineRecord(
        line_id=model.line_id,
        factory_id=model.factory_id,
        line_name=model.line_name,
        specialization=model.specialization,
        standard_operator_capacity=(
            model.standard_operator_capacity
        ),
        planning_efficiency=model.planning_efficiency,
        active=model.active,
        dataset_provenance=model.dataset_provenance,
    )


class PostgresReferenceDataRepository:
    """Synchronize and read factory/production-line reference data."""

    def __init__(
        self,
        session_factory: sessionmaker[Session] = SessionLocal,
    ) -> None:
        self.session_factory = session_factory

    def sync(
        self,
        factories: list[Mapping[str, Any]],
        production_lines: list[Mapping[str, Any]],
    ) -> ReferenceDataSyncResult:
        """
        Upsert reference data atomically.

        Factories are synchronized before production lines so the
        production_lines.factory_id foreign key is satisfied.
        """

        factory_rows = [
            dict(row)
            for row in factories
        ]

        line_rows = [
            dict(row)
            for row in production_lines
        ]

        with self.session_factory.begin() as session:
            if factory_rows:
                factory_insert = insert(
                    Factory
                ).values(
                    factory_rows
                )

                factory_statement = (
                    factory_insert
                    .on_conflict_do_update(
                        index_elements=[
                            Factory.factory_id
                        ],
                        set_={
                            "factory_name":
                                factory_insert.excluded.factory_name,
                            "country":
                                factory_insert.excluded.country,
                            "timezone":
                                factory_insert.excluded.timezone,
                            "weekly_closure_day":
                                factory_insert.excluded.weekly_closure_day,
                            "dataset_provenance":
                                factory_insert.excluded.dataset_provenance,
                        },
                    )
                )

                session.execute(
                    factory_statement
                )

            if line_rows:
                line_insert = insert(
                    ProductionLine
                ).values(
                    line_rows
                )

                line_statement = (
                    line_insert
                    .on_conflict_do_update(
                        index_elements=[
                            ProductionLine.line_id
                        ],
                        set_={
                            "factory_id":
                                line_insert.excluded.factory_id,
                            "line_name":
                                line_insert.excluded.line_name,
                            "specialization":
                                line_insert.excluded.specialization,
                            "standard_operator_capacity":
                                line_insert.excluded.standard_operator_capacity,
                            "planning_efficiency":
                                line_insert.excluded.planning_efficiency,
                            "active":
                                line_insert.excluded.active,
                            "dataset_provenance":
                                line_insert.excluded.dataset_provenance,
                        },
                    )
                )

                session.execute(
                    line_statement
                )

        return ReferenceDataSyncResult(
            factories_processed=len(
                factory_rows
            ),
            production_lines_processed=len(
                line_rows
            ),
        )

    def list_factories(
        self,
    ) -> list[FactoryRecord]:
        """Return factories ordered by factory ID."""

        statement = (
            select(Factory)
            .order_by(
                Factory.factory_id
            )
        )

        with self.session_factory() as session:
            models = session.scalars(
                statement
            ).all()

            return [
                _to_factory_record(model)
                for model in models
            ]

    def get_factory(
        self,
        factory_id: str,
    ) -> FactoryRecord | None:
        """Return one factory by natural ID."""

        statement = (
            select(Factory)
            .where(
                Factory.factory_id
                == factory_id
            )
        )

        with self.session_factory() as session:
            model = session.scalar(
                statement
            )

            if model is None:
                return None

            return _to_factory_record(
                model
            )

    def list_production_lines(
        self,
        *,
        factory_id: str | None = None,
        active: bool | None = None,
    ) -> list[ProductionLineRecord]:
        """Return production lines with optional filters."""

        statement = select(
            ProductionLine
        )

        if factory_id is not None:
            statement = statement.where(
                ProductionLine.factory_id
                == factory_id
            )

        if active is not None:
            statement = statement.where(
                ProductionLine.active
                == active
            )

        statement = statement.order_by(
            ProductionLine.line_id
        )

        with self.session_factory() as session:
            models = session.scalars(
                statement
            ).all()

            return [
                _to_production_line_record(
                    model
                )
                for model in models
            ]

    def get_production_line(
        self,
        line_id: str,
    ) -> ProductionLineRecord | None:
        """Return one production line by natural ID."""

        statement = (
            select(ProductionLine)
            .where(
                ProductionLine.line_id
                == line_id
            )
        )

        with self.session_factory() as session:
            model = session.scalar(
                statement
            )

            if model is None:
                return None

            return _to_production_line_record(
                model
            )

"""PostgreSQL persistence for LinePulse reference data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

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


class PostgresReferenceDataRepository:
    """Synchronize factory and production-line reference data."""

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

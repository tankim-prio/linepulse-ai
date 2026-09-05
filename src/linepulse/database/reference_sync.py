"""Load, validate, and synchronize LinePulse reference datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

from linepulse.database.reference_repository import (
    PostgresReferenceDataRepository,
    ReferenceDataSyncResult,
)
from linepulse.ingestion import CsvIngestionService
from linepulse.settings import DEFAULT_DATA_DIR


FACTORY_COLUMNS = (
    "factory_id",
    "factory_name",
    "country",
    "timezone",
    "weekly_closure_day",
    "dataset_provenance",
)

PRODUCTION_LINE_COLUMNS = (
    "line_id",
    "factory_id",
    "line_name",
    "specialization",
    "standard_operator_capacity",
    "planning_efficiency",
    "active",
    "dataset_provenance",
)


class ReferenceDataWriter(Protocol):
    def sync(
        self,
        factories,
        production_lines,
    ) -> ReferenceDataSyncResult:
        ...


def _parse_bool(
    value: object,
) -> bool:
    """Convert supported CSV boolean values to bool."""

    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {
        "true",
        "1",
        "yes",
    }:
        return True

    if text in {
        "false",
        "0",
        "no",
    }:
        return False

    raise ValueError(
        f"Unsupported boolean value: {value!r}"
    )


def _validate_columns(
    frame: pd.DataFrame,
    expected: tuple[str, ...],
    dataset_name: str,
) -> None:
    actual = tuple(
        str(column)
        for column in frame.columns
    )

    if actual != expected:
        raise ValueError(
            f"{dataset_name} columns do not match "
            f"the expected schema. "
            f"Expected {expected}; got {actual}."
        )


def _validate_reference_frames(
    factories: pd.DataFrame,
    production_lines: pd.DataFrame,
) -> None:
    """Validate reference-data integrity before database writes."""

    _validate_columns(
        factories,
        FACTORY_COLUMNS,
        "factories",
    )

    _validate_columns(
        production_lines,
        PRODUCTION_LINE_COLUMNS,
        "production_lines",
    )

    if factories.isna().any().any():
        raise ValueError(
            "factories contains null values."
        )

    if production_lines.isna().any().any():
        raise ValueError(
            "production_lines contains null values."
        )

    if factories[
        "factory_id"
    ].duplicated().any():
        raise ValueError(
            "factories contains duplicate factory_id values."
        )

    if production_lines[
        "line_id"
    ].duplicated().any():
        raise ValueError(
            "production_lines contains duplicate line_id values."
        )

    factory_ids = set(
        factories[
            "factory_id"
        ].astype(str)
    )

    referenced_factory_ids = set(
        production_lines[
            "factory_id"
        ].astype(str)
    )

    missing = sorted(
        referenced_factory_ids
        - factory_ids
    )

    if missing:
        raise ValueError(
            "production_lines references unknown "
            f"factory_id values: {missing}"
        )


class ReferenceDataSyncService:
    """Synchronize manifest-declared reference CSVs to PostgreSQL."""

    def __init__(
        self,
        data_dir: Path | str = DEFAULT_DATA_DIR,
        repository: ReferenceDataWriter | None = None,
    ) -> None:
        self.data_dir = Path(
            data_dir
        ).resolve()

        self.ingestion = CsvIngestionService(
            self.data_dir
        )

        self.repository = (
            repository
            or PostgresReferenceDataRepository()
        )

    def sync(
        self,
    ) -> ReferenceDataSyncResult:
        factories = pd.read_csv(
            self._dataset_path(
                "factories"
            )
        )

        production_lines = pd.read_csv(
            self._dataset_path(
                "production_lines"
            )
        )

        _validate_reference_frames(
            factories,
            production_lines,
        )

        factory_rows = [
            {
                "factory_id":
                    str(row.factory_id),
                "factory_name":
                    str(row.factory_name),
                "country":
                    str(row.country),
                "timezone":
                    str(row.timezone),
                "weekly_closure_day":
                    str(row.weekly_closure_day),
                "dataset_provenance":
                    str(row.dataset_provenance),
            }
            for row in factories.itertuples(
                index=False
            )
        ]

        line_rows = [
            {
                "line_id":
                    str(row.line_id),
                "factory_id":
                    str(row.factory_id),
                "line_name":
                    str(row.line_name),
                "specialization":
                    str(row.specialization),
                "standard_operator_capacity":
                    int(
                        row.standard_operator_capacity
                    ),
                "planning_efficiency":
                    float(
                        row.planning_efficiency
                    ),
                "active":
                    _parse_bool(
                        row.active
                    ),
                "dataset_provenance":
                    str(row.dataset_provenance),
            }
            for row in production_lines.itertuples(
                index=False
            )
        ]

        return self.repository.sync(
            factory_rows,
            line_rows,
        )

    def _dataset_path(
        self,
        dataset_name: str,
    ) -> Path:
        rows = self.ingestion.manifest[
            self.ingestion.manifest[
                "dataset_name"
            ].astype(str)
            == dataset_name
        ]

        if rows.empty:
            raise KeyError(
                "Required reference dataset "
                f"is not declared: {dataset_name}"
            )

        path = (
            self.data_dir
            / str(
                rows.iloc[0][
                    "relative_path"
                ]
            )
        ).resolve()

        if not path.is_file():
            raise FileNotFoundError(
                "Declared reference dataset "
                f"file missing: {path}"
            )

        return path

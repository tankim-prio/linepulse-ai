"""Validate declared LinePulse CSV files and stage them safely."""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
from pandas.errors import ParserError

from linepulse.settings import DEFAULT_DATA_DIR


_SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class IngestionIssue:
    code: str
    message: str
    column: str | None = None
    rows: tuple[int, ...] = ()


@dataclass(frozen=True)
class IngestionResult:
    dataset_name: str
    accepted: bool
    row_count: int
    column_count: int
    source_sha256: str | None
    issues: tuple[IngestionIssue, ...]
    staged_path: str | None = None
    already_staged: bool = False


class CsvIngestionService:
    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self.data_dir = Path(data_dir).resolve()
        metadata = self.data_dir / "metadata"

        self.manifest = pd.read_csv(
            metadata / "dataset_manifest.csv",
            encoding="utf-8-sig",
        )
        self.dictionary = pd.read_csv(
            metadata / "data_dictionary.csv",
            encoding="utf-8-sig",
        )

    def declared_datasets(self) -> tuple[str, ...]:
        return tuple(
            self.manifest["dataset_name"]
            .dropna()
            .astype(str)
            .tolist()
        )

    def validate_csv(
        self,
        dataset_name: str,
        source_path: Path | str,
    ) -> IngestionResult:

        source = Path(source_path)

        if dataset_name not in self.declared_datasets():
            return self._reject(
                dataset_name,
                "undeclared_dataset",
                f"Dataset '{dataset_name}' is not declared.",
            )

        if not _SAFE_NAME.fullmatch(dataset_name):
            return self._reject(
                dataset_name,
                "unsafe_dataset_name",
                "Dataset name is unsafe.",
            )

        if source.suffix.lower() != ".csv":
            return self._reject(
                dataset_name,
                "invalid_file_type",
                "Only CSV files are accepted.",
            )

        if not source.is_file():
            return self._reject(
                dataset_name,
                "missing_source_file",
                f"File does not exist: {source}",
            )

        payload = source.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()

        try:
            frame = pd.read_csv(
                io.BytesIO(payload),
                encoding="utf-8-sig",
            )
        except (UnicodeDecodeError, ParserError, ValueError) as exc:
            return IngestionResult(
                dataset_name,
                False,
                0,
                0,
                digest,
                (
                    IngestionIssue(
                        "csv_parse_error",
                        f"CSV could not be parsed: {exc}",
                    ),
                ),
            )

        issues = self._validate_frame(dataset_name, frame)

        return IngestionResult(
            dataset_name=dataset_name,
            accepted=not issues,
            row_count=len(frame),
            column_count=len(frame.columns),
            source_sha256=digest,
            issues=tuple(issues),
        )

    def validate_and_stage(
        self,
        dataset_name: str,
        source_path: Path | str,
        staging_dir: Path | str,
    ) -> IngestionResult:

        result = self.validate_csv(dataset_name, source_path)

        if not result.accepted:
            return result

        source = Path(source_path)
        payload = source.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()

        if digest != result.source_sha256:
            return replace(
                result,
                accepted=False,
                issues=(
                    IngestionIssue(
                        "source_changed",
                        "Source changed after validation.",
                    ),
                ),
            )

        root = Path(staging_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)

        destination = (
            root / f"{dataset_name}-{digest}.csv"
        ).resolve()

        if destination.parent != root:
            return replace(
                result,
                accepted=False,
                issues=(
                    IngestionIssue(
                        "unsafe_staging_path",
                        "Staging path escaped staging directory.",
                    ),
                ),
            )

        if destination.exists():
            return replace(
                result,
                staged_path=str(destination),
                already_staged=True,
            )

        with destination.open("xb") as handle:
            handle.write(payload)

        return replace(
            result,
            staged_path=str(destination),
        )

    def _validate_frame(
        self,
        dataset_name: str,
        frame: pd.DataFrame,
    ) -> list[IngestionIssue]:

        issues: list[IngestionIssue] = []

        contract = self.dictionary[
            self.dictionary["dataset_name"] == dataset_name
        ]

        expected = contract["column_name"].astype(str).tolist()

        missing = [c for c in expected if c not in frame.columns]
        extra = [c for c in frame.columns if c not in expected]

        if missing:
            issues.append(
                IngestionIssue(
                    "missing_columns",
                    f"Missing columns: {missing}",
                )
            )

        if extra:
            issues.append(
                IngestionIssue(
                    "unexpected_columns",
                    f"Unexpected columns: {extra}",
                )
            )

        required = contract[
            contract["required"]
            .astype(str)
            .str.lower()
            .eq("true")
        ]

        for column in required["column_name"].astype(str):
            if column not in frame.columns:
                continue

            blank = frame[column].isna()

            if frame[column].dtype == object:
                blank = blank | (
                    frame[column]
                    .astype("string")
                    .str.strip()
                    .eq("")
                    .fillna(False)
                )

            if blank.any():
                rows = tuple(
                    int(i) + 2
                    for i in frame.index[blank]
                )
                issues.append(
                    IngestionIssue(
                        "required_value_missing",
                        f"Required value missing in '{column}'.",
                        column,
                        rows,
                    )
                )

        issues.extend(
            self._check_numeric_types(contract, frame)
        )
        issues.extend(
            self._check_primary_key(dataset_name, frame)
        )

        return issues

    def _check_numeric_types(
        self,
        contract: pd.DataFrame,
        frame: pd.DataFrame,
    ) -> list[IngestionIssue]:

        issues = []

        for row in contract.itertuples(index=False):
            column = str(row.column_name)
            dtype = str(row.data_type).lower()

            if column not in frame.columns:
                continue

            if not (
                dtype.startswith("int")
                or dtype.startswith("float")
            ):
                continue

            values = frame[column]
            converted = pd.to_numeric(
                values,
                errors="coerce",
            )

            invalid = values.notna() & converted.isna()

            if invalid.any():
                rows = tuple(
                    int(i) + 2
                    for i in frame.index[invalid]
                )
                issues.append(
                    IngestionIssue(
                        "invalid_numeric_value",
                        f"Invalid numeric value in '{column}'.",
                        column,
                        rows,
                    )
                )

        return issues

    def _check_primary_key(
        self,
        dataset_name: str,
        frame: pd.DataFrame,
    ) -> list[IngestionIssue]:

        row = self.manifest[
            self.manifest["dataset_name"] == dataset_name
        ].iloc[0]

        raw = str(row["primary_key"]).strip()

        if not raw or raw.lower() == "nan":
            return []

        keys = [
            value.strip()
            for value in re.split(r"[,|;+]", raw)
            if value.strip()
        ]

        if any(key not in frame.columns for key in keys):
            return []

        issues = []

        null_mask = frame[keys].isna().any(axis=1)

        if null_mask.any():
            issues.append(
                IngestionIssue(
                    "primary_key_null",
                    f"Primary key {keys} contains null values.",
                )
            )

        duplicate = frame.duplicated(
            subset=keys,
            keep=False,
        )

        if duplicate.any():
            rows = tuple(
                int(i) + 2
                for i in frame.index[duplicate]
            )
            issues.append(
                IngestionIssue(
                    "duplicate_primary_key",
                    f"Primary key {keys} contains duplicates.",
                    rows=rows,
                )
            )

        return issues

    @staticmethod
    def _reject(
        dataset_name: str,
        code: str,
        message: str,
    ) -> IngestionResult:
        return IngestionResult(
            dataset_name=dataset_name,
            accepted=False,
            row_count=0,
            column_count=0,
            source_sha256=None,
            issues=(IngestionIssue(code, message),),
        )

"""Validation pipeline for the LinePulse synthetic data package.

The validator is linear in the number of rows plus foreign-key values: O(n).
It keeps each CSV in memory once, so auxiliary memory is O(n). The bundled
25k-row dataset is comfortably within the target 8 GB machine.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from linepulse.data.contracts import (
    ALLOWED_SPLITS,
    DAILY_RISK_FEATURES,
    DAILY_RISK_TARGET,
    FOREIGN_KEYS,
    FORBIDDEN_DAILY_RISK_FEATURES,
    FORBIDDEN_ORDER_DELAY_FEATURES,
    ORDER_DELAY_FEATURES,
    ORDER_DELAY_TARGET,
    REQUIRED_NOTE_LANGUAGES,
)


BENGALI_PATTERN = re.compile(r"[\u0980-\u09FF]")


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str
    severity: str = "error"


@dataclass(frozen=True)
class ManifestEntry:
    dataset_name: str
    relative_path: str
    row_count: int
    column_count: int
    primary_key: str


@dataclass
class ValidationReport:
    data_dir: str
    generated_at: str
    dataset_count: int
    row_count: int
    checks: list[CheckResult]

    @property
    def passed(self) -> bool:
        return all(
            check.passed
            for check in self.checks
            if check.severity == "error"
        )

    @property
    def failed_count(self) -> int:
        return sum(
            not check.passed
            for check in self.checks
            if check.severity == "error"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "data_dir": self.data_dir,
            "generated_at": self.generated_at,
            "passed": self.passed,
            "dataset_count": self.dataset_count,
            "row_count": self.row_count,
            "check_count": len(self.checks),
            "failed_count": self.failed_count,
            "checks": [asdict(check) for check in self.checks],
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


class DataValidator:
    """Run structural, relational, temporal, and safety checks."""

    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.results: list[CheckResult] = []
        self.frames: dict[str, pd.DataFrame] = {}
        self.entries: list[ManifestEntry] = []

    def run(self) -> ValidationReport:
        self.results = []
        self.frames = {}
        self.entries = []

        manifest = self._read_manifest()
        if manifest is not None:
            try:
                self.entries = self._parse_manifest(manifest)
            except (TypeError, ValueError) as exc:
                self._add(
                    "manifest:values",
                    False,
                    f"Manifest contains invalid values: {exc}",
                )
                return self._build_report()
            self._validate_manifest_names()
            self._load_declared_datasets()
            self._validate_shapes()
            self._validate_dictionary()
            self._validate_primary_keys()
            self._validate_foreign_keys()
            self._validate_provenance()
            self._validate_business_rules()
            self._validate_time_splits()
            self._validate_languages()
            self._validate_feature_contracts()
            self._validate_evaluation_safety()

        return self._build_report()

    def _build_report(self) -> ValidationReport:
        return ValidationReport(
            data_dir=str(self.data_dir),
            generated_at=datetime.now(timezone.utc).isoformat(),
            dataset_count=len(self.frames),
            row_count=sum(len(frame) for frame in self.frames.values()),
            checks=self.results.copy(),
        )

    def _add(
        self,
        name: str,
        passed: bool,
        detail: str,
        severity: str = "error",
    ) -> None:
        self.results.append(CheckResult(name, bool(passed), detail, severity))

    def _read_manifest(self) -> pd.DataFrame | None:
        path = self.data_dir / "metadata" / "dataset_manifest.csv"
        if not path.is_file():
            self._add(
                "manifest:exists",
                False,
                f"Missing manifest: {path}",
            )
            return None
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig")
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            self._add("manifest:readable", False, f"Cannot read manifest: {exc}")
            return None
        required = {
            "dataset_name",
            "relative_path",
            "row_count",
            "column_count",
            "primary_key",
        }
        missing = sorted(required - set(frame.columns))
        self._add(
            "manifest:schema",
            not missing,
            "Manifest schema is valid."
            if not missing
            else f"Missing manifest columns: {missing}",
        )
        return frame if not missing else None

    @staticmethod
    def _parse_manifest(frame: pd.DataFrame) -> list[ManifestEntry]:
        entries: list[ManifestEntry] = []
        for row in frame.fillna("").itertuples(index=False):
            entries.append(
                ManifestEntry(
                    dataset_name=str(row.dataset_name),
                    relative_path=str(row.relative_path),
                    row_count=int(row.row_count),
                    column_count=int(row.column_count),
                    primary_key=str(row.primary_key),
                )
            )
        return entries

    def _validate_manifest_names(self) -> None:
        names = [entry.dataset_name for entry in self.entries]
        duplicates = sorted(
            name for name in set(names) if names.count(name) > 1
        )
        self._add(
            "manifest:unique_dataset_names",
            not duplicates,
            "Dataset names are unique."
            if not duplicates
            else f"Duplicate dataset names: {duplicates}",
        )

    def _safe_dataset_path(self, relative_path: str) -> Path | None:
        candidate = (self.data_dir / relative_path).resolve()
        try:
            candidate.relative_to(self.data_dir)
        except ValueError:
            return None
        return candidate

    def _load_declared_datasets(self) -> None:
        for entry in self.entries:
            path = self._safe_dataset_path(entry.relative_path)
            if path is None:
                self._add(
                    f"inventory:{entry.dataset_name}",
                    False,
                    f"Unsafe path outside data directory: {entry.relative_path}",
                )
                continue
            if not path.is_file():
                self._add(
                    f"inventory:{entry.dataset_name}",
                    False,
                    f"Missing declared file: {entry.relative_path}",
                )
                continue
            try:
                frame = pd.read_csv(path, encoding="utf-8-sig")
            except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                self._add(
                    f"inventory:{entry.dataset_name}",
                    False,
                    f"Cannot read {entry.relative_path}: {exc}",
                )
                continue
            self.frames[entry.dataset_name] = frame
            self._add(
                f"inventory:{entry.dataset_name}",
                True,
                f"Loaded {entry.relative_path}.",
            )

    def _validate_shapes(self) -> None:
        for entry in self.entries:
            frame = self.frames.get(entry.dataset_name)
            if frame is None:
                continue
            actual = (len(frame), len(frame.columns))
            expected = (entry.row_count, entry.column_count)
            self._add(
                f"shape:{entry.dataset_name}",
                actual == expected,
                f"Expected {expected[0]}x{expected[1]}; found "
                f"{actual[0]}x{actual[1]}.",
            )

    def _validate_dictionary(self) -> None:
        dictionary = self.frames.get("data_dictionary")
        if dictionary is None:
            path = self.data_dir / "metadata" / "data_dictionary.csv"
            if not path.is_file():
                self._add("dictionary:exists", False, "Data dictionary is missing.")
                return
            try:
                dictionary = pd.read_csv(path, encoding="utf-8-sig")
            except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                self._add("dictionary:readable", False, str(exc))
                return

        required_dictionary_columns = {
            "dataset_name",
            "column_name",
            "required",
        }
        missing = required_dictionary_columns - set(dictionary.columns)
        self._add(
            "dictionary:schema",
            not missing,
            "Data dictionary schema is valid."
            if not missing
            else f"Missing dictionary columns: {sorted(missing)}",
        )
        if missing:
            return

        for entry in self.entries:
            frame = self.frames.get(entry.dataset_name)
            if frame is None:
                continue
            rows = dictionary[dictionary["dataset_name"] == entry.dataset_name]
            required_mask = rows["required"].astype(str).str.lower().isin(
                {"true", "1", "yes"}
            )
            required_columns = set(rows.loc[required_mask, "column_name"])
            missing_columns = sorted(required_columns - set(frame.columns))
            self._add(
                f"required_columns:{entry.dataset_name}",
                not missing_columns,
                "All required columns exist."
                if not missing_columns
                else f"Missing required columns: {missing_columns}",
            )
            if missing_columns:
                continue
            null_counts = {
                column: int(frame[column].isna().sum())
                for column in required_columns
                if frame[column].isna().any()
            }
            self._add(
                f"required_values:{entry.dataset_name}",
                not null_counts,
                "Required columns contain no nulls."
                if not null_counts
                else f"Null values in required columns: {null_counts}",
            )

    @staticmethod
    def _primary_key_columns(value: str) -> list[str]:
        return [part.strip() for part in value.split("+") if part.strip()]

    def _validate_primary_keys(self) -> None:
        for entry in self.entries:
            frame = self.frames.get(entry.dataset_name)
            if frame is None:
                continue
            columns = self._primary_key_columns(entry.primary_key)
            missing = [column for column in columns if column not in frame.columns]
            if missing:
                self._add(
                    f"primary_key:{entry.dataset_name}",
                    False,
                    f"Primary-key columns missing: {missing}",
                )
                continue
            null_rows = int(frame[columns].isna().any(axis=1).sum())
            duplicate_rows = int(frame.duplicated(columns, keep=False).sum())
            passed = null_rows == 0 and duplicate_rows == 0
            self._add(
                f"primary_key:{entry.dataset_name}",
                passed,
                f"Columns={columns}; null rows={null_rows}; "
                f"duplicate rows={duplicate_rows}.",
            )

    def _validate_foreign_keys(self) -> None:
        for child_name, relations in FOREIGN_KEYS.items():
            child = self.frames.get(child_name)
            if child is None:
                continue
            for child_columns, parent_name, parent_columns in relations:
                parent = self.frames.get(parent_name)
                check_name = (
                    f"foreign_key:{child_name}."
                    f"{'+'.join(child_columns)}->{parent_name}."
                    f"{'+'.join(parent_columns)}"
                )
                if parent is None:
                    self._add(check_name, False, "Parent dataset is unavailable.")
                    continue
                missing_child = [
                    column
                    for column in child_columns
                    if column not in child.columns
                ]
                missing_parent = [
                    column
                    for column in parent_columns
                    if column not in parent.columns
                ]
                if missing_child or missing_parent:
                    self._add(
                        check_name,
                        False,
                        "Foreign-key columns missing: "
                        f"child={missing_child}, parent={missing_parent}",
                    )
                    continue
                child_keys = set(
                    map(
                        tuple,
                        child[list(child_columns)]
                        .dropna()
                        .astype(str)
                        .itertuples(index=False, name=None),
                    )
                )
                parent_keys = set(
                    map(
                        tuple,
                        parent[list(parent_columns)]
                        .dropna()
                        .astype(str)
                        .itertuples(index=False, name=None),
                    )
                )
                unresolved = child_keys - parent_keys
                self._add(
                    check_name,
                    not unresolved,
                    "All non-null foreign keys resolve."
                    if not unresolved
                    else f"Unresolved key count: {len(unresolved)}.",
                )

    def _validate_provenance(self) -> None:
        for dataset_name, frame in self.frames.items():
            if "dataset_provenance" not in frame.columns:
                continue
            values = set(frame["dataset_provenance"].dropna().astype(str))
            self._add(
                f"provenance:{dataset_name}",
                values == {"synthetic"},
                f"Observed values: {sorted(values)}.",
            )

    def _nonnegative(
        self,
        dataset_name: str,
        columns: Iterable[str],
    ) -> None:
        frame = self.frames.get(dataset_name)
        if frame is None:
            return
        available = [column for column in columns if column in frame.columns]
        if not available:
            self._add(
                f"business:nonnegative:{dataset_name}",
                False,
                "No expected numeric columns are available.",
            )
            return
        numeric = frame[available].apply(pd.to_numeric, errors="coerce")
        invalid = int((numeric < 0).sum().sum())
        self._add(
            f"business:nonnegative:{dataset_name}",
            invalid == 0,
            f"Negative value count across {available}: {invalid}.",
        )

    def _validate_business_rules(self) -> None:
        self._nonnegative(
            "hourly_production",
            [
                "hourly_target",
                "actual_output",
                "wip_qty",
                "downtime_minutes",
                "inspected_qty",
                "defect_count",
            ],
        )
        self._nonnegative(
            "machine_downtime",
            ["duration_minutes"],
        )
        self._nonnegative(
            "quality_defects",
            ["defect_count", "inspected_qty"],
        )
        self._nonnegative(
            "material_status",
            ["required_qty", "available_qty"],
        )

        attendance = self.frames.get("attendance_summaries")
        if attendance is not None:
            valid = (
                (attendance["present_workers"] <= attendance["planned_workers"])
                & (
                    attendance["skilled_present_workers"]
                    <= attendance["present_workers"]
                )
                & (attendance["planned_workers"] > 0)
            )
            self._add(
                "business:attendance_hierarchy",
                bool(valid.all()),
                f"Invalid attendance rows: {int((~valid).sum())}.",
            )

        self._validate_binary_label("daily_line_ml", DAILY_RISK_TARGET)
        self._validate_binary_label("order_outcomes", ORDER_DELAY_TARGET)

    def _validate_binary_label(self, dataset_name: str, column: str) -> None:
        frame = self.frames.get(dataset_name)
        if frame is None or column not in frame.columns:
            self._add(
                f"label:{dataset_name}.{column}",
                False,
                "Label column is unavailable.",
            )
            return
        values = set(pd.to_numeric(frame[column], errors="coerce").dropna())
        self._add(
            f"label:{dataset_name}.{column}",
            values == {0, 1},
            f"Observed label values: {sorted(values)}.",
        )

    @staticmethod
    def _expected_split(timestamp: pd.Timestamp) -> str:
        day = timestamp.date()
        if day <= datetime(2025, 4, 30).date():
            return "train"
        if day <= datetime(2025, 5, 31).date():
            return "validation"
        return "test"

    def _validate_time_splits(self) -> None:
        split_table = self.frames.get("split_assignments")
        if split_table is None:
            self._add("split:assignments_available", False, "Missing split table.")
            return
        dates = pd.to_datetime(split_table["work_date"], errors="coerce")
        expected = dates.map(
            lambda value: self._expected_split(value)
            if pd.notna(value)
            else "invalid"
        )
        actual = split_table["data_split"].astype(str)
        valid = dates.notna() & actual.isin(ALLOWED_SPLITS) & (actual == expected)
        self._add(
            "split:chronology",
            bool(valid.all()),
            f"Invalid split rows: {int((~valid).sum())}.",
        )

        split_map = dict(zip(split_table["work_date"].astype(str), actual))
        for dataset_name, frame in self.frames.items():
            if dataset_name == "split_assignments":
                continue
            if not {"work_date", "data_split"}.issubset(frame.columns):
                continue
            expected_split = frame["work_date"].astype(str).map(split_map)
            comparable = expected_split.notna()
            mismatches = int(
                (
                    frame.loc[comparable, "data_split"].astype(str)
                    != expected_split[comparable]
                ).sum()
            )
            self._add(
                f"split:work_date_alignment:{dataset_name}",
                mismatches == 0,
                f"Mismatched rows: {mismatches}.",
            )

        daily = self.frames.get("daily_line_ml")
        if daily is not None:
            snapshot = pd.to_datetime(
                daily["snapshot_at"],
                errors="coerce",
                utc=True,
            )
            snapshot_valid = snapshot.notna() & (snapshot.dt.hour == 13)
            self._add(
                "snapshot:daily_risk_time",
                bool(snapshot_valid.all()),
                f"Invalid 13:00 UTC snapshots: {int((~snapshot_valid).sum())}.",
            )

    def _validate_languages(self) -> None:
        notes = self.frames.get("supervisor_notes")
        if notes is not None:
            languages = set(notes["language"].dropna().astype(str))
            self._add(
                "language:note_coverage",
                languages == REQUIRED_NOTE_LANGUAGES,
                f"Observed languages: {sorted(languages)}.",
            )
            expected_bengali = notes["language"].isin({"bn", "mixed"})
            has_bengali = notes["note_text"].fillna("").astype(str).map(
                lambda value: bool(BENGALI_PATTERN.search(value))
            )
            invalid = int((expected_bengali & ~has_bengali).sum())
            self._add(
                "language:bangla_notes",
                invalid == 0,
                f"Bangla/mixed notes without Bengali Unicode: {invalid}.",
            )

        sections = self.frames.get("document_sections")
        if sections is not None:
            has_bengali = sections["bangla_text"].fillna("").astype(str).map(
                lambda value: bool(BENGALI_PATTERN.search(value))
            )
            self._add(
                "language:bangla_document_sections",
                bool(has_bengali.all()),
                f"Sections without Bengali Unicode: {int((~has_bengali).sum())}.",
            )

    def _validate_feature_contracts(self) -> None:
        daily = self.frames.get("daily_line_ml")
        if daily is not None:
            required = set(DAILY_RISK_FEATURES) | {DAILY_RISK_TARGET}
            missing = sorted(required - set(daily.columns))
            leaked = sorted(
                set(DAILY_RISK_FEATURES) & FORBIDDEN_DAILY_RISK_FEATURES
            )
            self._add(
                "feature_contract:daily_risk",
                not missing and not leaked,
                f"Missing={missing}; forbidden features selected={leaked}.",
            )

        orders = self.frames.get("order_outcomes")
        if orders is not None:
            required = set(ORDER_DELAY_FEATURES) | {ORDER_DELAY_TARGET}
            missing = sorted(required - set(orders.columns))
            leaked = sorted(
                set(ORDER_DELAY_FEATURES) & FORBIDDEN_ORDER_DELAY_FEATURES
            )
            self._add(
                "feature_contract:order_delay",
                not missing and not leaked,
                f"Missing={missing}; forbidden features selected={leaked}.",
            )

    def _validate_evaluation_safety(self) -> None:
        rag = self.frames.get("rag_gold_questions")
        if rag is not None:
            expected_status = rag["answerable"].map(
                {True: "supported", False: "insufficient_evidence"}
            )
            valid = expected_status == rag["expected_status"]
            self._add(
                "evaluation:rag_answerability",
                bool(valid.all()),
                f"Mismatched RAG outcomes: {int((~valid).sum())}.",
            )

        agent = self.frames.get("agent_eval_cases")
        if agent is not None:
            should_execute = agent["expected_should_execute"].astype(bool)
            execute_state = agent["expected_graph_outcome"] == "execute_action"
            valid = should_execute == execute_state
            self._add(
                "evaluation:agent_execution_gate",
                bool(valid.all()),
                f"Inconsistent agent execution cases: {int((~valid).sum())}.",
            )

        security = self.frames.get("security_eval_cases")
        if security is not None:
            denied = security["expected_decision"].astype(str).str.startswith(
                ("deny", "block")
            )
            writes = pd.to_numeric(
                security["expected_write_count"],
                errors="coerce",
            )
            invalid = denied & (writes != 0)
            self._add(
                "evaluation:security_write_gate",
                not bool(invalid.any()),
                f"Denied cases expecting writes: {int(invalid.sum())}.",
            )

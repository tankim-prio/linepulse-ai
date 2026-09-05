"""Run the LinePulse vertical slice end to end."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from linepulse.database.risk_repository import (
    PostgresRiskEventRepository,
)
from linepulse.features import build_daily_snapshot_features
from linepulse.ingestion import (
    CsvIngestionService,
    IngestionIssue,
)
from linepulse.publishing import DatasetPublisher
from linepulse.risk import (
    RiskEvent,
    RiskEventStore,
    build_risk_event,
    evaluate_rule_risk,
)
from linepulse.settings import DEFAULT_DATA_DIR


class RiskEventWriter(Protocol):
    """Persistence interface required by the workflow."""

    def append(self, event: RiskEvent) -> bool:
        ...


RiskEventStoreFactory = Callable[[Path], RiskEventWriter]


def _postgres_store_factory(
    _: Path,
) -> RiskEventWriter:
    """Create the default durable PostgreSQL risk-event store."""

    return PostgresRiskEventRepository()


@dataclass(frozen=True)
class VerticalSliceResult:
    accepted: bool
    dataset_name: str
    validation_issues: tuple[IngestionIssue, ...]
    staged_path: str | None = None
    published_path: str | None = None
    snapshot_count: int = 0
    events_created: int = 0
    events_reused: int = 0
    risk_event_path: str | None = None


class VerticalSliceWorkflow:
    def __init__(
        self,
        data_dir: Path | str = DEFAULT_DATA_DIR,
        risk_event_store_factory: RiskEventStoreFactory | None = None,
    ) -> None:
        self.data_dir = Path(data_dir).resolve()

        self.ingestion = CsvIngestionService(
            self.data_dir
        )

        self.publisher = DatasetPublisher()

        self.risk_event_store_factory = (
            risk_event_store_factory
            or _postgres_store_factory
        )

    def run_hourly_production_csv(
        self,
        source_path: Path | str,
        work_dir: Path | str,
    ) -> VerticalSliceResult:
        dataset_name = "hourly_production"

        work = Path(work_dir).resolve()
        work.mkdir(
            parents=True,
            exist_ok=True,
        )

        ingestion_result = (
            self.ingestion.validate_and_stage(
                dataset_name,
                source_path,
                work / "staging",
            )
        )

        if not ingestion_result.accepted:
            return VerticalSliceResult(
                accepted=False,
                dataset_name=dataset_name,
                validation_issues=ingestion_result.issues,
            )

        publish_result = self.publisher.publish(
            dataset_name,
            ingestion_result.staged_path,
            work / "published",
        )

        hourly = pd.read_csv(
            publish_result.published_path
        )

        attendance = pd.read_csv(
            self._dataset_path(
                "attendance_summaries",
                "attendance",
            )
        )

        notes = pd.read_csv(
            self._dataset_path(
                "supervisor_notes"
            )
        )

        styles = pd.read_csv(
            self._dataset_path(
                "styles"
            )
        )

        snapshots = build_daily_snapshot_features(
            hourly_production=hourly,
            attendance_summaries=attendance,
            supervisor_notes=notes,
            styles=styles,
        )

        jsonl_path = work / "risk_events.jsonl"

        store = self.risk_event_store_factory(
            jsonl_path
        )

        created = 0
        reused = 0

        for snapshot in snapshots.to_dict(
            orient="records"
        ):
            assessment = evaluate_rule_risk(
                snapshot
            )

            event = build_risk_event(
                snapshot,
                assessment,
            )

            if store.append(event):
                created += 1
            else:
                reused += 1

        risk_event_path = (
            str(jsonl_path)
            if isinstance(store, RiskEventStore)
            else None
        )

        return VerticalSliceResult(
            accepted=True,
            dataset_name=dataset_name,
            validation_issues=(),
            staged_path=ingestion_result.staged_path,
            published_path=publish_result.published_path,
            snapshot_count=len(snapshots),
            events_created=created,
            events_reused=reused,
            risk_event_path=risk_event_path,
        )

    def _dataset_path(
        self,
        *dataset_names: str,
    ) -> Path:
        for dataset_name in dataset_names:
            rows = self.ingestion.manifest[
                self.ingestion.manifest[
                    "dataset_name"
                ].astype(str)
                == dataset_name
            ]

            if rows.empty:
                continue

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
                    f"Declared dataset file missing: {path}"
                )

            return path

        raise KeyError(
            "None of the required datasets "
            f"are declared: {dataset_names}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the LinePulse "
            "hourly-production vertical slice."
        )
    )

    parser.add_argument(
        "--source",
        required=True,
    )

    parser.add_argument(
        "--work-dir",
        required=True,
    )

    args = parser.parse_args()

    result = VerticalSliceWorkflow().run_hourly_production_csv(
        args.source,
        args.work_dir,
    )

    print(
        json.dumps(
            asdict(result),
            indent=2,
            ensure_ascii=False,
        )
    )

    return 0 if result.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())

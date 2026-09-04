"""Run the first LinePulse vertical slice end to end."""

from __future__ import annotations
import argparse, json
from dataclasses import asdict, dataclass
from pathlib import Path
import pandas as pd
from linepulse.features import build_daily_snapshot_features
from linepulse.ingestion import CsvIngestionService, IngestionIssue
from linepulse.publishing import DatasetPublisher
from linepulse.risk import RiskEventStore, build_risk_event, evaluate_rule_risk
from linepulse.settings import DEFAULT_DATA_DIR

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
    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.ingestion = CsvIngestionService(self.data_dir)
        self.publisher = DatasetPublisher()

    def run_hourly_production_csv(self, source_path: Path | str, work_dir: Path | str) -> VerticalSliceResult:
        dataset_name = "hourly_production"
        work = Path(work_dir).resolve()
        work.mkdir(parents=True, exist_ok=True)
        ingestion_result = self.ingestion.validate_and_stage(dataset_name, source_path, work / "staging")
        if not ingestion_result.accepted:
            return VerticalSliceResult(False, dataset_name, ingestion_result.issues)
        publish_result = self.publisher.publish(dataset_name, ingestion_result.staged_path, work / "published")
        hourly = pd.read_csv(publish_result.published_path)
        attendance = pd.read_csv(self._dataset_path("attendance_summaries", "attendance"))
        notes = pd.read_csv(self._dataset_path("supervisor_notes"))
        styles = pd.read_csv(self._dataset_path("styles"))
        snapshots = build_daily_snapshot_features(
            hourly_production=hourly,
            attendance_summaries=attendance,
            supervisor_notes=notes,
            styles=styles,
        )
        event_path = work / "risk_events.jsonl"
        store = RiskEventStore(event_path)
        created = reused = 0
        for snapshot in snapshots.to_dict(orient="records"):
            event = build_risk_event(snapshot, evaluate_rule_risk(snapshot))
            if store.append(event):
                created += 1
            else:
                reused += 1
        return VerticalSliceResult(
            True, dataset_name, (),
            ingestion_result.staged_path,
            publish_result.published_path,
            len(snapshots), created, reused, str(event_path),
        )

    def _dataset_path(self, *dataset_names: str) -> Path:
        for dataset_name in dataset_names:
            rows = self.ingestion.manifest[
                self.ingestion.manifest["dataset_name"].astype(str) == dataset_name
            ]
            if rows.empty:
                continue
            path = (self.data_dir / str(rows.iloc[0]["relative_path"])).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Declared dataset file missing: {path}")
            return path
        raise KeyError(f"None of the required datasets are declared: {dataset_names}")

def main() -> int:
    parser = argparse.ArgumentParser(description="Run the LinePulse hourly-production vertical slice.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()
    result = VerticalSliceWorkflow().run_hourly_production_csv(args.source, args.work_dir)
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    return 0 if result.accepted else 2

if __name__ == "__main__":
    raise SystemExit(main())

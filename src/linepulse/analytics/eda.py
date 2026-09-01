"""Generate a reproducible exploratory report for daily production risk.

Runtime is O(n * p) for n rows and p selected columns. In-memory usage is
O(n * p), which is small for the bundled 1,224-row modeling table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from linepulse.data.contracts import (
    DAILY_RISK_FEATURES,
    DAILY_RISK_TARGET,
)
from linepulse.settings import DEFAULT_DATA_DIR, PROJECT_ROOT


DEFAULT_INPUT_PATH = DEFAULT_DATA_DIR / "modeling" / "daily_line_ml.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "eda"
IDENTIFIER_COLUMNS = (
    "snapshot_id",
    "factory_id",
    "work_date",
    "snapshot_at",
    "line_id",
    "order_id",
    "style_id",
    "data_split",
)
CATEGORICAL_FEATURES = (
    "complexity_band",
    "material_status",
    "root_cause_signal",
)


def load_daily_risk_data(path: Path | str) -> pd.DataFrame:
    """Load the modeling table and enforce its analytical schema."""

    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Modeling dataset not found: {input_path}")

    frame = pd.read_csv(input_path, encoding="utf-8-sig")
    required = (
        set(IDENTIFIER_COLUMNS)
        | set(DAILY_RISK_FEATURES)
        | {DAILY_RISK_TARGET, "dataset_provenance"}
    )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required modeling columns: {missing}")
    if frame.empty:
        raise ValueError("The modeling dataset is empty.")

    frame = frame.copy()
    frame["work_date"] = pd.to_datetime(
        frame["work_date"],
        errors="raise",
    )
    frame["snapshot_at"] = pd.to_datetime(
        frame["snapshot_at"],
        errors="raise",
        utc=True,
    )
    return frame


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _save_bar_chart(
    values: pd.Series,
    title: str,
    x_label: str,
    y_label: str,
    path: Path,
    color: str = "#0F766E",
) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    values.plot(kind="bar", ax=axis, color=color)
    axis.set_title(title, fontweight="bold")
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_eda_report(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    """Create tables, charts, and a machine-readable EDA summary."""

    frame = load_daily_risk_data(input_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    target = pd.to_numeric(frame[DAILY_RISK_TARGET], errors="raise").astype(int)
    label_counts = target.value_counts().sort_index()
    label_rates = target.value_counts(normalize=True).sort_index()
    split_counts = frame["data_split"].value_counts().reindex(
        ["train", "validation", "test"],
        fill_value=0,
    )
    split_miss_rate = frame.groupby("data_split", observed=True)[
        DAILY_RISK_TARGET
    ].mean().reindex(["train", "validation", "test"])

    summary = {
        "dataset": str(Path(input_path)),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "date_start": frame["work_date"].min().date().isoformat(),
        "date_end": frame["work_date"].max().date().isoformat(),
        "duplicate_snapshot_ids": int(frame["snapshot_id"].duplicated().sum()),
        "missing_cells": int(frame.isna().sum().sum()),
        "target": DAILY_RISK_TARGET,
        "target_counts": {
            str(int(label)): int(count)
            for label, count in label_counts.items()
        },
        "target_rates": {
            str(int(label)): round(float(rate), 6)
            for label, rate in label_rates.items()
        },
        "split_counts": {
            str(split): int(count)
            for split, count in split_counts.items()
        },
        "split_target_rates": {
            str(split): round(float(rate), 6)
            for split, rate in split_miss_rate.dropna().items()
        },
        "synthetic_only": bool(
            (frame["dataset_provenance"] == "synthetic").all()
        ),
    }
    _write_json(destination / "dataset_summary.json", summary)

    missing = pd.DataFrame(
        {
            "column": frame.columns,
            "missing_count": frame.isna().sum().values,
            "missing_rate": frame.isna().mean().values,
        }
    ).sort_values(["missing_count", "column"], ascending=[False, True])
    missing.to_csv(destination / "missing_values.csv", index=False)

    numeric_columns = frame[list(DAILY_RISK_FEATURES)].select_dtypes(
        include="number"
    ).columns
    numeric_summary = frame[numeric_columns].describe().T.reset_index()
    numeric_summary = numeric_summary.rename(columns={"index": "feature"})
    numeric_summary.to_csv(destination / "numeric_summary.csv", index=False)

    categorical_rows: list[dict[str, object]] = []
    for column in (*CATEGORICAL_FEATURES, "line_id", "data_split"):
        counts = frame[column].fillna("<missing>").value_counts(dropna=False)
        for value, count in counts.items():
            categorical_rows.append(
                {
                    "column": column,
                    "value": str(value),
                    "count": int(count),
                    "rate": round(float(count / len(frame)), 6),
                }
            )
    pd.DataFrame(categorical_rows).to_csv(
        destination / "categorical_summary.csv",
        index=False,
    )

    split_summary = pd.DataFrame(
        {
            "rows": split_counts,
            "target_miss_rate": split_miss_rate,
        }
    ).reset_index(names="data_split")
    split_summary.to_csv(destination / "split_summary.csv", index=False)

    _save_bar_chart(
        label_counts.rename(index={0: "Target met", 1: "Target missed"}),
        "Daily target outcome balance",
        "Outcome",
        "Rows",
        destination / "target_balance.png",
    )
    _save_bar_chart(
        split_counts,
        "Chronological dataset split",
        "Split",
        "Rows",
        destination / "split_volume.png",
        color="#2563EB",
    )

    line_miss_rate = frame.groupby("line_id", observed=True)[
        DAILY_RISK_TARGET
    ].mean().sort_values(ascending=False)
    _save_bar_chart(
        line_miss_rate,
        "Target-miss rate by production line",
        "Production line",
        "Target-miss rate",
        destination / "line_miss_rate.png",
        color="#DC7B24",
    )

    numeric_for_correlation = list(numeric_columns) + [DAILY_RISK_TARGET]
    correlations = (
        frame[numeric_for_correlation]
        .corr(numeric_only=True)[DAILY_RISK_TARGET]
        .drop(DAILY_RISK_TARGET)
        .dropna()
        .sort_values(key=lambda values: values.abs(), ascending=False)
        .head(12)
        .sort_values()
    )
    fig, axis = plt.subplots(figsize=(8, 5.5))
    colors = ["#B91C1C" if value < 0 else "#0F766E" for value in correlations]
    correlations.plot(kind="barh", ax=axis, color=colors)
    axis.set_title("Top numeric correlations with target miss", fontweight="bold")
    axis.set_xlabel("Pearson correlation")
    axis.set_ylabel("Feature")
    axis.axvline(0, color="#475569", linewidth=0.8)
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        destination / "target_correlations.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate LinePulse daily-risk EDA artifacts.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = generate_eda_report(args.input, args.output_dir)
    print(
        "EDA complete: "
        f"rows={summary['rows']}, "
        f"columns={summary['columns']}, "
        f"missing_cells={summary['missing_cells']}"
    )
    print(f"Artifacts: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


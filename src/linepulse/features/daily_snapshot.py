"""Build leakage-safe daily line snapshots from operational tables.

The builder uses information available by the 13:00 UTC prediction snapshot
plus historical information from completed prior line-days.

Important:
- Final daily labels and completion fields are never used.
- ``changeover_minutes`` is intentionally not synthesized because the
  bundled raw operational tables do not contain a recoverable source for
  the observed changeover duration.
- The bundled synthetic generator is not available. Historical rolling
  features therefore use an explicit, reproducible production definition:
  mean of the previous up-to-three completed observed line-days.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from linepulse.data.contracts import FORBIDDEN_DAILY_RISK_FEATURES
from linepulse.settings import DEFAULT_DATA_DIR


SNAPSHOT_HOUR_UTC = 13

KEY_COLUMNS = ["work_date", "line_id", "order_id"]

EXACT_RECONSTRUCTED_FEATURES = (
    "day_in_order",
    "complexity_band",
    "sam_minutes",
    "planned_workers",
    "present_workers",
    "attendance_rate",
    "target_to_snapshot",
    "output_to_snapshot",
    "progress_ratio",
    "downtime_to_snapshot",
    "inspected_to_snapshot",
    "defects_to_snapshot",
    "defect_rate_to_snapshot",
    "material_status",
    "root_cause_signal",
    "previous_day_miss",
)

PROVISIONAL_HISTORY_FEATURES = (
    "rolling_3d_efficiency",
    "rolling_3d_downtime",
)

EXTERNAL_REQUIRED_FEATURES = ("changeover_minutes",)


def _require_columns(
    frame: pd.DataFrame,
    dataset_name: str,
    required: set[str],
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}"
        )


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    decimals: int,
) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce")
    numerator = pd.to_numeric(numerator, errors="coerce")

    result = numerator.div(denominator.where(denominator.ne(0)))
    return result.fillna(0.0).round(decimals)


def build_daily_snapshot_features(
    hourly_production: pd.DataFrame,
    attendance_summaries: pd.DataFrame,
    supervisor_notes: pd.DataFrame,
    styles: pd.DataFrame,
) -> pd.DataFrame:
    """Construct daily line features using only point-in-time-safe inputs."""

    hourly = hourly_production.copy()
    attendance = attendance_summaries.copy()
    notes = supervisor_notes.copy()
    style_master = styles.copy()

    _require_columns(
        hourly,
        "hourly_production",
        {
            "factory_id",
            "work_date",
            "hour_start",
            "line_id",
            "order_id",
            "style_id",
            "hourly_target",
            "actual_output",
            "planned_workers",
            "present_workers",
            "downtime_minutes",
            "inspected_qty",
            "defect_count",
            "material_status",
            "data_split",
            "dataset_provenance",
        },
    )

    _require_columns(
        attendance,
        "attendance_summaries",
        {
            "work_date",
            "line_id",
            "planned_workers",
            "present_workers",
        },
    )

    _require_columns(
        notes,
        "supervisor_notes",
        {
            "work_date",
            "line_id",
            "order_id",
            "created_at",
            "root_cause_label",
        },
    )

    _require_columns(
        style_master,
        "styles",
        {
            "style_id",
            "complexity_band",
            "sam_minutes",
        },
    )

    hourly["work_date"] = pd.to_datetime(
        hourly["work_date"]
    ).dt.normalize()

    attendance["work_date"] = pd.to_datetime(
        attendance["work_date"]
    ).dt.normalize()

    notes["work_date"] = pd.to_datetime(
        notes["work_date"]
    ).dt.normalize()

    notes["created_at"] = pd.to_datetime(
        notes["created_at"],
        utc=True,
    )

    hourly["hour_num"] = (
        hourly["hour_start"]
        .astype(str)
        .str.slice(0, 2)
        .astype(int)
    )

    # The 13:00 snapshot contains completed hourly periods 08:00-12:00.
    pre_snapshot = hourly[
        hourly["hour_num"] < SNAPSHOT_HOUR_UTC
    ].copy()

    if pre_snapshot.empty:
        raise ValueError(
            "No hourly production records exist before the 13:00 snapshot."
        )

    snapshot = (
        pre_snapshot.groupby(KEY_COLUMNS, as_index=False)
        .agg(
            factory_id=("factory_id", "first"),
            style_id=("style_id", "first"),
            target_to_snapshot=("hourly_target", "sum"),
            output_to_snapshot=("actual_output", "sum"),
            downtime_to_snapshot=("downtime_minutes", "sum"),
            inspected_to_snapshot=("inspected_qty", "sum"),
            defects_to_snapshot=("defect_count", "sum"),
            data_split=("data_split", "first"),
            dataset_provenance=("dataset_provenance", "first"),
        )
    )

    snapshot["snapshot_at"] = pd.to_datetime(
        snapshot["work_date"].dt.strftime("%Y-%m-%d")
        + "T13:00:00Z",
        utc=True,
    )

    # Latest material status known before the snapshot.
    latest_material = (
        pre_snapshot.sort_values(
            KEY_COLUMNS + ["hour_num"]
        )
        .groupby(KEY_COLUMNS, as_index=False)
        .tail(1)[KEY_COLUMNS + ["material_status"]]
    )

    snapshot = snapshot.merge(
        latest_material,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    # Attendance is a line/day operational summary.
    attendance_features = attendance[
        [
            "work_date",
            "line_id",
            "planned_workers",
            "present_workers",
        ]
    ].drop_duplicates(["work_date", "line_id"])

    snapshot = snapshot.merge(
        attendance_features,
        on=["work_date", "line_id"],
        how="left",
        validate="many_to_one",
    )

    snapshot["attendance_rate"] = _safe_ratio(
        snapshot["present_workers"],
        snapshot["planned_workers"],
        4,
    )

    snapshot["progress_ratio"] = _safe_ratio(
        snapshot["output_to_snapshot"],
        snapshot["target_to_snapshot"],
        4,
    )

    snapshot["defect_rate_to_snapshot"] = _safe_ratio(
        snapshot["defects_to_snapshot"],
        snapshot["inspected_to_snapshot"],
        4,
    )

    # Style master attributes.
    style_features = style_master[
        ["style_id", "complexity_band", "sam_minutes"]
    ].drop_duplicates("style_id")

    snapshot = snapshot.merge(
        style_features,
        on="style_id",
        how="left",
        validate="many_to_one",
    )

    # day_in_order is the sequential observed production day for a
    # line/order pair. This reproduces all 1,224 bundled snapshots.
    snapshot = snapshot.sort_values(
        ["line_id", "order_id", "work_date"]
    ).reset_index(drop=True)

    snapshot["day_in_order"] = (
        snapshot.groupby(["line_id", "order_id"]).cumcount() + 1
    )

    # Latest supervisor root-cause note known at 13:00.
    notes["note_snapshot_at"] = pd.to_datetime(
        notes["work_date"].dt.strftime("%Y-%m-%d")
        + "T13:00:00Z",
        utc=True,
    )

    known_notes = notes[
        notes["created_at"] <= notes["note_snapshot_at"]
    ].copy()

    latest_notes = (
        known_notes.sort_values(
            KEY_COLUMNS + ["created_at"]
        )
        .groupby(KEY_COLUMNS, as_index=False)
        .tail(1)[KEY_COLUMNS + ["root_cause_label"]]
        .rename(
            columns={
                "root_cause_label": "root_cause_signal"
            }
        )
    )

    snapshot = snapshot.merge(
        latest_notes,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    snapshot["root_cause_signal"] = (
        snapshot["root_cause_signal"].fillna("none")
    )

    # Historical features use only completed prior line-days.
    history = (
        hourly.groupby(
            ["line_id", "work_date"],
            as_index=False,
        )
        .agg(
            historical_target=("hourly_target", "sum"),
            historical_output=("actual_output", "sum"),
            historical_downtime=("downtime_minutes", "sum"),
        )
        .sort_values(["line_id", "work_date"])
    )

    history["historical_efficiency"] = _safe_ratio(
        history["historical_output"],
        history["historical_target"],
        8,
    )

    history["historical_miss"] = (
        history["historical_output"]
        < history["historical_target"]
    ).astype(int)

    history["previous_day_miss"] = (
        history.groupby("line_id")["historical_miss"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    history["rolling_3d_efficiency"] = (
        history.groupby("line_id")["historical_efficiency"]
        .transform(
            lambda values: (
                values.shift(1)
                .rolling(3, min_periods=1)
                .mean()
            )
        )
        .fillna(0.92)
        .round(4)
    )

    history["rolling_3d_downtime"] = (
        history.groupby("line_id")["historical_downtime"]
        .transform(
            lambda values: (
                values.shift(1)
                .rolling(3, min_periods=1)
                .mean()
            )
        )
        .fillna(0.0)
        .round(2)
    )

    snapshot = snapshot.merge(
        history[
            [
                "line_id",
                "work_date",
                "previous_day_miss",
                "rolling_3d_efficiency",
                "rolling_3d_downtime",
            ]
        ],
        on=["line_id", "work_date"],
        how="left",
        validate="many_to_one",
    )

    output_columns = [
        "factory_id",
        "snapshot_at",
        "work_date",
        "line_id",
        "order_id",
        "style_id",
        "day_in_order",
        "complexity_band",
        "sam_minutes",
        "planned_workers",
        "present_workers",
        "attendance_rate",
        "target_to_snapshot",
        "output_to_snapshot",
        "progress_ratio",
        "downtime_to_snapshot",
        "inspected_to_snapshot",
        "defects_to_snapshot",
        "defect_rate_to_snapshot",
        "material_status",
        "root_cause_signal",
        "rolling_3d_efficiency",
        "rolling_3d_downtime",
        "previous_day_miss",
        "data_split",
        "dataset_provenance",
    ]

    result = snapshot[output_columns].sort_values(
        ["line_id", "work_date", "order_id"]
    ).reset_index(drop=True)

    leaked = set(result.columns) & FORBIDDEN_DAILY_RISK_FEATURES
    if leaked:
        raise AssertionError(
            f"Forbidden future information entered snapshot: {sorted(leaked)}"
        )

    return result


def build_daily_snapshot_features_from_directory(
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """Load bundled operational tables and build point-in-time snapshots."""

    base = Path(data_dir)

    return build_daily_snapshot_features(
        hourly_production=pd.read_csv(
            base / "relational" / "hourly_production.csv"
        ),
        attendance_summaries=pd.read_csv(
            base / "relational" / "attendance_summaries.csv"
        ),
        supervisor_notes=pd.read_csv(
            base / "relational" / "supervisor_notes.csv"
        ),
        styles=pd.read_csv(
            base / "relational" / "styles.csv"
        ),
    )

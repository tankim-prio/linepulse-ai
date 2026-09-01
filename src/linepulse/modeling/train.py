"""Train and evaluate the CPU baseline for daily production risk.

Logistic-regression training is approximately O(i * n * p), where i is the
solver iteration count, n is the number of rows, and p is the encoded feature
count. Memory usage is O(n * p). The bundled dataset is intentionally small.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from linepulse.analytics.eda import load_daily_risk_data
from linepulse.data.contracts import (
    DAILY_RISK_FEATURES,
    DAILY_RISK_TARGET,
    FORBIDDEN_DAILY_RISK_FEATURES,
)
from linepulse.settings import DEFAULT_DATA_DIR, PROJECT_ROOT


DEFAULT_INPUT_PATH = DEFAULT_DATA_DIR / "modeling" / "daily_line_ml.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "daily_target_risk.joblib"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "modeling"
RANDOM_STATE = 42
CATEGORICAL_FEATURES = [
    "complexity_band",
    "material_status",
    "root_cause_signal",
]
NUMERIC_FEATURES = [
    feature
    for feature in DAILY_RISK_FEATURES
    if feature not in CATEGORICAL_FEATURES
]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_logistic_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=RANDOM_STATE,
                    solver="liblinear",
                ),
            ),
        ]
    )


def build_dummy_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", DummyClassifier(strategy="prior")),
        ]
    )


def _positive_probabilities(model: Pipeline, features: pd.DataFrame):
    probabilities = model.predict_proba(features)
    class_labels = list(model.named_steps["classifier"].classes_)
    positive_index = class_labels.index(1)
    return probabilities[:, positive_index]


def select_threshold(labels: pd.Series, probabilities) -> float:
    """Select the validation threshold that maximizes F1."""

    candidates = [value / 100 for value in range(10, 91)]
    scored = []
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(labels, predictions, zero_division=0)
        scored.append((float(score), -abs(threshold - 0.5), threshold))
    return float(max(scored)[2])


def evaluate_predictions(
    labels: pd.Series,
    probabilities,
    threshold: float,
) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "rows": int(len(labels)),
        "positive_rate": round(float(labels.mean()), 6),
        "threshold": round(float(threshold), 4),
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 6),
        "pr_auc": round(float(average_precision_score(labels, probabilities)), 6),
        "accuracy": round(float(accuracy_score(labels, predictions)), 6),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(labels, predictions)),
            6,
        ),
        "precision": round(
            float(precision_score(labels, predictions, zero_division=0)),
            6,
        ),
        "recall": round(
            float(recall_score(labels, predictions, zero_division=0)),
            6,
        ),
        "f1": round(float(f1_score(labels, predictions, zero_division=0)), 6),
        "brier_score": round(float(brier_score_loss(labels, probabilities)), 6),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def _validate_chronology(frame: pd.DataFrame) -> None:
    dates = {
        split: frame.loc[frame["data_split"] == split, "work_date"]
        for split in ("train", "validation", "test")
    }
    if any(values.empty for values in dates.values()):
        raise ValueError("Train, validation, and test splits must all be present.")
    if not (
        dates["train"].max() < dates["validation"].min()
        and dates["validation"].max() < dates["test"].min()
    ):
        raise ValueError("Dataset splits are not strictly chronological.")


def _write_model_card(path: Path, metrics: dict[str, object]) -> None:
    validation = metrics["validation"]["logistic_regression"]
    test = metrics["test"]["logistic_regression"]
    content = f"""# Daily target-risk baseline model

## Purpose

Estimate whether a production line will miss its daily target using only
signals available at the 13:00 snapshot.

## Data and limitations

- Data is 100% synthetic and cannot establish real factory performance.
- Split is chronological; no random row split is used.
- The model is a CPU-friendly logistic-regression baseline.
- Predictions support human decisions and must not trigger autonomous actions.

## Selected features

{chr(10).join(f'- `{feature}`' for feature in DAILY_RISK_FEATURES)}

## Validation metrics

- PR-AUC: {validation['pr_auc']:.3f}
- ROC-AUC: {validation['roc_auc']:.3f}
- Precision: {validation['precision']:.3f}
- Recall: {validation['recall']:.3f}
- F1: {validation['f1']:.3f}
- Selected threshold: {validation['threshold']:.2f}

## Held-out test metrics

- PR-AUC: {test['pr_auc']:.3f}
- ROC-AUC: {test['roc_auc']:.3f}
- Precision: {test['precision']:.3f}
- Recall: {test['recall']:.3f}
- F1: {test['f1']:.3f}

## Leakage controls

The feature contract excludes daily final output, target labels, high-defect
labels, and completion fields. Test data is evaluated once after threshold
selection on validation data.
"""
    path.write_text(content, encoding="utf-8")


def train_daily_risk_model(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    report_dir: Path | str = DEFAULT_REPORT_DIR,
) -> dict[str, object]:
    """Train the dummy and logistic baselines and save audited artifacts."""

    leaked = set(DAILY_RISK_FEATURES) & FORBIDDEN_DAILY_RISK_FEATURES
    if leaked:
        raise ValueError(f"Forbidden features selected: {sorted(leaked)}")

    frame = load_daily_risk_data(input_path)
    _validate_chronology(frame)
    synthetic_only = bool(
        (frame["dataset_provenance"] == "synthetic").all()
    )
    if not synthetic_only:
        raise ValueError("This baseline pipeline accepts only declared synthetic data.")
    target_values = set(frame[DAILY_RISK_TARGET].dropna().astype(int))
    if target_values != {0, 1}:
        raise ValueError(f"Target must contain both classes; found {target_values}")

    splits = {
        name: frame[frame["data_split"] == name].copy()
        for name in ("train", "validation", "test")
    }
    train_x = splits["train"][list(DAILY_RISK_FEATURES)]
    train_y = splits["train"][DAILY_RISK_TARGET].astype(int)
    validation_x = splits["validation"][list(DAILY_RISK_FEATURES)]
    validation_y = splits["validation"][DAILY_RISK_TARGET].astype(int)
    test_x = splits["test"][list(DAILY_RISK_FEATURES)]
    test_y = splits["test"][DAILY_RISK_TARGET].astype(int)

    dummy = build_dummy_pipeline()
    logistic = build_logistic_pipeline()
    dummy.fit(train_x, train_y)
    logistic.fit(train_x, train_y)

    dummy_validation_probability = _positive_probabilities(dummy, validation_x)
    logistic_validation_probability = _positive_probabilities(
        logistic,
        validation_x,
    )
    selected_threshold = select_threshold(
        validation_y,
        logistic_validation_probability,
    )
    dummy_test_probability = _positive_probabilities(dummy, test_x)
    logistic_test_probability = _positive_probabilities(logistic, test_x)

    metrics: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(Path(input_path)),
        "target": DAILY_RISK_TARGET,
        "feature_count": len(DAILY_RISK_FEATURES),
        "features": list(DAILY_RISK_FEATURES),
        "forbidden_feature_overlap": sorted(leaked),
        "split_rows": {
            name: int(len(split_frame))
            for name, split_frame in splits.items()
        },
        "validation": {
            "dummy_prior": evaluate_predictions(
                validation_y,
                dummy_validation_probability,
                0.5,
            ),
            "logistic_regression": evaluate_predictions(
                validation_y,
                logistic_validation_probability,
                selected_threshold,
            ),
        },
        "test": {
            "dummy_prior": evaluate_predictions(
                test_y,
                dummy_test_probability,
                0.5,
            ),
            "logistic_regression": evaluate_predictions(
                test_y,
                logistic_test_probability,
                selected_threshold,
            ),
        },
        "synthetic_only": synthetic_only,
    }

    destination_model = Path(model_path)
    destination_reports = Path(report_dir)
    destination_model.parent.mkdir(parents=True, exist_ok=True)
    destination_reports.mkdir(parents=True, exist_ok=True)

    artifact = {
        "pipeline": logistic,
        "features": list(DAILY_RISK_FEATURES),
        "target": DAILY_RISK_TARGET,
        "threshold": selected_threshold,
        "trained_at": metrics["generated_at"],
        "synthetic_training_data": True,
    }
    joblib.dump(artifact, destination_model)
    (destination_reports / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    prediction_rows = []
    for split_name, split_frame, labels, probabilities in (
        (
            "validation",
            splits["validation"],
            validation_y,
            logistic_validation_probability,
        ),
        ("test", splits["test"], test_y, logistic_test_probability),
    ):
        rows = split_frame[
            ["snapshot_id", "work_date", "line_id", "order_id"]
        ].copy()
        rows["data_split"] = split_name
        rows["actual_label"] = labels.values
        rows["risk_probability"] = probabilities
        rows["predicted_label"] = (
            probabilities >= selected_threshold
        ).astype(int)
        prediction_rows.append(rows)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        destination_reports / "predictions.csv",
        index=False,
    )
    _write_model_card(destination_reports / "model_card.md", metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the LinePulse daily target-risk baseline.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metrics = train_daily_risk_model(
        input_path=args.input,
        model_path=args.model,
        report_dir=args.report_dir,
    )
    validation = metrics["validation"]["logistic_regression"]
    test = metrics["test"]["logistic_regression"]
    print(
        "Training complete: "
        f"validation_pr_auc={validation['pr_auc']:.3f}, "
        f"test_pr_auc={test['pr_auc']:.3f}, "
        f"threshold={validation['threshold']:.2f}"
    )
    print(f"Model: {args.model.resolve()}")
    print(f"Reports: {args.report_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

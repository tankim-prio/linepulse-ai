from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib

from linepulse.data.contracts import FORBIDDEN_DAILY_RISK_FEATURES
from linepulse.modeling.train import train_daily_risk_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "linepulse"
    / "modeling"
    / "daily_line_ml.csv"
)


class ModelingTests(unittest.TestCase):
    def test_training_outputs_are_leakage_safe_and_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "daily_target_risk.joblib"
            report_dir = root / "reports"

            metrics = train_daily_risk_model(
                input_path=INPUT_PATH,
                model_path=model_path,
                report_dir=report_dir,
            )
            artifact = joblib.load(model_path)

            selected = set(artifact["features"])
            validation = metrics["validation"]
            test = metrics["test"]

            self.assertFalse(selected & FORBIDDEN_DAILY_RISK_FEATURES)
            self.assertEqual(metrics["split_rows"], {
                "train": 808,
                "validation": 208,
                "test": 208,
            })
            self.assertGreater(
                validation["logistic_regression"]["pr_auc"],
                validation["dummy_prior"]["pr_auc"],
            )
            self.assertGreater(
                test["logistic_regression"]["pr_auc"],
                test["dummy_prior"]["pr_auc"],
            )
            self.assertGreaterEqual(artifact["threshold"], 0.0)
            self.assertLessEqual(artifact["threshold"], 1.0)
            self.assertTrue((report_dir / "metrics.json").is_file())
            self.assertTrue((report_dir / "predictions.csv").is_file())
            self.assertTrue((report_dir / "model_card.md").is_file())


if __name__ == "__main__":
    unittest.main()


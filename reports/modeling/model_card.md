# Daily target-risk baseline model

## Purpose

Estimate whether a production line will miss its daily target using only
signals available at the 13:00 snapshot.

## Data and limitations

- Data is 100% synthetic and cannot establish real factory performance.
- Split is chronological; no random row split is used.
- The model is a CPU-friendly logistic-regression baseline.
- Predictions support human decisions and must not trigger autonomous actions.

## Selected features

- `day_in_order`
- `complexity_band`
- `sam_minutes`
- `planned_workers`
- `present_workers`
- `attendance_rate`
- `target_to_snapshot`
- `output_to_snapshot`
- `progress_ratio`
- `downtime_to_snapshot`
- `inspected_to_snapshot`
- `defects_to_snapshot`
- `defect_rate_to_snapshot`
- `material_status`
- `changeover_minutes`
- `root_cause_signal`
- `rolling_3d_efficiency`
- `rolling_3d_downtime`
- `previous_day_miss`

## Validation metrics

- PR-AUC: 0.977
- ROC-AUC: 0.940
- Precision: 0.908
- Recall: 0.932
- F1: 0.920
- Selected threshold: 0.26

## Held-out test metrics

- PR-AUC: 0.992
- ROC-AUC: 0.979
- Precision: 0.934
- Recall: 0.959
- F1: 0.946

## Leakage controls

The feature contract excludes daily final output, target labels, high-defect
labels, and completion fields. Test data is evaluated once after threshold
selection on validation data.

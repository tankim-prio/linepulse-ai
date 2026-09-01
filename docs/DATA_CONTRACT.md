# Phase 1 data contract

## Trust boundary

All bundled records are synthetic. Every tabular dataset must declare
`dataset_provenance=synthetic`. No individual employee identifiers are allowed.

## Required guarantees

The validator enforces the following:

1. Every dataset declared in `metadata/dataset_manifest.csv` exists.
2. Actual row and column counts match the manifest.
3. Required dictionary columns exist and contain no null values.
4. Declared primary keys are non-null and unique.
5. Explicit foreign keys resolve to their parent datasets.
6. Production quantities and durations are non-negative.
7. Present workers never exceed planned workers.
8. Both ML labels contain positive and negative examples.
9. Train, validation, and test dates follow the fixed chronological split.
10. Supervisor notes include Bangla, English, and mixed-language examples.
11. Bangla notes and document sections contain Bengali Unicode characters.
12. The model feature allowlists exclude all declared outcome and target fields.

## Modeling contracts

### Daily target-risk model

- Unit: one production line and work date.
- Prediction time: 13:00 UTC in the synthetic scenario.
- Target: `label_daily_target_missed`.
- Allowed inputs: fields known by `snapshot_at`.
- Forbidden inputs: `daily_actual_output`, actual completion values, and any
  target or post-outcome field.

### Order-delay model

- Unit: one completed production order.
- Prediction time: order start.
- Target: `label_order_delayed`.
- Allowed inputs: order quantity, style SAM, complexity, schedule slack, and
  prior line performance.
- Forbidden inputs: actual completion date, actual duration, and target fields.

## Split contract

- Train: through 2025-04-30.
- Validation: 2025-05-01 through 2025-05-31.
- Test: 2025-06-01 onward.

Rows must never be randomly moved across these periods during model training.

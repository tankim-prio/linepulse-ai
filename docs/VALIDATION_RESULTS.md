# Verification results

Verification date: 2026-09-01

## Data and software checks

- Declared datasets: **21**
- Total declared rows: **25,277**
- Data-contract checks: **199 passed, 0 failed**
- Automated tests: **5 passed, 0 failed**
- EDA artifact generation: **passed**
- Editable package installation: **passed**
- Saved-model loading: **passed**
- CLI execution: **passed**

The tests cover the valid dataset, missing declared files, duplicate primary
keys, required analytical artifacts, leakage-safe features, chronological
splits, model serialization, and comparison with a dummy benchmark.

## Synthetic baseline results

The following results were reproduced on the bundled synthetic dataset:

| Dataset | Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| Validation | Dummy prior | 0.712 | 0.500 | 0.712 | 1.000 | 0.831 |
| Validation | Logistic regression | 0.977 | 0.940 | 0.908 | 0.932 | 0.920 |
| Held-out test | Dummy prior | 0.707 | 0.500 | 0.707 | 1.000 | 0.828 |
| Held-out test | Logistic regression | 0.992 | 0.979 | 0.934 | 0.959 | 0.946 |

The classification threshold was selected on validation data and then applied
unchanged to the held-out test set.

These unusually strong metrics reflect intentionally learnable synthetic
relationships. They do not demonstrate real factory accuracy or business
impact.

## Verified environments

- Build verification: Python 3.12, pandas 2.2.3, scikit-learn 1.8.0,
  matplotlib 3.10.8, CPU only.
- Windows setup and data validation were independently executed with Python
  3.11.9 and passed.

The new analytics PowerShell command must still be executed on the target
Windows computer. Its underlying Python modules and automated tests were
executed successfully in the build environment.


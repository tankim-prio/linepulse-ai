# LinePulse AI

LinePulse AI is a CPU-friendly portfolio project for garment production-risk
prediction, bilingual operational knowledge retrieval, and human-approved
corrective-action workflows.

The bundled data is **100% synthetic**. It contains no real factory, buyer,
employee, machine, or order information. Model results must never be presented
as measured business impact.

## Implemented capabilities

- 21 related operational, modeling, RAG, agent, and security datasets;
- schema, key, relationship, chronology, provenance, and safety validation;
- reproducible exploratory analysis with summary tables and charts;
- a dummy benchmark and leakage-safe logistic-regression risk model;
- chronological train, validation, and held-out test evaluation;
- saved model metadata, predictions, metrics, and model card;
- automated success and failure-injection tests;
- Windows PowerShell commands for setup, validation, and execution.

## Hardware target

- Windows 10
- Intel Core i3 8th generation or similar
- 8 GB RAM
- No GPU required
- Python 3.10 or newer; Python 3.11 is recommended

## Windows setup

Open PowerShell in the repository root:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
.\scripts\validate_project.ps1
```

The scripts call `.venv\Scripts\python.exe` directly, so activating the virtual
environment is optional.

## Run analytics and model training

```powershell
.\scripts\run_analytics.ps1
```

This command performs the following in order:

1. validates all declared datasets;
2. generates the exploratory report and PNG charts;
3. trains the dummy and logistic-regression baselines;
4. selects the classification threshold using validation data;
5. evaluates the selected model once on held-out test data; and
6. runs the complete automated test suite.

Generated outputs:

```text
reports/eda/                         EDA tables, JSON summary, and PNG charts
reports/modeling/metrics.json        Validation and held-out test metrics
reports/modeling/predictions.csv     Validation and test predictions
reports/modeling/model_card.md       Intended use, metrics, and limitations
models/daily_target_risk.joblib      Trained pipeline and decision threshold
```

## Manual commands

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m linepulse.cli --data-dir data\linepulse
.\.venv\Scripts\python.exe -m linepulse.analytics.eda
.\.venv\Scripts\python.exe -m linepulse.modeling.train
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Repository structure

```text
linepulse-ai/
├── data/linepulse/            Synthetic datasets and bilingual SOPs
├── docs/                      Contracts, validation evidence, and roadmap
├── models/                    Generated trained-model artifacts
├── reports/eda/               Generated analytical summaries and charts
├── reports/modeling/          Generated evaluation and model-card artifacts
├── scripts/                   Windows setup and execution commands
├── src/linepulse/analytics/   Reproducible EDA implementation
├── src/linepulse/data/        Data contracts and validation
├── src/linepulse/modeling/    Leakage-safe training and evaluation
└── tests/                     Automated tests
```

## Modeling controls

- Unit of prediction: one production line on one work date.
- Prediction time: synthetic 13:00 UTC snapshot.
- Target: `label_daily_target_missed`.
- Split strategy: chronological, never random.
- Threshold selection: validation data only.
- Held-out test data: evaluated after threshold selection.
- Forbidden inputs: daily final output, outcome labels, and completion fields.
- Intended role: decision support; no autonomous operational actions.

## Git workflow

After completing and reviewing a coherent change:

```powershell
git status
git add .
git commit -m "Add reproducible risk analytics and baseline model"
git push
```

Never commit `.env`, credentials, personal data, real factory data, or trained
artifacts derived from confidential information.

See `docs/VALIDATION_RESULTS.md` for verified implementation results and
`docs/ROADMAP.md` for planned capabilities.


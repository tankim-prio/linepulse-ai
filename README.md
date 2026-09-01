# LinePulse AI

LinePulse AI is a portfolio-grade, CPU-friendly system for garment production
risk prediction, bilingual operational knowledge retrieval, and
human-approved corrective-action workflows.

The included data is **100% synthetic**. It contains no real factory, buyer,
employee, machine, or order records. Synthetic results must not be presented as
real business impact.

## Current milestone: Phase 1 complete

This starter repository contains:

- 21 related CSV datasets and eight bilingual synthetic SOPs;
- an executable data-contract validator;
- primary-key, foreign-key, schema, business-rule, language, split, and
  provenance checks;
- standard-library automated tests;
- Windows PowerShell setup and verification scripts;
- the complete project roadmap and data contract.

## Hardware target

- Windows 10
- Intel Core i3 8th generation or similar
- 8 GB RAM
- No GPU required
- Python 3.10 or newer

## Quick start on Windows

Open PowerShell in this repository and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
.\scripts\check_phase1.ps1
```

The scripts use `.venv\Scripts\python.exe` directly, so activating the virtual
environment is optional.

## Create the Git repository

After the checks pass:

```powershell
git init
git add .
git commit -m "Complete LinePulse Phase 1 data foundation"
```

Then create an empty GitHub repository and follow GitHub's displayed commands
to add the remote and push the `main` branch. Never commit `.env` or real
factory data.

## Manual setup

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m linepulse.cli --data-dir data\linepulse
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The selected Python version must be at least 3.10.

## Validation command

```powershell
.\.venv\Scripts\linepulse-validate.exe `
  --data-dir data\linepulse `
  --report reports\data_validation_report.json
```

The command exits with code `0` only when every error-level check passes. This
makes it suitable for local checks and future continuous integration.

## Repository structure

```text
linepulse-ai/
├── data/linepulse/        Synthetic CSV, JSON, Markdown, and Excel artifacts
├── docs/                  Roadmap and data contract
├── reports/               Generated validation reports
├── scripts/               Windows setup and verification scripts
├── src/linepulse/         Application package
└── tests/                 Automated Phase 1 tests
```

## Important modeling boundary

For the daily target-risk model, features must contain only information
available at `snapshot_at`. Never use `daily_actual_output`, final completion
fields, or label columns as inputs. The next milestone will implement this
leakage-safe feature pipeline and a CPU baseline model.

## Next milestone

Phase 2 will add exploratory analysis, a reproducible feature builder, a dummy
baseline, and the first scikit-learn target-risk model using the existing
time-based train, validation, and test splits.

See `docs/PHASE1_RESULTS.md` for the checks actually executed on this package.

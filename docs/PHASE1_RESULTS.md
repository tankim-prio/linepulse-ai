# Phase 1 verification results

Verification date: 2026-09-01

## Confirmed results

- Dataset validator: **passed**
- Declared datasets: **21**
- Total declared rows: **25,277**
- Validation checks: **199 passed, 0 failed**
- Automated tests: **3 passed, 0 failed**
- Editable package installation: **passed in an isolated environment**
- Installed command `linepulse-validate`: **executed successfully**

The automated tests verified:

1. the complete bundled package passes its contract;
2. a missing declared CSV produces a validation failure; and
3. a duplicated primary key produces a validation failure.

## Environment used for verification

- Python 3.12
- pandas 2.2.3
- CPU execution only
- No GPU libraries or services

## Verification boundary

The Python package, CLI, validation logic, and tests were executed in the build
environment. The PowerShell scripts were reviewed but were not executed because
PowerShell was unavailable in that Linux environment. They use standard
Windows Python-launcher and virtual-environment commands.

The data remains synthetic. Passing validation confirms internal consistency;
it does not establish real-world factory accuracy or business impact.


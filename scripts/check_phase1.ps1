$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$DataDir = Join-Path $ProjectRoot "data\linepulse"
$Report = Join-Path $ProjectRoot "reports\data_validation_report.json"

if (-not (Test-Path $Python)) {
    throw "Virtual environment not found. Run scripts\setup_windows.ps1 first."
}

Set-Location $ProjectRoot
& $Python -m linepulse.cli --data-dir $DataDir --report $Report
& $Python -m unittest discover -s tests -v

Write-Host "Phase 1 validation and tests passed."


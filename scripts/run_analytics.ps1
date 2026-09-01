$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Virtual environment not found. Run scripts\setup_windows.ps1 first."
}

Set-Location $ProjectRoot
& $Python -m linepulse.cli `
    --data-dir "data\linepulse" `
    --report "reports\data_validation_report.json"
& $Python -m linepulse.analytics.eda
& $Python -m linepulse.modeling.train
& $Python -m unittest discover -s tests -v

Write-Host "Analytics, model training, and automated tests completed."


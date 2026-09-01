$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VirtualEnv = Join-Path $ProjectRoot ".venv"
$Python = Join-Path $VirtualEnv "Scripts\python.exe"

Set-Location $ProjectRoot

if (-not (Test-Path $Python)) {
    py -3 -m venv $VirtualEnv
}

& $Python -m pip install --upgrade pip
& $Python -m pip install -e $ProjectRoot

Write-Host "LinePulse development environment is ready."

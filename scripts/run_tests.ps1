#!/usr/bin/env pwsh
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = $env:PYTHON
if (-not $Python) { $Python = "python" }

function Log { param([string]$Message) Write-Host "[run_tests] $Message" }

Set-Location $Root

if (-not $env:SKIP_RUFF) {
    if (Get-Command ruff -ErrorAction SilentlyContinue) {
        Log "ruff check webapp"
        ruff check webapp
    }
    else {
        Log "ruff not installed; set SKIP_RUFF=1 to skip"
        exit 1
    }
}
else {
    Log "SKIP_RUFF set; skipping ruff"
}

if (-not $env:SKIP_MYPY) {
    if (Get-Command mypy -ErrorAction SilentlyContinue) {
        Log "mypy (formats + tests)"
        mypy webapp/parser/handlers/formats webapp/tests
    }
    else {
        Log "mypy not installed; set SKIP_MYPY=1 to skip"
        exit 1
    }
}
else {
    Log "SKIP_MYPY set; skipping mypy"
}

Log "pytest $($Args -join ' ')"
& $Python -m pytest @Args

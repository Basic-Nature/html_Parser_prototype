#!/usr/bin/env pwsh
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$EnvFile = if ($env:ENV_FILE) { $env:ENV_FILE } else { Join-Path $Root ".env" }
$Required = @("FLASK_SECRET_KEY", "POSTGRES_HOST", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD")

function Log { param([string]$Message) Write-Host "[run_webapp] $Message" }
function Fatal { param([string]$Message) Write-Host "[run_webapp][fatal] $Message" -ForegroundColor Red; exit 1 }

Set-Location -Path $Root

if (Test-Path $EnvFile) {
    Log "loading env from $EnvFile"
    foreach ($line in Get-Content $EnvFile) {
        if ($line -match '^(?<k>[A-Za-z_][A-Za-z0-9_]*)=(?<v>.*)$') {
            $key = $Matches.k
            $val = $Matches.v
            if ($val.StartsWith('"') -and $val.EndsWith('"')) { $val = $val.Trim('"') }
            if ($val.StartsWith("'") -and $val.EndsWith("'")) { $val = $val.Trim("'") }
            $envPath = "Env:\$key"
            Set-Item -Path $envPath -Value $val
        }
    }
} else {
    Log "env file not found at $EnvFile (set ENV_FILE to override)"
}

$missing = @()
foreach ($var in $Required) {
    $current = [Environment]::GetEnvironmentVariable($var)
    if (-not $current) { $missing += $var }
}
if ($missing.Count -gt 0) {
    Fatal "Missing required env vars: $($missing -join ' '). Set them in $EnvFile or environment."
}

# Ensure runtime directories exist
foreach ($path in @("input", "output", "uploads", "log")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Root $path) | Out-Null
}

if (-not $env:EMBEDDING_CACHE_DB_MODE) {
    $envLower = "local"
    if ($env:DEPLOY_ENV) {
        $envLower = $env:DEPLOY_ENV.ToLower()
    }
    if ($envLower -in @("", "local", "dev", "development", "test")) {
        $env:EMBEDDING_CACHE_DB_MODE = "off"
        Log "EMBEDDING_CACHE_DB_MODE defaulting to off for $envLower (override to rw/ro as needed)"
    }
}

Log "starting webapp via python -m webapp.Smart_Elections_Parser_Webapp"
& $Python -m webapp.Smart_Elections_Parser_Webapp @Args

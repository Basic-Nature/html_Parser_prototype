Here is the updated `scripts\run_webapp.ps1` file with the suggested code changes incorporated:
````````powershell
#!/usr/bin/env pwsh
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = $env:PYTHON
if (-not $Python) { $Python = "python" }
$EnvFile = $env:ENV_FILE
if (-not $EnvFile) { $EnvFile = Join-Path $Root ".env" }
$Required = @("FLASK_SECRET_KEY", "POSTGRES_HOST", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD")

function Log { param([string]$Message) Write-Host "[run_webapp] $Message" }
function Fatal { param([string]$Message) Write-Host "[run_webapp][fatal] $Message" -ForegroundColor Red; exit 1 }

Set-Location $Root

if (Test-Path $EnvFile) {
    Log "loading env from $EnvFile"
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^(?<k>[A-Za-z_][A-Za-z0-9_]*)=(?<v>.*)$') {
            $key = $Matches['k']
            $val = $Matches['v']
            if ($val.StartsWith('"') -and $val.EndsWith('"')) { $val = $val.Trim('"') }
            if ($val.StartsWith("'") -and $val.EndsWith("'")) { $val = $val.Trim("'") }
            $env:$key = $val
        }
    }
} else {
    Log "env file not found at $EnvFile (set ENV_FILE to override)"
}

$missing = @()
foreach ($var in $Required) {
    if (-not $env:$var) { $missing += $var }
}
if ($missing.Count -gt 0) {
    Fatal "Missing required env vars: $($missing -join ' '). Set them in $EnvFile or environment."
}

# Ensure runtime directories exist
New-Item -ItemType Directory -Force -Path (Join-Path $Root "input") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root "output") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root "uploads") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root "log") | Out-Null

if (-not $env:EMBEDDING_CACHE_DB_MODE) {
    $envLower = ($env:DEPLOY_ENV) ? $env:DEPLOY_ENV.ToLower() : "local"
    if ($envLower -in @("", "local", "dev", "development", "test")) {
        $env:EMBEDDING_CACHE_DB_MODE = "off"
        Log "EMBEDDING_CACHE_DB_MODE defaulting to off for $envLower (override to rw/ro as needed)"
    }
}

Log "starting webapp via python -m webapp.Smart_Elections_Parser_Webapp"
& $Python -m webapp.Smart_Elections_Parser_Webapp @Args

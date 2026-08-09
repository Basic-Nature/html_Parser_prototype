[CmdletBinding()]
param(
    [switch]$ShowCoreDiff
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (
    Resolve-Path (Join-Path $PSScriptRoot "..\..")
).Path

$AuditScript = Join-Path `
    $RepoRoot `
    "scripts\maintenance\audit_docs.ps1"

function Write-GateSection {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Title
    )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host $Title
    Write-Host "============================================================"
    Write-Host ""
}

function Invoke-GateProcess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [string[]]$Arguments = @(),

        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    Write-GateSection -Title $Description

    $process = Start-Process `
        -FilePath $FilePath `
        -ArgumentList $Arguments `
        -WorkingDirectory $RepoRoot `
        -NoNewWindow `
        -Wait `
        -PassThru

    if ($process.ExitCode -ne 0) {
        throw (
            "$Description failed with exit code " +
            "$($process.ExitCode)."
        )
    }
}

Push-Location $RepoRoot

try {
    Write-Host ""
    Write-Host "Election Pulse Documentation Verification Gate"
    Write-Host "Repository: $RepoRoot"

    $npmCommand = Get-Command npm.cmd -ErrorAction Stop
    $gitCommand = Get-Command git.exe -ErrorAction Stop
    $powershellCommand = Get-Command powershell.exe -ErrorAction Stop

    Invoke-GateProcess `
        -FilePath $npmCommand.Source `
        -Arguments @(
            "run",
            "lint:md"
        ) `
        -Description "Markdown lint"

    Invoke-GateProcess `
        -FilePath $powershellCommand.Source `
        -Arguments @(
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            "`"$AuditScript`""
        ) `
        -Description "Documentation audit"

    Invoke-GateProcess `
        -FilePath $gitCommand.Source `
        -Arguments @(
            "--no-pager",
            "diff",
            "--check"
        ) `
        -Description "Git whitespace validation"

    if ($ShowCoreDiff) {
        Invoke-GateProcess `
            -FilePath $gitCommand.Source `
            -Arguments @(
                "--no-pager",
                "diff",
                "--",
                "docs/CORE/constants_reference.md"
            ) `
            -Description "CORE constants diff"
    }

    Write-GateSection `
        -Title "Repository documentation status"

    & $gitCommand.Source `
        --no-pager `
        status `
        --short `
        -- `
        docs `
        scripts/maintenance

    if ($LASTEXITCODE -ne 0) {
        throw (
            "Git status failed with exit code " +
            "$LASTEXITCODE."
        )
    }

    Write-Host ""
    Write-Host "Documentation verification gate complete."
}
catch {
    Write-Host ""
    Write-Host "DOCUMENTATION VERIFICATION FAILED"
    Write-Host $_.Exception.Message
    Write-Host ""

    throw
}
finally {
    Pop-Location
}
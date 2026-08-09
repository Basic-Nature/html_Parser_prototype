<#
.SYNOPSIS
    Reorganize Election Pulse documentation around architectural domains.

.DESCRIPTION
    This script establishes the domain-oriented documentation structure for
    Election Pulse while preserving repository history and avoiding speculative
    file moves.

    Key principles:

    - Dry-run by default.
    - Use git mv for tracked files.
    - Never auto-delete documentation based on heuristics.
    - Preserve implementation history.
    - Keep generated documentation separate from architectural authority.
    - Rewrite only known active-document links when explicitly requested.
    - Generate a review manifest for files that still need classification.
    - Remain compatible with Windows PowerShell 5.1.

.PARAMETER Apply
    Actually perform directory creation and file moves.
    Without this switch the script performs a dry run.

.PARAMETER RewriteKnownLinks
    Rewrite references to known moved files in active documentation.
    Historical and archived documentation is deliberately excluded.

.PARAMETER Validate
    Run repository/documentation validation after the reorganization.

.EXAMPLE
    .\scripts\maintenance\reorganize_docs.ps1

    Dry run.

.EXAMPLE
    .\scripts\maintenance\reorganize_docs.ps1 -Apply

    Apply known-safe moves.

.EXAMPLE
    .\scripts\maintenance\reorganize_docs.ps1 -Apply -RewriteKnownLinks

    Apply moves and update known active-document references.

.EXAMPLE
    .\scripts\maintenance\reorganize_docs.ps1 -Apply -RewriteKnownLinks -Validate

    Apply moves, repair known references, and run validation.
#>

[CmdletBinding()]
param(
    [switch]$Apply,
    [switch]$RewriteKnownLinks,
    [switch]$Validate
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

# UTF-8 without BOM so rewritten Markdown/YAML files remain Git-friendly.
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

# ---------------------------------------------------------------------------
# Repository discovery
# ---------------------------------------------------------------------------

function Get-RepositoryRoot {
    try {
        $root = (& git rev-parse --show-toplevel 2>$null)

        if ($LASTEXITCODE -eq 0 -and $root) {
            return ([string]$root).Trim()
        }
    }
    catch {
        # Fall through to filesystem inference.
    }

    # Script is expected at:
    # <repo>\scripts\maintenance\reorganize_docs.ps1
    return (
        Split-Path `
            (Split-Path $PSScriptRoot -Parent) `
            -Parent
    )
}

$RepoRoot = Get-RepositoryRoot
$DocsRoot = Join-Path $RepoRoot "docs"
$ReportRoot = Join-Path $RepoRoot "output\reports\docs-reorganization"

if (-not (Test-Path -LiteralPath $DocsRoot)) {
    throw "Documentation directory not found: $DocsRoot"
}

Push-Location $RepoRoot

try {

# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------

function Write-Section {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Title
    )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host $Title
    Write-Host "============================================================"
}

function Write-Action {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    if ($Apply) {
        Write-Host "[APPLY] $Message"
    }
    else {
        Write-Host "[DRY RUN] $Message"
    }
}

function ConvertTo-RepoPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    return $Path.Replace("\", "/").TrimStart("/")
}

function Get-RepositoryRelativePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $root = [System.IO.Path]::GetFullPath($RepoRoot)
    $full = [System.IO.Path]::GetFullPath($Path)

    $rootNormalized = $root.Replace("/", "\").TrimEnd("\")
    $fullNormalized = $full.Replace("/", "\")

    if (
        $fullNormalized.StartsWith(
            $rootNormalized + "\",
            [System.StringComparison]::OrdinalIgnoreCase
        )
    ) {
        return $fullNormalized.Substring($rootNormalized.Length + 1).Replace("\", "/")
    }

    throw "Path is outside repository root: $Path"
}

function Get-AbsoluteRepoPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    $normalized = $RelativePath.Replace("/", "\")
    return Join-Path $RepoRoot $normalized
}

# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

function Test-GitTracked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    $path = ConvertTo-RepoPath $RelativePath

    $previousErrorActionPreference = $ErrorActionPreference

    try {
        $ErrorActionPreference = "SilentlyContinue"

        & git ls-files --error-unmatch -- $path 2>$null 1>$null

        return ($LASTEXITCODE -eq 0)
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
}

function Test-GitTrackedExact {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    $normalized = ConvertTo-RepoPath $RelativePath
    $previousErrorActionPreference = $ErrorActionPreference

    try {
        $ErrorActionPreference = "SilentlyContinue"

        $trackedPaths = @(
            & git ls-files -- $normalized 2>$null
        )

        foreach ($trackedPath in $trackedPaths) {
            if (
                ([string]$trackedPath).Trim() -ceq $normalized
            ) {
                return $true
            }
        }

        return $false
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
}
function New-DocumentationDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    $absolutePath = Get-AbsoluteRepoPath $RelativePath

    if (Test-Path -LiteralPath $absolutePath) {
        return
    }

    Write-Action "Create directory: $RelativePath"

    if ($Apply) {
        New-Item `
            -ItemType Directory `
            -Force `
            -Path $absolutePath |
            Out-Null
    }
}

function Get-FileSha256 {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }

    return (
        Get-FileHash `
            -LiteralPath $Path `
            -Algorithm SHA256
    ).Hash
}

function Move-RepositoryFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Source,

        [Parameter(Mandatory = $true)]
        [string]$Destination
    )

    $sourceRelative = ConvertTo-RepoPath $Source
    $destinationRelative = ConvertTo-RepoPath $Destination

    $sourceTrackedExact = Test-GitTrackedExact $sourceRelative
    $destinationTrackedExact = Test-GitTrackedExact $destinationRelative

    # Git already contains the intended destination path.
    # This matters especially for case-only renames on Windows,
    # where both filesystem paths may appear to exist.
    if (
        -not $sourceTrackedExact -and
        $destinationTrackedExact
    ) {
        Write-Host "[OK] Already moved: $destinationRelative"
        return
    }

    $sourceAbsolute = Get-AbsoluteRepoPath $sourceRelative
    $destinationAbsolute = Get-AbsoluteRepoPath $destinationRelative

    $sourceExists = Test-Path -LiteralPath $sourceAbsolute
    $destinationExists = Test-Path -LiteralPath $destinationAbsolute

    # Idempotency: move was already completed.
    if (-not $sourceExists -and $destinationExists) {
        Write-Host "[OK] Already moved: $destinationRelative"
        return
    }

    # Nothing exists at either location.
    if (-not $sourceExists -and -not $destinationExists) {
        Write-Host "[SKIP] Source does not exist: $sourceRelative"
        return
    }

    # Never overwrite another document.
    if (
        $sourceExists -and
        $destinationExists -and
        ($sourceAbsolute -ne $destinationAbsolute)
    ) {
        Write-Warning @"
Cannot move because destination already exists.

Source:
  $sourceRelative

Destination:
  $destinationRelative
"@
        return
    }

    $destinationDirectory = Split-Path $destinationAbsolute -Parent

    if (-not (Test-Path -LiteralPath $destinationDirectory)) {
        $relativeDirectory = Get-RepositoryRelativePath $destinationDirectory
        New-DocumentationDirectory $relativeDirectory
    }

    Write-Action "$sourceRelative -> $destinationRelative"

    if (-not $Apply) {
        return
    }

    $tracked = Test-GitTracked $sourceRelative

    # Windows file systems are normally case-insensitive.
    # A two-step move is necessary for case-only renames.
    $caseOnlyRename = (
        $sourceRelative.ToLowerInvariant() -eq
        $destinationRelative.ToLowerInvariant()
    ) -and (
        $sourceRelative -cne $destinationRelative
    )

    if ($tracked) {

        if ($caseOnlyRename) {
            $tempRelative = "$destinationRelative.__docs_reorg_tmp__"

            & git mv -- $sourceRelative $tempRelative

            if ($LASTEXITCODE -ne 0) {
                throw "git mv failed: $sourceRelative -> $tempRelative"
            }

            & git mv -- $tempRelative $destinationRelative

            if ($LASTEXITCODE -ne 0) {
                throw "git mv failed: $tempRelative -> $destinationRelative"
            }
        }
        else {
            & git mv -- $sourceRelative $destinationRelative

            if ($LASTEXITCODE -ne 0) {
                throw "git mv failed: $sourceRelative -> $destinationRelative"
            }
        }
    }
    else {
        $sourceHash = Get-FileSha256 $sourceAbsolute

        Move-Item `
            -LiteralPath $sourceAbsolute `
            -Destination $destinationAbsolute

        $destinationHash = Get-FileSha256 $destinationAbsolute

    if (
        $sourceHash -and
        $destinationHash -and
        $sourceHash -cne $destinationHash
    ) {
        $message = @(
            "Content verification failed after moving untracked documentation."
            ""
            "Source:"
            "  $sourceRelative"
            ""
            "Destination:"
            "  $destinationRelative"
            ""
            "Source SHA256:"
            "  $sourceHash"
            ""
            "Destination SHA256:"
            "  $destinationHash"
        ) -join [Environment]::NewLine

        throw $message
    }

        Write-Host "[OK] Content hash verified: $destinationRelative"
    }
}

# ---------------------------------------------------------------------------
# Domain structure
# ---------------------------------------------------------------------------

Write-Section "Election Pulse Documentation Domains"

$DomainDirectories = @(

    # Long-lived architectural boundaries.
    "docs/ARCHITECTURE",

    # Contracts implemented by current source code.
    "docs/CORE",

    # End-user and operator-facing capabilities.
    "docs/FEATURES",

    # Validation, reconciliation, integrity, and review.
    "docs/QUALITY",

    # Deployment and runtime environments.
    "docs/DEPLOYMENT",
    "docs/DEPLOYMENT/security",

    # Contributor workflows and generated repository diagnostics.
    "docs/DEVELOPMENT",
    "docs/DEVELOPMENT/generated",

    # Integrity principles, policy, provenance, and decisions.
    "docs/GOVERNANCE",
    "docs/GOVERNANCE/decision-records",

    # Work currently underway.
    "docs/implementation-phases",

    # Superseded or completed implementations.
    "docs/implementation-history",
    "docs/implementation-history/2026-url-parser",
    "docs/implementation-history/2026-context-storage",
    "docs/implementation-history/2026-navigation-learning",
    "docs/implementation-history/2026-state-handler-integration",
    "docs/implementation-history/2026-audits",
    "docs/implementation-history/2026-deployment",
    "docs/implementation-history/2026-incidents",
    "docs/implementation-history/2026-ballot-lens",
    "docs/implementation-history/2026-data-import",
    "docs/implementation-history/2026-integrity-ui",
    "docs/implementation-history/2026-validation",
    "docs/implementation-history/2026-status-reconciliation",
    "docs/implementation-history/2026-data-comparison",
    "docs/implementation-history/2026-google-sheets",
    "docs/implementation-history/core-consolidation",
    "docs/implementation-history/operations",
    "docs/implementation-history/temp-consolidation",
    "docs/implementation-history/2026-governance",

    # Historical material retained without current authority.
    "docs/archived",
    "docs/session-logs",

    # Jekyll infrastructure.
    "docs/_data",
    "docs/_layouts",
    "docs/assets"
)

foreach ($directory in $DomainDirectories) {
    New-DocumentationDirectory $directory
}

# ---------------------------------------------------------------------------
# Explicit disposition manifest
# ---------------------------------------------------------------------------
#
# IMPORTANT:
# These are deliberate architectural decisions.
#
# This section should grow as documentation is reviewed.
# Do not replace it with automatic "guess and move" behavior.
# ---------------------------------------------------------------------------

Write-Section "Known Documentation Moves"

$KnownMoves = @(

    # -----------------------------------------------------------------------
    # Historical feature implementations
    # -----------------------------------------------------------------------

    @{
        Source = "docs/FEATURES/URL_PARSER_IMPLEMENTATION_SUMMARY.md"
        Destination = "docs/implementation-history/2026-url-parser/URL_PARSER_IMPLEMENTATION_SUMMARY.md"
        Reason = "Implementation summary; useful history but not active feature authority."
    },

    @{
        Source = "docs/FEATURES/STORAGE_ARCHITECTURE.md"
        Destination = "docs/implementation-history/2026-context-storage/STORAGE_ARCHITECTURE_V1.md"
        Reason = "Legacy context/storage architecture superseded by the domain model."
    },

    @{
        Source = "docs/EXECUTIVE-SUMMARY.md"
        Destination = "docs/implementation-history/2026-navigation-learning/EXECUTIVE_SUMMARY_V1.md"
        Reason = "Historical executive summary for the dynamic navigation and learning implementation."
    },

    @{
        Source = "docs/SESSION-SUMMARY.md"
        Destination = "docs/implementation-history/2026-navigation-learning/SESSION_SUMMARY_V1.md"
        Reason = "Historical session summary for the dynamic navigation and learning implementation."
    },

    @{
        Source = "docs/IMPLEMENTATION-STATE.md"
        Destination = "docs/implementation-history/2026-navigation-learning/IMPLEMENTATION_STATE_V1.md"
        Reason = "Historical implementation-state document for the dynamic navigation and learning architecture."
    },

    @{
        Source = "docs/VALIDATION-STATUS.md"
        Destination = "docs/implementation-history/2026-navigation-learning/VALIDATION_STATUS_V1.md"
        Reason = "Historical validation and production-readiness report for the dynamic navigation and learning implementation."
    },

    # -----------------------------------------------------------------------
    # Former CORE architecture documents
    # -----------------------------------------------------------------------

    @{
        Source = "docs/CORE/ARCHITECTURE.md"
        Destination = "docs/implementation-history/core-consolidation/ARCHITECTURE_V1.md"
        Reason = "Legacy architecture mixes intended and implemented behavior."
    },

    @{
        Source = "docs/CORE/DATA_MODELS.md"
        Destination = "docs/implementation-history/core-consolidation/DATA_MODELS_V1.md"
        Reason = "Legacy model predates the canonical election/evidence split."
    },

    # -----------------------------------------------------------------------
    # Deployment history
    # -----------------------------------------------------------------------

    @{
        Source = "docs/DEPLOYMENT/OPERATIONS.md"
        Destination = "docs/implementation-history/operations/OPERATIONS_TEMPLATE_V1.md"
        Reason = "Historical operational template contains unsupported or aspirational procedures."
    },

    # -----------------------------------------------------------------------
    # Deployment security
    # -----------------------------------------------------------------------

    @{
        Source = "docs/DEPLOYMENT/CSP_SECURITY_MODEL.md"
        Destination = "docs/DEPLOYMENT/security/CSP_SECURITY_MODEL.md"
        Reason = "CSP belongs under deployment security."
    },

    @{
        Source = "docs/DEPLOYMENT/AZURE_CSP_DEPLOYMENT.md"
        Destination = "docs/DEPLOYMENT/security/CSP_DEPLOYMENT_CHECKLIST.md"
        Reason = "Azure CSP operational checklist belongs under deployment security."
    },

    # Normalize active deployment-security filenames after the directory move.

    @{
        Source = "docs/DEPLOYMENT/security/CSP_SECURITY_MODEL.md"
        Destination = "docs/DEPLOYMENT/security/csp_model.md"
        Reason = "Normalize active documentation filenames."
    },

    @{
        Source = "docs/DEPLOYMENT/security/CSP_DEPLOYMENT_CHECKLIST.md"
        Destination = "docs/DEPLOYMENT/security/csp_deployment_checklist.md"
        Reason = "Normalize active documentation filenames."
    },

    # Existing deployment SECURITY.md describes deployed/runtime implementation,
    # whereas root SECURITY.md defines repository-wide policy.

    @{
        Source = "docs/DEPLOYMENT/SECURITY.md"
        Destination = "docs/DEPLOYMENT/security/deployment_security.md"
        Reason = "Separate deployed security implementation from root security policy."
    },

    # -----------------------------------------------------------------------
    # CORE reference material
    # -----------------------------------------------------------------------

    @{
        Source = "docs/CORE/CONSTANTS.md"
        Destination = "docs/CORE/constants_reference.md"
        Reason = "Implementation reference rather than architecture."
    },

    # -----------------------------------------------------------------------
    # DEVELOPMENT generated reports
    # -----------------------------------------------------------------------

    @{
        Source = "docs/DEVELOPMENT/pipeline_map.md"
        Destination = "docs/DEVELOPMENT/generated/pipeline_map.md"
        Reason = "Generated repository snapshot; not architectural authority."
    },

    @{
        Source = "docs/DEVELOPMENT/project_audit.md"
        Destination = "docs/DEVELOPMENT/generated/project_audit.md"
        Reason = "Generated repository snapshot; not architectural authority."
    },

    @{
        Source = "docs/DEVELOPMENT/todos.md"
        Destination = "docs/DEVELOPMENT/generated/todos.md"
        Reason = "Generated source-marker report."
    },

    @{
        Source = "docs/DEVELOPMENT/todos_high.md"
        Destination = "docs/DEVELOPMENT/generated/todos_high.md"
        Reason = "Generated source-marker report."
    },

    @{
        Source = "docs/DEVELOPMENT/todos_medium.md"
        Destination = "docs/DEVELOPMENT/generated/todos_medium.md"
        Reason = "Generated source-marker report."
    },

    @{
        Source = "docs/DEVELOPMENT/todos_low.md"
        Destination = "docs/DEVELOPMENT/generated/todos_low.md"
        Reason = "Generated source-marker report."
    },

    # Manual explanation stays outside generated/.
    @{
        Source = "docs/DEVELOPMENT/TODOS_OVERVIEW.md"
        Destination = "docs/DEVELOPMENT/todos_overview.md"
        Reason = "Manual documentation describing generated TODO reports."
    },

    # -----------------------------------------------------------------------
    # Deployment filename normalization
    # -----------------------------------------------------------------------

    @{
        Source = "docs/DEPLOYMENT/CI_TOPOLOGY.md"
        Destination = "docs/DEPLOYMENT/ci_cd.md"
        Reason = "Authoritative CI/CD topology."
    },

    @{
        Source = "docs/DEPLOYMENT/DEPLOYMENT.md"
        Destination = "docs/DEPLOYMENT/deployment.md"
        Reason = "Normalize active deployment documentation filename."
    },

    @{
        Source = "docs/DEPLOYMENT/POST_DEPLOY_VERIFICATION.md"
        Destination = "docs/DEPLOYMENT/post_deploy_verification.md"
        Reason = "Normalize active deployment documentation filename."
    },

    # -----------------------------------------------------------------------
    # Governance policy and integrity guidance
    # -----------------------------------------------------------------------

    @{
        Source = "docs/GOVERNANCE/GOVERNANCE.md"
        Destination = "docs/implementation-history/2026-governance/GOVERNANCE_V1.md"
        Reason = "Legacy governance framework superseded by the authoritative governance domain index."
    },

    # -----------------------------------------------------------------------
    # Dynamic navigation / learning documentation set
    # -----------------------------------------------------------------------

    @{
        Source = "docs/DOCUMENTATION-INDEX.md"
        Destination = "docs/implementation-history/2026-navigation-learning/DOCUMENTATION_INDEX_V1.md"
        Reason = "Historical navigation-learning documentation index with session-specific production-readiness claims."
    },

    @{
        Source = "docs/QUICK-START.md"
        Destination = "docs/implementation-history/2026-navigation-learning/QUICK_START_V1.md"
        Reason = "Historical quick start for the JSONL navigation-recipe architecture."
    },

    @{
        Source = "docs/TECHNICAL-REFERENCE.md"
        Destination = "docs/implementation-history/2026-navigation-learning/TECHNICAL_REFERENCE_V1.md"
        Reason = "Historical API specification for registry, scaffold, navigation recipes, and legacy context persistence."
    },

    # -----------------------------------------------------------------------
    # State-handler implementation history
    # -----------------------------------------------------------------------

    @{
        Source = "docs/STATE_HANDLER_INTEGRATION.md"
        Destination = "docs/implementation-history/2026-state-handler-integration/STATE_HANDLER_INTEGRATION_V1.md"
        Reason = "Phase-specific implementation summary containing historical benchmarks, TODOs, and handler-generation assumptions."
    },

    # -----------------------------------------------------------------------
    # Former temporary audits and investigations
    # -----------------------------------------------------------------------

    @{
        Source = "docs/temp/CORE_INFRASTRUCTURE_GAP_AUDIT_2026-03-02.md"
        Destination = "docs/implementation-history/2026-audits/CORE_INFRASTRUCTURE_GAP_AUDIT_2026-03-02.md"
        Reason = "Dated infrastructure audit retained as historical repository evidence."
    },

    @{
        Source = "docs/temp/GITHUB_ACTIONS_RUNNER_SETUP_PROGRESS.md"
        Destination = "docs/implementation-history/2026-deployment/GITHUB_ACTIONS_RUNNER_SETUP_PROGRESS.md"
        Reason = "Historical GitHub Actions runner setup progress."
    },

    @{
        Source = "docs/temp/GOOGLE_SHEETS_500_ERROR_FIX.md"
        Destination = "docs/implementation-history/2026-incidents/GOOGLE_SHEETS_500_ERROR_FIX.md"
        Reason = "Resolved Google Sheets endpoint incident/fix record."
    },

    @{
        Source = "docs/temp/MULTI_PATHWAY_QUICK_REFERENCE.md"
        Destination = "docs/implementation-history/2026-ballot-lens/MULTI_PATHWAY_QUICK_REFERENCE.md"
        Reason = "Historical BallotLens multi-pathway testing reference."
    },

    @{
        Source = "docs/temp/README.md"
        Destination = "docs/implementation-history/temp-consolidation/TEMP_DOCUMENTATION_README_V1.md"
        Reason = "Former temporary-documentation directory guide retained for migration provenance."
    },

    @{
        Source = "docs/temp/REMOTE_PARITY_INVESTIGATION_2026-04-15.md"
        Destination = "docs/implementation-history/2026-deployment/REMOTE_PARITY_INVESTIGATION_2026-04-15.md"
        Reason = "Dated remote/local parity investigation retained as deployment history."
    },

    @{
        Source = "docs/temp/SELF_HOSTED_RUNNER_SETUP.md"
        Destination = "docs/implementation-history/2026-deployment/SELF_HOSTED_RUNNER_SETUP_V1.md"
        Reason = "Historical self-hosted runner design; current workflow uses GitHub-hosted runners."
    },

    # -----------------------------------------------------------------------
    # Remaining temporary documentation
    # -----------------------------------------------------------------------

    @{
        Source = "docs/temp/DATA_IMPORT_TEST_RUNBOOK.md"
        Destination = "docs/implementation-history/2026-data-import/DATA_IMPORT_TEST_RUNBOOK.md"
        Reason = "Historical data-import verification runbook retained for implementation provenance."
    },

    @{
        Source = "docs/temp/DELIVERY_SUMMARY.md"
        Destination = "docs/implementation-history/2026-status-reconciliation/DELIVERY_SUMMARY.md"
        Reason = "Historical delivery summary for the status-reconciliation subsystem."
    },

    @{
        Source = "docs/temp/FIX_COMPLETE_SUMMARY.md"
        Destination = "docs/implementation-history/2026-status-reconciliation/FIX_COMPLETE_SUMMARY.md"
        Reason = "Historical completion/fix summary associated with status reconciliation; retained pending redundancy review."
    },

    @{
        Source = "docs/temp/integrity_ui_integration_summary.md"
        Destination = "docs/implementation-history/2026-integrity-ui/INTEGRITY_UI_INTEGRATION_SUMMARY.md"
        Reason = "Completed integrity-monitoring UI integration summary retained as implementation history."
    },

    @{
        Source = "docs/temp/LIVE_VALIDATION_REPORT.md"
        Destination = "docs/implementation-history/2026-validation/LIVE_VALIDATION_REPORT.md"
        Reason = "Dated live-validation report retained as validation evidence rather than active documentation."
    },

    @{
        Source = "docs/temp/quality_dashboard_integrity_enhancement.md"
        Destination = "docs/implementation-history/2026-integrity-ui/QUALITY_DASHBOARD_INTEGRITY_ENHANCEMENT.md"
        Reason = "Completed quality-dashboard integrity enhancement retained as implementation history."
    },

    @{
        Source = "docs/temp/SESSION_WORK_COMPLETE.md"
        Destination = "docs/implementation-history/2026-ballot-lens/SESSION_WORK_COMPLETE.md"
        Reason = "Historical multi-pathway Ballot Lens session summary."
    },

    @{
        Source = "docs/temp/STATUS_IMPLEMENTATION_SUMMARY.md"
        Destination = "docs/implementation-history/2026-status-reconciliation/STATUS_IMPLEMENTATION_SUMMARY.md"
        Reason = "Historical implementation summary for the status-reconciliation subsystem."
    },

    @{
        Source = "docs/temp/STATUS_QUICK_START.md"
        Destination = "docs/implementation-history/2026-status-reconciliation/STATUS_QUICK_START.md"
        Reason = "Historical quick-start documentation for the status-reconciliation implementation."
    },

    @{
        Source = "docs/temp/STATUS_RECONCILIATION_GUIDE.md"
        Destination = "docs/implementation-history/2026-status-reconciliation/STATUS_RECONCILIATION_GUIDE.md"
        Reason = "Historical status-reconciliation subsystem guide."
    },

    @{
        Source = "docs/temp/URL_STATUS_SYSTEM_REFERENCE.md"
        Destination = "docs/implementation-history/2026-status-reconciliation/URL_STATUS_SYSTEM_REFERENCE.md"
        Reason = "Historical URL status/dashboard reference associated with status reconciliation."
    },

    # -----------------------------------------------------------------------
    # FEATURES domain cleanup
    # -----------------------------------------------------------------------

    @{
        Source = "docs/FEATURES/BALLOT_LENS_PATHWAYS.md"
        Destination = "docs/implementation-history/2026-ballot-lens/BALLOT_LENS_PATHWAYS_V1.md"
        Reason = "Historical Ballot Lens multi-pathway testing guide retained as implementation and validation history."
    },

    @{
        Source = "docs/FEATURES/MULTI_PATHWAY_SUMMARY.md"
        Destination = "docs/implementation-history/2026-ballot-lens/MULTI_PATHWAY_SUMMARY_V1.md"
        Reason = "Historical Ballot Lens integration and validation summary."
    },

    @{
        Source = "docs/FEATURES/ML_TRAINING_ENHANCEMENTS.md"
        Destination = "docs/implementation-history/2026-ml-training/ML_TRAINING_ENHANCEMENTS_V1.md"
        Reason = "Historical ML training enhancement setup and implementation documentation."
    },

    @{
        Source = "docs/FEATURES/NLP_ML_TRAINING_ASSESSMENT.md"
        Destination = "docs/implementation-history/2026-ml-training/NLP_ML_TRAINING_ASSESSMENT_V1.md"
        Reason = "Historical NLP and ML training readiness assessment containing implementation-state and roadmap claims."
    },

    @{
        Source = "docs/FEATURES/PARSER_ROUTER_INTEGRATION.md"
        Destination = "docs/implementation-history/2026-parser-integration/PARSER_ROUTER_INTEGRATION_V1.md"
        Reason = "Historical parser router and format-detection integration summary."
    },

    @{
        Source = "docs/FEATURES/PARSER_VALIDATION_SUMMARY.md"
        Destination = "docs/implementation-history/2026-parser-integration/PARSER_VALIDATION_SUMMARY_V1.md"
        Reason = "Historical parser validation and integration summary containing implementation completion claims."
    },

    @{
        Source = "docs/FEATURES/SELENIUM_NLP_INTEGRATION.md"
        Destination = "docs/implementation-history/2026-parser-integration/SELENIUM_NLP_INTEGRATION_V1.md"
        Reason = "Historical Selenium and NLP integration strategy tied to earlier parser and context-system implementation."
    },

    @{
        Source = "docs/FEATURES/URL_PARSER_TRAINING.md"
        Destination = "docs/implementation-history/2026-url-training/URL_PARSER_TRAINING_V1.md"
        Reason = "Historical URL parsing and training-data implementation guide."
    },

    # -----------------------------------------------------------------------
    # FEATURES promoted into active domains
    # -----------------------------------------------------------------------

    @{
        Source = "docs/FEATURES/CONFIDENCE_FRAMEWORK.md"
        Destination = "docs/QUALITY/confidence_framework.md"
        Reason = "Confidence scoring belongs to validation, review, and quality rather than user-facing feature documentation."
    },

    @{
        Source = "docs/FEATURES/INTEGRITY_MONITORING.md"
        Destination = "docs/QUALITY/integrity_monitoring.md"
        Reason = "Integrity monitoring and drift detection belong to the quality and review domain."
    },

    @{
        Source = "docs/FEATURES/INTEGRITY_GUIDELINES.md"
        Destination = "docs/GOVERNANCE/integrity_guidelines.md"
        Reason = "Election-integrity principles and responsible-use guidance belong to governance."
    },

    @{
        Source = "docs/FEATURES/GOOGLE_SHEETS_CREDENTIALS.md"
        Destination = "docs/DEVELOPMENT/google_sheets_credentials.md"
        Reason = "Google Sheets credential loading is developer and configuration guidance rather than a user-facing feature."
    },

    @{
        Source = "docs/FEATURES/GUIDES.md"
        Destination = "docs/DEVELOPMENT/guides.md"
        Reason = "Handler development and contributor guidance belongs to the development domain."
    },

    @{
        Source = "docs/FEATURES/ELECTION_OPERATIONS.md"
        Destination = "docs/DEPLOYMENT/election_operations.md"
        Reason = "Election operations guidance is operational documentation rather than a product feature."
    },

    # -----------------------------------------------------------------------
    # QUALITY domain cleanup
    # -----------------------------------------------------------------------

    @{
        Source = "docs/QUALITY/DATA_COMPARISON_ROADMAP.md"
        Destination = "docs/implementation-history/2026-data-comparison/DATA_COMPARISON_ROADMAP_V1.md"
        Reason = "Historical data-comparison roadmap containing dated implementation plans and TODOs."
    },

    @{
        Source = "docs/QUALITY/GOOGLE_SHEETS_MIGRATION.md"
        Destination = "docs/implementation-history/2026-google-sheets/GOOGLE_SHEETS_MIGRATION_V1.md"
        Reason = "Historical Google Sheets to local-database migration strategy retained as implementation provenance."
    },

    # -----------------------------------------------------------------------
    # QUALITY active filename normalization
    # -----------------------------------------------------------------------

    @{
        Source = "docs/QUALITY/VERIFICATION.md"
        Destination = "docs/QUALITY/verification.md"
        Reason = "Normalize the active verification contract filename for the authoritative QUALITY documentation set."
    },

    @{
        Source = "docs/QUALITY/QUARANTINE_SYSTEM.md"
        Destination = "docs/QUALITY/quarantine_system.md"
        Reason = "Normalize the active quarantine contract filename for the authoritative QUALITY documentation set."
    },

    @{
        Source = "docs/QUALITY/ML_FRAMEWORK.md"
        Destination = "docs/QUALITY/ml_quality.md"
        Reason = "Rename the legacy ML framework document to reflect its quality-assurance responsibility."
    }
)

function Get-PlannedDocumentationPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    $normalized = ConvertTo-RepoPath $RelativePath

    foreach ($item in $KnownMoves) {
        $source = ConvertTo-RepoPath $item.Source

        if (
            $normalized.Equals(
                $source,
                [System.StringComparison]::OrdinalIgnoreCase
            )
        ) {
            return ConvertTo-RepoPath $item.Destination
        }
    }

    return $normalized
}
foreach ($item in $KnownMoves) {
    Write-Host ""
    Write-Host "Reason: $($item.Reason)"

    Move-RepositoryFile `
        -Source $item.Source `
        -Destination $item.Destination
}

# ---------------------------------------------------------------------------
# Explicit delete manifest
# ---------------------------------------------------------------------------
#
# Intentionally empty.
#
# Documentation should NEVER be deleted simply because a filename resembles
# "old", "final", "complete", etc.
#
# Add entries here only after manual review proves the document contains
# nothing worth retaining in implementation-history or archived/.
# ---------------------------------------------------------------------------

$DeleteManifest = @()

if ($DeleteManifest.Count -gt 0) {
    Write-Section "Explicit Documentation Removals"

    foreach ($relativePath in $DeleteManifest) {
        $normalized = ConvertTo-RepoPath $relativePath
        $absolute = Get-AbsoluteRepoPath $normalized

        if (-not (Test-Path -LiteralPath $absolute)) {
            Write-Host "[SKIP] Missing: $normalized"
            continue
        }

        Write-Action "Remove documentation file: $normalized"

        if ($Apply) {
            if (Test-GitTracked $normalized) {
                & git rm -- $normalized

                if ($LASTEXITCODE -ne 0) {
                    throw "git rm failed: $normalized"
                }
            }
            else {
                Remove-Item -LiteralPath $absolute
            }
        }
    }
}

# ---------------------------------------------------------------------------
# Known link rewrite map
# ---------------------------------------------------------------------------

$KnownPathReplacements = [ordered]@{

    "docs/CORE/CONSTANTS.md" =
        "docs/CORE/constants_reference.md"

    "docs/DEPLOYMENT/CSP_SECURITY_MODEL.md" =
        "docs/DEPLOYMENT/security/csp_model.md"

    "docs/DEPLOYMENT/AZURE_CSP_DEPLOYMENT.md" =
        "docs/DEPLOYMENT/security/csp_deployment_checklist.md"

    "docs/DEPLOYMENT/SECURITY.md" =
        "docs/DEPLOYMENT/security/deployment_security.md"

    "docs/DEPLOYMENT/CI_TOPOLOGY.md" =
        "docs/DEPLOYMENT/ci_cd.md"

    "docs/DEPLOYMENT/DEPLOYMENT.md" =
        "docs/DEPLOYMENT/deployment.md"

    "docs/DEPLOYMENT/POST_DEPLOY_VERIFICATION.md" =
        "docs/DEPLOYMENT/post_deploy_verification.md"

    "docs/DEVELOPMENT/pipeline_map.md" =
        "docs/DEVELOPMENT/generated/pipeline_map.md"

    "docs/DEVELOPMENT/project_audit.md" =
        "docs/DEVELOPMENT/generated/project_audit.md"

    "docs/DEVELOPMENT/todos.md" =
        "docs/DEVELOPMENT/generated/todos.md"

    "docs/DEVELOPMENT/todos_high.md" =
        "docs/DEVELOPMENT/generated/todos_high.md"

    "docs/DEVELOPMENT/todos_medium.md" =
        "docs/DEVELOPMENT/generated/todos_medium.md"

    "docs/DEVELOPMENT/todos_low.md" =
        "docs/DEVELOPMENT/generated/todos_low.md"

    "docs/DEVELOPMENT/TODOS_OVERVIEW.md" =
        "docs/DEVELOPMENT/todos_overview.md"

    "docs/FEATURES/CONFIDENCE_FRAMEWORK.md" =
        "docs/QUALITY/confidence_framework.md"

    "docs/FEATURES/INTEGRITY_MONITORING.md" =
        "docs/QUALITY/integrity_monitoring.md"

    "docs/FEATURES/INTEGRITY_GUIDELINES.md" =
        "docs/GOVERNANCE/integrity_guidelines.md"

    "docs/FEATURES/GOOGLE_SHEETS_CREDENTIALS.md" =
        "docs/DEVELOPMENT/google_sheets_credentials.md"

    "docs/FEATURES/GUIDES.md" =
        "docs/DEVELOPMENT/guides.md"

    "docs/FEATURES/ELECTION_OPERATIONS.md" =
        "docs/DEPLOYMENT/election_operations.md"
    "docs/QUALITY/VERIFICATION.md" =
        "docs/QUALITY/verification.md"

    "docs/QUALITY/QUARANTINE_SYSTEM.md" =
        "docs/QUALITY/quarantine_system.md"

    "docs/QUALITY/ML_FRAMEWORK.md" =
        "docs/QUALITY/ml_quality.md"

    "docs/GOVERNANCE/GOVERNANCE.md" =
        "docs/GOVERNANCE/README.md"
}

function Test-ActiveDocumentationPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FullName
    )

    $relative = Get-RepositoryRelativePath $FullName

    $excludedPrefixes = @(
        "docs/implementation-history/",
        "docs/archived/",
        "docs/session-logs/",
        "docs/temp/",
        "docs/DEVELOPMENT/generated/"
    )

    foreach ($prefix in $excludedPrefixes) {
        if ($relative.StartsWith(
            $prefix,
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
            return $false
        }
    }

    return $true
}

function Update-KnownDocumentLinks {

    Write-Section "Known Link Rewrites"

    $candidateFiles = @()

    $rootFiles = @(
        "README.md",
        "CONTRIBUTING.md",
        "SECURITY.md"
    )

    foreach ($rootFile in $rootFiles) {
        $absolute = Join-Path $RepoRoot $rootFile

        if (Test-Path -LiteralPath $absolute) {
            $candidateFiles += Get-Item -LiteralPath $absolute
        }
    }

    $candidateFiles += Get-ChildItem `
        -LiteralPath $DocsRoot `
        -Recurse `
        -File |
        Where-Object {
            $_.Extension -in @(
                ".md",
                ".html",
                ".yml",
                ".yaml"
            )
        } |
        Where-Object {
            Test-ActiveDocumentationPath $_.FullName
        }

    $candidateFiles = $candidateFiles |
        Sort-Object FullName -Unique

    foreach ($file in $candidateFiles) {

        $content = [System.IO.File]::ReadAllText($file.FullName)

        if ($null -eq $content) {
            $content = ""
        }

        $updated = $content
        $changes = 0

        foreach ($oldPath in $KnownPathReplacements.Keys) {
            $newPath = $KnownPathReplacements[$oldPath]

            if ($updated.Contains($oldPath)) {
                $updated = $updated.Replace($oldPath, $newPath)
                $changes++
            }
        }

        if ($changes -eq 0) {
            continue
        }

        $relative = $file.FullName.Replace(
            $RepoRoot.TrimEnd("\") + "\",
            ""
        )

        Write-Action "Rewrite $changes known path reference(s): $relative"

        if ($Apply) {
            [System.IO.File]::WriteAllText(
                $file.FullName,
                $updated,
                $Utf8NoBom
            )
        }
    }
}

if ($RewriteKnownLinks) {
    Update-KnownDocumentLinks
}

# ---------------------------------------------------------------------------
# Review classification
# ---------------------------------------------------------------------------
#
# This is intentionally REPORT-ONLY.
#
# It helps identify likely domains without moving files automatically.
# ---------------------------------------------------------------------------

function Get-DocumentationClassification {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    $normalized = ConvertTo-RepoPath $RelativePath
    $name = [System.IO.Path]::GetFileNameWithoutExtension($normalized)
    $probe = ($normalized + " " + $name).ToLowerInvariant()

    $domain = "REVIEW"
    $disposition = "REVIEW"
    $reason = "No explicit disposition rule exists yet."
    if ($normalized -eq "docs/index.md") {
        $domain = "DOCS-HOME"
        $disposition = "KEEP/REWRITE"
        $reason = "GitHub Pages/Jekyll documentation landing page."
    }
    elseif ($normalized -eq "docs/README.md") {
        $domain = "DOCS-HOME"
        $disposition = "KEEP/REWRITE"
        $reason = "Documentation repository guide and navigation companion."
    }
    elseif ($normalized -match "^docs/ARCHITECTURE/") {
        $domain = "ARCHITECTURE"
        $disposition = "KEEP/REWRITE"
        $reason = "Already located in the architecture domain."
    }
    elseif ($normalized -match "^docs/CORE/") {
        $domain = "CORE"
        $disposition = "KEEP/VERIFY"
        $reason = "Current implementation contract/reference."
    }
    elseif ($normalized -match "^docs/DEVELOPMENT/generated/") {
        $domain = "DEVELOPMENT"
        $disposition = "GENERATED"
        $reason = "Machine-generated repository evidence."
    }
    elseif ($normalized -match "^docs/DEVELOPMENT/") {
        $domain = "DEVELOPMENT"
        $disposition = "KEEP/REVIEW"
        $reason = "Contributor/development documentation."
    }
    elseif ($normalized -match "^docs/DEPLOYMENT/") {
        $domain = "DEPLOYMENT"
        $disposition = "KEEP/REVIEW"
        $reason = "Deployment/runtime documentation."
    }
    elseif ($normalized -match "^docs/QUALITY/") {
        $domain = "QUALITY"
        $disposition = "KEEP/REVIEW"
        $reason = "Validation/integrity documentation."
    }
    elseif ($normalized -match "^docs/GOVERNANCE/") {
        $domain = "GOVERNANCE"
        $disposition = "KEEP/REVIEW"
        $reason = "Policy/governance documentation."
    }
    elseif ($normalized -match "^docs/FEATURES/") {
        $domain = "FEATURES"
        $disposition = "REVIEW"
        $reason = "Feature documentation should remain only if user-facing and current."
    }
    elseif ($normalized -match "^docs/implementation-history/") {
        $domain = "HISTORY"
        $disposition = "HISTORICAL"
        $reason = "Preserved implementation history."
    }
    elseif ($normalized -match "^docs/implementation-phases/") {
        $domain = "PHASES"
        $disposition = "CURRENT/PLANNED"
        $reason = "Current or planned implementation work."
    }
    elseif ($normalized -match "^docs/archived/") {
        $domain = "ARCHIVE"
        $disposition = "ARCHIVED"
        $reason = "Retained without current authority."
    }
    elseif ($normalized -match "^docs/session-logs/") {
        $domain = "SESSION-LOGS"
        $disposition = "HISTORICAL"
        $reason = "Chronological working record."
    }
    elseif (
        $probe -match
        "architecture|canonical|evidence.model|context.system|parser.pipeline|storage.architecture|automation"
    ) {
        $domain = "ARCHITECTURE"
        $disposition = "REVIEW FOR MERGE"
        $reason = "Filename/content naming suggests architectural responsibility."
    }
    elseif (
        $probe -match
        "risk|confidence|quality|quarantine|integrity|reconciliation|validation|assurance"
    ) {
        $domain = "QUALITY"
        $disposition = "REVIEW FOR MOVE/MERGE"
        $reason = "Appears related to validation, integrity, confidence, or review."
    }
    elseif (
        $probe -match
        "deploy|azure|docker|csp|ci|workflow|production"
    ) {
        $domain = "DEPLOYMENT"
        $disposition = "REVIEW FOR MOVE/MERGE"
        $reason = "Appears related to deployment or runtime infrastructure."
    }
    elseif (
        $probe -match
        "test|debug|development|developer|contribut|script"
    ) {
        $domain = "DEVELOPMENT"
        $disposition = "REVIEW FOR MOVE/MERGE"
        $reason = "Appears contributor/developer oriented."
    }
    elseif (
        $probe -match
        "phase|implementation|complete|summary|status|migration"
    ) {
        $domain = "HISTORY"
        $disposition = "REVIEW FOR HISTORY"
        $reason = "Filename suggests implementation status/history rather than durable architecture."
    }

    return [PSCustomObject]@{
        Domain = $domain
        Disposition = $disposition
        Reason = $reason
    }
}

# ---------------------------------------------------------------------------
# Produce review manifest
# ---------------------------------------------------------------------------

Write-Section "Documentation Review Manifest"

if ($Apply) {
    New-Item `
        -ItemType Directory `
        -Force `
        -Path $ReportRoot |
        Out-Null
}
elseif (-not (Test-Path -LiteralPath $ReportRoot)) {
    # Reports are useful even in dry-run mode.
    New-Item `
        -ItemType Directory `
        -Force `
        -Path $ReportRoot |
        Out-Null
}

$MarkdownFiles = Get-ChildItem `
    -LiteralPath $DocsRoot `
    -Recurse `
    -File `
    -Filter "*.md" |
    Sort-Object FullName

$ReviewRows = foreach ($file in $MarkdownFiles) {

    $relative = Get-RepositoryRelativePath $file.FullName
    $effectiveRelative = Get-PlannedDocumentationPath $relative

    $classification = Get-DocumentationClassification $effectiveRelative

    $content = [System.IO.File]::ReadAllText($file.FullName)

    if ($null -eq $content) {
        $content = ""
    }

    $lineCount = if ($content.Length -eq 0) {
        0
    }
    else {
        ($content -split "\r?\n").Count
    }

    $firstHeading = ""

    $headingMatch = [regex]::Match(
        $content,
        "(?m)^#\s+(.+)$"
    )

    if ($headingMatch.Success) {
        $firstHeading = $headingMatch.Groups[1].Value.Trim()
    }

    $currentDomain = "ROOT-DOCS"

    if ($relative -match "^docs/([^/]+)/") {
        $currentDomain = $Matches[1]
    }

    [PSCustomObject]@{
        RelativePath = $relative
        PlannedPath = $effectiveRelative
        CurrentDomain = $currentDomain
        PlannedDomain = $plannedDomain
        SuggestedDomain = $classification.Domain
        SuggestedDisposition = $classification.Disposition
        Reason = $classification.Reason
        Title = $firstHeading
        Lines = $lineCount
        Bytes = $file.Length
    }
}

$ReviewManifestPath = Join-Path $ReportRoot "review_manifest.csv"

$ReviewRows |
    Export-Csv `
        -Path $ReviewManifestPath `
        -NoTypeInformation `
        -Encoding UTF8

Write-Host "Review manifest:"
Write-Host "  $ReviewManifestPath"

# ---------------------------------------------------------------------------
# Domain summary
# ---------------------------------------------------------------------------

$DomainSummary = $ReviewRows |
    Group-Object SuggestedDomain |
    Sort-Object Name |
    ForEach-Object {

        [PSCustomObject]@{
            Domain = $_.Name
            Documents = $_.Count
            TotalLines = (
                $_.Group |
                Measure-Object Lines -Sum
            ).Sum
        }
    }

$DomainSummaryPath = Join-Path $ReportRoot "domain_summary.csv"

$DomainSummary |
    Export-Csv `
        -Path $DomainSummaryPath `
        -NoTypeInformation `
        -Encoding UTF8

Write-Host ""
Write-Host "Domain summary:"
Write-Host "  $DomainSummaryPath"

# ---------------------------------------------------------------------------
# Generate target architecture map
# ---------------------------------------------------------------------------

$ArchitectureMapPath = Join-Path $ReportRoot "target_structure.txt"

@"
Election Pulse Documentation Authority Model
=============================================

ROOT
----
README.md
    What Election Pulse is and why it exists.

CONTRIBUTING.md
    How Election Pulse should be changed safely.

SECURITY.md
    Repository-wide security policy and reporting.

LICENSE
    Legal terms.


docs/index.md
-------------
GitHub Pages documentation landing page.

Provides concise navigation by domain and audience.
It does not define technical contracts.


docs/README.md
--------------
Documentation guide for contributors and maintainers.

Explains documentation authority, organization, and where new
documentation belongs.

docs/ARCHITECTURE
-----------------
Durable system boundaries and domain contracts.

README.md
system_overview.md
parser_pipeline.md
canonical_election_model.md
evidence_model.md
context_system.md
storage_architecture.md
automation.md


docs/CORE
---------
Contracts the current implementation actually follows.

README.md
implemented_contracts.md
constants_reference.md


docs/FEATURES
-------------
Current user-facing or operator-facing capabilities.

Feature documents describe what users can do.
They do not define architecture.


docs/QUALITY
------------
Validation and election-data assurance.

validation.md
integrity_monitoring.md
risk_assessment.md
quarantine.md


docs/DEPLOYMENT
---------------
Deployment, cloud runtime, CI/CD, and post-deployment verification.

deployment.md
ci_cd.md
post_deploy_verification.md

security/
    deployment_security.md
    csp_model.md
    csp_deployment_checklist.md


docs/DEVELOPMENT
----------------
Contributor workflow and repository development.

README.md
testing.md
debugging.md
repository_structure.md
todos_overview.md

generated/
    pipeline_map.md
    project_audit.md
    todos.md
    todos_high.md
    todos_medium.md
    todos_low.md


docs/GOVERNANCE
---------------
Project integrity principles, provenance policy, responsible use,
and architectural/project decisions.

README.md
data_provenance.md
responsible_use.md

decision-records/


docs/implementation-phases
--------------------------
Current and planned work.

README.md
current_phase.md


docs/implementation-history
---------------------------
Completed and superseded implementation records.

Historical documents remain useful evidence but are not current
architecture contracts.


docs/archived
-------------
Material retained only for traceability.


docs/session-logs
-----------------
Chronological working records.


AUTHORITY ORDER
---------------

1. Current source code
2. docs/CORE implemented contracts
3. docs/ARCHITECTURE intended boundaries
4. Active domain documentation
5. Current implementation phase
6. Implementation history
7. Archived/session material

Generated reports provide evidence about the repository but never
override architecture or implemented contracts.


CORE SEPARATIONS
----------------

Evidence != Knowledge

Parser output != Canonical data

Runtime state != Durable election records

Source trust != Data verification

Logging event != Presentation transport

CLI != Parser engine

Web UI != Parser engine

Application orchestration != Parse orchestration

Parse orchestration != Election-domain logic

Safeguard policy != Business logic
"@ |
    Set-Content `
        -Path $ArchitectureMapPath `
        -Encoding UTF8

Write-Host ""
Write-Host "Target structure:"
Write-Host "  $ArchitectureMapPath"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if ($Validate) {

    Write-Section "Validation"

    Write-Host ""
    Write-Host "[1/4] Git whitespace validation"

    & git diff --check

    if ($LASTEXITCODE -ne 0) {
        Write-Warning "git diff --check reported problems."
    }
    else {
        Write-Host "[OK] git diff --check"
    }

    Write-Host ""
    Write-Host "[2/4] Documentation audit"

    $AuditScript = Join-Path `
        $RepoRoot `
        "scripts\maintenance\audit_docs.ps1"

    if (Test-Path -LiteralPath $AuditScript) {
        & $AuditScript
    }
    else {
        Write-Warning "audit_docs.ps1 not found."
    }

    Write-Host ""
    Write-Host "[3/4] Markdown lint"

    $PackageJson = Join-Path $RepoRoot "package.json"

    if (
        (Test-Path -LiteralPath $PackageJson) -and
        (Get-Command npm -ErrorAction SilentlyContinue)
    ) {
        $packageContent = [System.IO.File]::ReadAllText($PackageJson)

        if ($packageContent -match '"lint:md"\s*:') {
            & npm run lint:md

            if ($LASTEXITCODE -ne 0) {
                Write-Warning "Markdown lint reported problems."
            }
        }
        else {
            Write-Host "[SKIP] package.json has no lint:md script."
        }
    }
    else {
        Write-Host "[SKIP] npm/package.json unavailable."
    }

    Write-Host ""
    Write-Host "[4/4] Git status"

    & git status --short
}

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

Write-Section "Reorganization Summary"

if ($Apply) {
    Write-Host "Mode: APPLY"
}
else {
    Write-Host "Mode: DRY RUN"
}

Write-Host ""
Write-Host "Known documentation moves: $($KnownMoves.Count)"
Write-Host "Automatic deletions:      $($DeleteManifest.Count)"
Write-Host "Markdown documents found: $($ReviewRows.Count)"

Write-Host ""
Write-Host "Reports:"
Write-Host "  $ReviewManifestPath"
Write-Host "  $DomainSummaryPath"
Write-Host "  $ArchitectureMapPath"

Write-Host ""
Write-Host "Suggested-domain counts:"
$DomainSummary | Format-Table -AutoSize

if (-not $Apply) {
    Write-Host ""
    Write-Host "No repository files were moved."
    Write-Host ""
    Write-Host "Review the output, then apply with:"
    Write-Host ""
    Write-Host "  & .\scripts\maintenance\reorganize_docs.ps1 -Apply"
}

if ($Apply -and -not $RewriteKnownLinks) {
    Write-Host ""
    Write-Host "Known links were NOT rewritten."
    Write-Host "When ready:"
    Write-Host ""
    Write-Host "  & .\scripts\maintenance\reorganize_docs.ps1 -Apply -RewriteKnownLinks"
}

Write-Host ""
Write-Host "The script does not commit changes."
Write-Host "Review the staged and unstaged diff before committing."

}
finally {
    Pop-Location
}
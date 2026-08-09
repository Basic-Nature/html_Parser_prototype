param(
    [string]$DocsRoot = "docs",
    [string]$OutputRoot = "output\reports\docs-audit"
)

$ErrorActionPreference = "Stop"

function Get-TextContent {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $content = Get-Content -LiteralPath $Path -Raw -ErrorAction Stop

    if ($null -eq $content) {
        return ""
    }

    return [string]$content
}

function Get-DocumentationAuthority {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    $path = $RelativePath.Replace("\", "/")

    if (
        $path -eq "docs/index.md" -or
        $path -eq "docs/README.md"
    ) {
        return [PSCustomObject]@{
            Authority = "ACTIVE"
            Actionable = $true
            Domain = "DOCS-HOME"
        }
    }

    if ($path.StartsWith(
        "docs/DEVELOPMENT/generated/",
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        return [PSCustomObject]@{
            Authority = "GENERATED"
            Actionable = $false
            Domain = "DEVELOPMENT"
        }
    }

    if ($path.StartsWith(
        "docs/implementation-history/",
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        return [PSCustomObject]@{
            Authority = "HISTORY"
            Actionable = $false
            Domain = "IMPLEMENTATION-HISTORY"
        }
    }

    if ($path.StartsWith(
        "docs/archived/",
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        return [PSCustomObject]@{
            Authority = "ARCHIVE"
            Actionable = $false
            Domain = "ARCHIVE"
        }
    }

    if ($path.StartsWith(
        "docs/session-logs/",
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        return [PSCustomObject]@{
            Authority = "SESSION"
            Actionable = $false
            Domain = "SESSION-LOGS"
        }
    }

    if ($path.StartsWith(
        "docs/implementation-phases/",
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        return [PSCustomObject]@{
            Authority = "PHASE"
            Actionable = $true
            Domain = "IMPLEMENTATION-PHASES"
        }
    }

    if (
        $path.StartsWith("docs/ARCHITECTURE/", [System.StringComparison]::OrdinalIgnoreCase) -or
        $path.StartsWith("docs/CORE/", [System.StringComparison]::OrdinalIgnoreCase) -or
        $path.StartsWith("docs/FEATURES/", [System.StringComparison]::OrdinalIgnoreCase) -or
        $path.StartsWith("docs/QUALITY/", [System.StringComparison]::OrdinalIgnoreCase) -or
        $path.StartsWith("docs/DEPLOYMENT/", [System.StringComparison]::OrdinalIgnoreCase) -or
        $path.StartsWith("docs/DEVELOPMENT/", [System.StringComparison]::OrdinalIgnoreCase) -or
        $path.StartsWith("docs/GOVERNANCE/", [System.StringComparison]::OrdinalIgnoreCase)
    ) {
        $parts = $path -split "/"

        return [PSCustomObject]@{
            Authority = "ACTIVE"
            Actionable = $true
            Domain = if ($parts.Count -gt 1) {
                $parts[1].ToUpperInvariant()
            } else {
                "DOCS"
            }
        }
    }

    return [PSCustomObject]@{
        Authority = "UNCLASSIFIED"
        Actionable = $true
        Domain = "UNCLASSIFIED"
    }
}

function Resolve-DocumentationLink {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourceFile,

        [Parameter(Mandatory = $true)]
        [string]$Target
    )

    $rawTarget = $Target.Trim()

    # Remove surrounding angle brackets used by some Markdown links.
    if (
        $rawTarget.StartsWith("<") -and
        $rawTarget.EndsWith(">")
    ) {
        $rawTarget = $rawTarget.Substring(
            1,
            $rawTarget.Length - 2
        )
    }

    # Strip fragment and query string for filesystem checking.
    $pathOnly = ($rawTarget -split "#", 2)[0]
    $pathOnly = ($pathOnly -split "\?", 2)[0]

    if ([string]::IsNullOrWhiteSpace($pathOnly)) {
        return [PSCustomObject]@{
            Exists = $true
            ResolvedPath = ""
            Resolution = "ANCHOR_ONLY"
        }
    }

    # Decode URL-encoded spaces/path characters when possible.
    try {
        $pathOnly = [System.Uri]::UnescapeDataString($pathOnly)
    }
    catch {
        # Keep original value if decoding fails.
    }

    $sourceDirectory = Split-Path $SourceFile -Parent

    if ($pathOnly.StartsWith("/")) {
        $candidate = Join-Path `
            $RepoRoot `
            $pathOnly.TrimStart("/")
    }
    else {
        $candidate = Join-Path `
            $sourceDirectory `
            $pathOnly
    }

    $candidate = [System.IO.Path]::GetFullPath($candidate)

    # Exact target exists.
    if (Test-Path -LiteralPath $candidate) {
        return [PSCustomObject]@{
            Exists = $true
            ResolvedPath = $candidate
            Resolution = "EXACT"
        }
    }

    # GitHub Pages/Jekyll commonly links foo.html while source is foo.md.
    if (
        [System.IO.Path]::GetExtension($candidate) -ieq ".html"
    ) {
        $markdownCandidate = [System.IO.Path]::ChangeExtension(
            $candidate,
            ".md"
        )

        if (Test-Path -LiteralPath $markdownCandidate) {
            return [PSCustomObject]@{
                Exists = $true
                ResolvedPath = $markdownCandidate
                Resolution = "JEKYLL_HTML_TO_MD"
            }
        }
    }

    # Directory-style links may resolve to index.md.
    if (-not [System.IO.Path]::GetExtension($candidate)) {
        $indexCandidate = Join-Path $candidate "index.md"

        if (Test-Path -LiteralPath $indexCandidate) {
            return [PSCustomObject]@{
                Exists = $true
                ResolvedPath = $indexCandidate
                Resolution = "DIRECTORY_INDEX"
            }
        }

        $readmeCandidate = Join-Path $candidate "README.md"

        if (Test-Path -LiteralPath $readmeCandidate) {
            return [PSCustomObject]@{
                Exists = $true
                ResolvedPath = $readmeCandidate
                Resolution = "DIRECTORY_README"
            }
        }
    }

    return [PSCustomObject]@{
        Exists = $false
        ResolvedPath = $candidate
        Resolution = "MISSING"
    }
}

$repoRoot = (Resolve-Path ".").Path
$docsPath = (Resolve-Path $DocsRoot).Path
$outputPath = Join-Path $repoRoot $OutputRoot

New-Item -ItemType Directory -Force $outputPath | Out-Null

$markdownFiles = Get-ChildItem $docsPath -Recurse -File -Filter *.md |
    Sort-Object FullName

# ---------------------------------------------------------------------------
# Full inventory
# ---------------------------------------------------------------------------

$inventory = foreach ($file in $markdownFiles) {
    $relativePath = $file.FullName.Replace($repoRoot + "\", "")
    $content = Get-TextContent -Path $file.FullName

    if ($null -eq $content) {
        $content = ""
    }

    $lines = @(Get-Content $file.FullName)

    $titleMatch = $lines |
        Where-Object { $_ -match "^#\s+" } |
        Select-Object -First 1

    $frontMatterTitle = $null
    if ($content -match "(?ms)^---\s*.*?^title:\s*(.+?)\s*$.*?^---") {
        $frontMatterTitle = $Matches[1].Trim()
    }

    [PSCustomObject]@{
        RelativePath = $relativePath
        FileName = $file.Name
        Directory = Split-Path $relativePath -Parent
        Length = $file.Length
        Lines = $lines.Count
        HeadingTitle = if ($titleMatch) {
            ($titleMatch -replace "^#\s+", "").Trim()
        } else {
            ""
        }
        FrontMatterTitle = if ($frontMatterTitle) {
            $frontMatterTitle
        } else {
            ""
        }
        Modified = $file.LastWriteTime
    }
}

$inventory |
    Export-Csv `
        (Join-Path $outputPath "inventory.csv") `
        -NoTypeInformation `
        -Encoding utf8

# ---------------------------------------------------------------------------
# Titles and headings
# ---------------------------------------------------------------------------

$headingRows = foreach ($file in $markdownFiles) {
    $relativePath = $file.FullName.Replace($repoRoot + "\", "")

    Select-String `
        -Path $file.FullName `
        -Pattern "^(#{1,6})\s+(.+)$" |
        ForEach-Object {
            [PSCustomObject]@{
                RelativePath = $relativePath
                LineNumber = $_.LineNumber
                Level = $_.Matches[0].Groups[1].Value.Length
                Heading = $_.Matches[0].Groups[2].Value.Trim()
            }
        }
}

$headingRows |
    Export-Csv `
        (Join-Path $outputPath "headings.csv") `
        -NoTypeInformation `
        -Encoding utf8

# ---------------------------------------------------------------------------
# Stale or temporary claims
# ---------------------------------------------------------------------------

$stalePatterns = @(
    "Production Ready",
    "Production-Ready",
    "All Tests Pass",
    "All tests passed",
    "Current Session",
    "CURRENT SESSION",
    "Phase 1 Complete",
    "Phase 2 Complete",
    "Phase 3 Complete",
    "PHASE 1 COMPLETE",
    "PHASE 2 COMPLETE",
    "PHASE 3 COMPLETE",
    "Final Summary",
    "FINAL SUMMARY",
    "100% Complete",
    "Last Updated",
    "Current Status",
    "TBD",
    "TODO",
    "FIXME"
)

$staleRows = foreach ($file in $markdownFiles) {
    $relativePath = $file.FullName.Replace($repoRoot + "\", "")
    $authority = Get-DocumentationAuthority $relativePath

    Select-String `
        -Path $file.FullName `
        -Pattern $stalePatterns `
        -SimpleMatch |
        ForEach-Object {
            [PSCustomObject]@{
                RelativePath = $relativePath.Replace("\", "/")
                LineNumber = $_.LineNumber
                Text = $_.Line.Trim()
                Authority = $authority.Authority
                Domain = $authority.Domain
                Actionable = $authority.Actionable
            }
        }
}

$staleRows |
    Export-Csv `
        (Join-Path $outputPath "stale_claims.csv") `
        -NoTypeInformation `
        -Encoding utf8

$activeStaleClaims = @(
    $staleRows |
        Where-Object {
            $_.Authority -eq "ACTIVE" -or
            $_.Authority -eq "PHASE"
        }
)

$activeStaleClaims |
    Export-Csv `
        (Join-Path $outputPath "stale_claims_active.csv") `
        -NoTypeInformation `
        -Encoding utf8

# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------

$linkRows = foreach ($file in $markdownFiles) {
    $relativePath = $file.FullName.Replace($repoRoot + "\", "")
    $content = Get-TextContent -Path $file.FullName

    $authority = Get-DocumentationAuthority $relativePath

    $linkMatches = [regex]::Matches(
        $content,
        "\[[^\]]+\]\(([^)]+)\)"
    )

    foreach ($linkMatch in $linkMatches) {
        $target = $linkMatch.Groups[1].Value.Trim()

        $external = (
            $target -match "^(https?:|mailto:|tel:|javascript:|data:)" -or
            $target.StartsWith("#")
        )

        [PSCustomObject]@{
            RelativePath = $relativePath.Replace("\", "/")
            Target = $target
            External = $external
            Authority = $authority.Authority
            Actionable = $authority.Actionable
            Domain = $authority.Domain
        }
    }
}

$linkRows |
    Export-Csv `
        (Join-Path $outputPath "links.csv") `
        -NoTypeInformation `
        -Encoding utf8

# ---------------------------------------------------------------------------
# Broken local Markdown links
# ---------------------------------------------------------------------------

$brokenLinks = foreach ($row in $linkRows) {

    if ($row.External) {
        continue
    }

    $sourceFile = Join-Path `
        $RepoRoot `
        $row.RelativePath.Replace("/", "\")

    $resolution = Resolve-DocumentationLink `
        -SourceFile $sourceFile `
        -Target $row.Target

    if ($resolution.Exists) {
        continue
    }

    $recommendedAction = switch ($row.Authority) {
        "ACTIVE" {
            "FIX_LINK"
        }

        "PHASE" {
            "FIX_LINK"
        }

        "GENERATED" {
            "FIX_GENERATOR"
        }

        "HISTORY" {
            "PRESERVE_OR_OPTIONAL"
        }

        "ARCHIVE" {
            "PRESERVE_OR_OPTIONAL"
        }

        "SESSION" {
            "PRESERVE_OR_OPTIONAL"
        }

        default {
            "REVIEW"
        }
    }

    [PSCustomObject]@{
        RelativePath = $row.RelativePath
        Target = $row.Target
        Authority = $row.Authority
        Domain = $row.Domain
        Actionable = $row.Actionable
        RecommendedAction = $recommendedAction
        ExpectedPath = $resolution.ResolvedPath
        Resolution = $resolution.Resolution
    }
}

$brokenLinks |
    Export-Csv `
        (Join-Path $outputPath "broken_links.csv") `
        -NoTypeInformation `
        -Encoding utf8

# ---------------------------------------------------------------------------
# Broken-link authority reports
# ---------------------------------------------------------------------------

$activeBrokenLinks = @(
    $brokenLinks |
        Where-Object {
            $_.Authority -eq "ACTIVE" -or
            $_.Authority -eq "PHASE"
        }
)

$generatedBrokenLinks = @(
    $brokenLinks |
        Where-Object {
            $_.Authority -eq "GENERATED"
        }
)

$historicalBrokenLinks = @(
    $brokenLinks |
        Where-Object {
            $_.Authority -eq "HISTORY"
        }
)

$archivedBrokenLinks = @(
    $brokenLinks |
        Where-Object {
            $_.Authority -eq "ARCHIVE"
        }
)

$sessionBrokenLinks = @(
    $brokenLinks |
        Where-Object {
            $_.Authority -eq "SESSION"
        }
)

$unclassifiedBrokenLinks = @(
    $brokenLinks |
        Where-Object {
            $_.Authority -eq "UNCLASSIFIED"
        }
)

$activeBrokenLinks |
    Export-Csv `
        (Join-Path $outputPath "broken_links_active.csv") `
        -NoTypeInformation `
        -Encoding utf8

$generatedBrokenLinks |
    Export-Csv `
        (Join-Path $outputPath "broken_links_generated.csv") `
        -NoTypeInformation `
        -Encoding utf8

$historicalBrokenLinks |
    Export-Csv `
        (Join-Path $outputPath "broken_links_history.csv") `
        -NoTypeInformation `
        -Encoding utf8

$archivedBrokenLinks |
    Export-Csv `
        (Join-Path $outputPath "broken_links_archive.csv") `
        -NoTypeInformation `
        -Encoding utf8

$sessionBrokenLinks |
    Export-Csv `
        (Join-Path $outputPath "broken_links_session.csv") `
        -NoTypeInformation `
        -Encoding utf8

$unclassifiedBrokenLinks |
    Export-Csv `
        (Join-Path $outputPath "broken_links_unclassified.csv") `
        -NoTypeInformation `
        -Encoding utf8

# ---------------------------------------------------------------------------
# Duplicate file hashes
# ---------------------------------------------------------------------------

$hashGroups = $markdownFiles |
    Get-FileHash -Algorithm SHA256 |
    Group-Object Hash |
    Where-Object Count -gt 1

$duplicateHashes = foreach ($group in $hashGroups) {
    foreach ($item in $group.Group) {
        [PSCustomObject]@{
            Hash = $group.Name
            Path = $item.Path.Replace($repoRoot + "\", "")
        }
    }
}

$duplicateHashes |
    Export-Csv `
        (Join-Path $outputPath "duplicate_files.csv") `
        -NoTypeInformation `
        -Encoding utf8

# ---------------------------------------------------------------------------
# Duplicate or near-duplicate titles
# ---------------------------------------------------------------------------

$duplicateTitles = $inventory |
    Where-Object {
        -not [string]::IsNullOrWhiteSpace($_.HeadingTitle)
    } |
    Group-Object HeadingTitle |
    Where-Object Count -gt 1 |
    ForEach-Object {
        foreach ($item in $_.Group) {
            [PSCustomObject]@{
                Title = $_.Name
                RelativePath = $item.RelativePath
            }
        }
    }

$duplicateTitles |
    Export-Csv `
        (Join-Path $outputPath "duplicate_titles.csv") `
        -NoTypeInformation `
        -Encoding utf8

# ---------------------------------------------------------------------------
# Inline HTML, CSS, and JavaScript
# ---------------------------------------------------------------------------

$inlinePatterns = @(
    "<style",
    "style=",
    "<script",
    "onclick=",
    "onchange=",
    "onload="
)

$inlineRows = foreach ($file in $markdownFiles) {
    $relativePath = $file.FullName.Replace($repoRoot + "\", "")

    Select-String `
        -Path $file.FullName `
        -Pattern $inlinePatterns `
        -SimpleMatch |
        ForEach-Object {
            [PSCustomObject]@{
                RelativePath = $relativePath
                LineNumber = $_.LineNumber
                Text = $_.Line.Trim()
            }
        }
}

$inlineRows |
    Export-Csv `
        (Join-Path $outputPath "inline_assets.csv") `
        -NoTypeInformation `
        -Encoding utf8

# ---------------------------------------------------------------------------
# Folder summary
# ---------------------------------------------------------------------------

$folderSummary = $inventory |
    Group-Object Directory |
    Sort-Object Name |
    ForEach-Object {
        [PSCustomObject]@{
            Directory = $_.Name
            Files = $_.Count
            TotalLines = ($_.Group | Measure-Object Lines -Sum).Sum
            TotalBytes = ($_.Group | Measure-Object Length -Sum).Sum
        }
    }

$folderSummary |
    Export-Csv `
        (Join-Path $outputPath "folder_summary.csv") `
        -NoTypeInformation `
        -Encoding utf8

# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------

$summaryPath = Join-Path $outputPath "summary.txt"

@"
Election Pulse Documentation Audit
====================================

Docs root: $docsPath
Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

Markdown files: $($inventory.Count)
Headings: $($headingRows.Count)

Potential stale claims: $($staleRows.Count)
Actionable stale claims: $($activeStaleClaims.Count)

Links: $($linkRows.Count)

Broken local links: $($brokenLinks.Count)
Actionable broken links: $($activeBrokenLinks.Count)
Generated broken links: $($generatedBrokenLinks.Count)
Historical broken links: $($historicalBrokenLinks.Count)
Archived broken links: $($archivedBrokenLinks.Count)
Session-log broken links: $($sessionBrokenLinks.Count)
Unclassified broken links: $($unclassifiedBrokenLinks.Count)

Duplicate file entries: $($duplicateHashes.Count)
Duplicate title entries: $($duplicateTitles.Count)
Inline asset references: $($inlineRows.Count)

Reports:
- inventory.csv
- headings.csv
- stale_claims.csv
- stale_claims_active.csv
- links.csv
- broken_links.csv
- broken_links_active.csv
- broken_links_generated.csv
- broken_links_history.csv
- broken_links_archive.csv
- broken_links_session.csv
- broken_links_unclassified.csv
- duplicate_files.csv
- duplicate_titles.csv
- inline_assets.csv
- folder_summary.csv
"@ | Set-Content $summaryPath -Encoding utf8

Write-Host ""
Write-Host "Documentation audit complete."
Write-Host "Reports: $outputPath"
Write-Host ""

Get-Content $summaryPath
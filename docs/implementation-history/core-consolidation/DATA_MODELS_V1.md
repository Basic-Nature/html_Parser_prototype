---
layout: default
title: Data Models & Schema
---

## Data Models & Schema

This document defines the core data structures, schemas, and validation models used throughout the Smart Elections Parser system.

> **Note**: This document consolidates content from:
>
> - [VERIFIED_DATA_SCHEMA.md](../VERIFIED_DATA_SCHEMA.md)
> - [CONSTANTS_INVENTORY.md](../CONSTANTS_INVENTORY.md) - see [Constants Reference](./CONSTANTS.md)
> - [VERIFICATION_FRAMEWORK.md](../VERIFICATION_FRAMEWORK.md)
>
> For detailed information, consult the individual source documents linked above.

## 📋 Core Data Structures

### Contest/Race Object

```python
{
    "contest_id": str,           # Unique identifier
    "name": str,                 # Contest name (e.g., "President")
    "race_type": str,            # "general", "primary", "special"
    "party": str,                # "Democratic", "Republican", "All Parties"
    "jurisdiction": str,         # "Federal", "State", "County", "Local"
    "division": str,             # "State", "Congressional", "County", etc.
    "office": str,               # Office title (e.g., "Governor", "Senator")
    "candidates": list[dict],    # List of candidate objects
    "selected": bool,            # Whether user selected this contest
    "confidence": float,         # Extraction confidence score (0.0-1.0)
    "source": str,               # Data source identifier
    "metadata": dict             # Additional context
}
```

### Candidate Object

```python
{
    "name": str,                    # Candidate name
    "party": str,                   # Party affiliation (if applicable)
    "votes": int,                   # Vote count
    "votes_pct": float,             # Vote percentage
    "write_in": bool,               # Is write-in candidate
    "incumbent": bool,              # Is incumbent
    "registered": bool,             # Is registered on ballot
    "confidence": float,            # Extraction confidence
    "normalized_name": str          # Canonicalized name for matching
}
```

### Header Schema

The parsed data uses standardized column headers after normalization:

```list
Core Election Columns:
- candidate_name (or: name, candidate, contender)
- votes (or: vote_count, votes_count, total_votes)
- vote_percentage (or: pct, percent, vote_pct, %)
- party (or: party_affiliation, party_code)
- jurisdiction (or: county, city, district)
- precinct (or: precinct_name, precinct_number)
- election_type (or: race_type, contest_type)
- office (or: office_title, position)
- division (or: electoral_division)
- write_in_votes (or: write_ins)
- registered (or: registered_count)
```

### Validation Rules

**Name Validation**:

- Must contain at least one letter
- Max 100 characters
- No leading/trailing whitespace
- No excessive punctuation (< 3 consecutive special chars)

**Vote Count Validation**:

- Non-negative integer
- Consistent across all rows for same race
- Reasonable bounds (< population of jurisdiction)

**Percentage Validation**:

- Ranges 0.0–100.0
- Sum to ~100.0 ± 2% for complete candidate lists
- Consistent scale (either 0-100 or 0-1.0)

**Party Validation**:

- Match canonical party list
- Normalized to standard abbreviations
- Handle affiliation variations

**Jurisdiction Validation**:

- Must be valid U.S. state/county/precinct
- Matches election records when available
- Consistent across all rows

## 🔍 Data Quality Framework

### Confidence Scoring

Each extracted field receives a confidence score (0.0–1.0):

```python
confidence_score = (
    0.3 * header_match +        # Column identification accuracy
    0.3 * value_validation +    # Conformance to expected format
    0.2 * context_consistency + # Agreement with surrounding data
    0.2 * source_reliability    # Trust in source document
)
```

**Score Interpretation**:

- **0.9+**: High confidence, ready for use without review
- **0.7–0.9**: Medium confidence, review recommended
- **0.5–0.7**: Low confidence, significant review needed
- **<0.5**: Very low confidence, expert review required

### Validation Strategies

#### Schema Validation

- Ensure required fields present
- Check field types and ranges
- Verify referential integrity

#### Semantic Validation

- Candidate names match canonical lists
- Vote totals consistent across report sections
- Percentages sum to expected total

#### Cross-Site Validation

- Compare with other sources for same election
- Flag discrepancies exceeding thresholds
- Report confidence for multi-source consensus

## 📊 Contest Selection Model

The system uses a multi-level contest selection strategy:

### Level 1: Contest Detection

- Identify all contests/races in source document
- Extract contest metadata (office, jurisdiction, party)
- Score quality of extraction (0.0–1.0)

### Level 2: Contest Filtering

- Remove duplicate contests
- Filter by user-selected criteria
- Apply jurisdiction/date filters

### Level 3: User Selection

- Present high-confidence contests to user
- Allow manual selection/deselection
- Provide default selections based on heuristics

## 🔐 Data Integrity Checks

### Consistency Checks

```python
checks = {
    "vote_total_matches_sum": votes == sum(candidate_votes),
    "percentages_sum_to_100": sum(percentages) ≈ 100 ± 2,
    "no_impossible_values": all(votes >= 0, 0 <= pct <= 100),
    "candidate_names_unique": len(names) == len(set(names)),
    "parties_consistent": party(candidate_1) consistent with contest type
}
```

### Anomaly Detection

- Unusually high/low vote percentages
- Vote patterns inconsistent with historical data
- Duplicate or near-duplicate candidate names
- Write-in vote counts exceeding registered lists

## 📝 Metadata Schema

All extracted data includes metadata:

```python
{
    "source_url": str,              # Original source
    "source_format": str,           # "pdf", "html", "csv", "json"
    "extraction_method": str,       # "panel", "section", "ml", "plugin"
    "parse_timestamp": str,         # ISO 8601 timestamp
    "parser_version": str,          # Version identifier
    "state": str,                   # U.S. state abbreviation
    "county": str,                  # County name (if applicable)
    "election_date": str,           # ISO 8601 date
    "election_type": str,           # "general", "primary", "special"
    "contests_found": int,          # Total contests detected
    "contests_selected": int,       # User-selected contests
    "extraction_confidence": float, # Overall confidence (0.0–1.0)
    "validation_passed": bool,      # All checks passed
    "warnings": list[str],          # Non-fatal issues
    "errors": list[str]             # Fatal issues
}
```

## 🔄 Data Pipeline

```tree
Raw Input (PDF/HTML/CSV/JSON)
    ↓
[Format Detection]
    ↓
[Content Parsing]
    ├→ Header Detection
    ├→ Row Extraction
    └→ Value Normalization
    ↓
[Validation Phase]
    ├→ Schema Validation
    ├→ Semantic Validation
    └→ Consistency Checks
    ↓
[Contest Selection]
    ├→ Detect Contests
    ├→ Filter Candidates
    └→ User Selection (if interactive)
    ↓
[Confidence Scoring]
    ├→ Per-Field Scoring
    ├→ Per-Contest Scoring
    └→ Overall Score
    ↓
[Output Formatting]
    ├→ Canonical Headers
    ├→ Cleaned Values
    └→ Metadata Attachment
    ↓
Validated Output (CSV/JSON + Metadata)
```

## ⚠️ Special Cases & Handling

### Write-In Handling

- Detect write-in candidate sections
- Ensure write-in counts tracked separately
- Validate write-in percentages when provided

### Multi-Level Contests

- Handle contests spanning multiple tables/sections
- Aggregate results from multiple sources
- Mark aggregation method in metadata

### Ambiguous Data

- Flag when column assignment uncertain
- Provide alternative interpretations
- Allow user to override automated choices

## 🔗 Field Normalization

### Name Normalization

```txt
Input: "JOHN Q. PUBLIC"
↓ [lowercase, trim, remove diacritics]
Output: "john q public"
```

### Vote Normalization

```txt
Input: "1,234"
↓ [remove formatting, convert to int]
Output: 1234
```

### Party Normalization

```txt
Input: "D", "DEM", "Democratic", "Dem."
↓ [map to canonical list]
Output: "Democratic"  // Canonical
```

---

**Related Documents**:

- [System Architecture](./ARCHITECTURE.md) - System design and data flow
- [Constants Reference](./CONSTANTS.md) - Enumerated values and static data
- [Verification Framework](../QUALITY/VERIFICATION.md) - QA and testing details

**Last Updated**: Consolidated from schema, constants, and verification documentation

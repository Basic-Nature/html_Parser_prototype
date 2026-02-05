---
layout: default
title: Constants & Reference Data
---

# Constants & Reference Data

Quick reference for all enumerated values, static lists, and configuration constants used throughout the Smart Elections Parser.

> **Note**: Complete constants inventory available in [CONSTANTS_INVENTORY.md](../CONSTANTS_INVENTORY.md)

## 🗺️ U.S. States

### State Codes (ANSI 5-2)

```
AL - Alabama              MT - Montana
AK - Alaska              NE - Nebraska
AZ - Arizona             NV - Nevada
AR - Arkansas            NH - New Hampshire
CA - California          NJ - New Jersey
CO - Colorado            NM - New Mexico
CT - Connecticut         NY - New York
DE - Delaware            NC - North Carolina
FL - Florida             ND - North Dakota
GA - Georgia             OH - Ohio
HI - Hawaii              OK - Oklahoma
ID - Idaho               OR - Oregon
IL - Illinois            PA - Pennsylvania
IN - Indiana             RI - Rhode Island
IA - Iowa                SC - South Carolina
KS - Kansas              SD - South Dakota
KY - Kentucky            TN - Tennessee
LA - Louisiana           TX - Texas
ME - Maine               UT - Utah
MD - Maryland            VT - Vermont
MA - Massachusetts       VA - Virginia
MI - Michigan            WA - Washington
MN - Minnesota           WV - West Virginia
MS - Mississippi         WI - Wisconsin
MO - Missouri            WY - Wyoming
DC - District of Columbia
```

## 🎯 Election Types

```
General     - General election (primary election: no)
Primary     - Primary election
Special     - Special/by-election
Runoff      - Runoff election
General Runoff - General election with runoff
```

## 🏆 Office Types

### Federal
- President
- Vice President
- U.S. Senate
- U.S. House of Representatives

### State
- Governor
- Lieutenant Governor
- Secretary of State
- Attorney General
- State Auditor
- State Treasurer
- Comptroller
- State Senate/Legislature
- State Assembly

### Local
- County Commissioner
- County Supervisor
- Mayor
- City Council
- School Board
- Sheriff
- Clerk
- Register of Deeds
- Assessor

## 🎭 Divisions

### Electoral Divisions
```
Federal     - Nationwide
State       - State-wide
Congressional - Congressional district
County      - County-wide
Precinct    - Precinct level
District    - General district (school, etc.)
City        - City incorporation
Town        - Town incorporation
Ward        - Ward/subdivision
```

## 🏛️ Jurisdictions

```
Federal     - Federal jurisdiction
State       - State-level
County      - County-level
City        - City incorporation
Town        - Town incorporation
Precinct    - Polling precinct
District    - Special district (school, fire, etc.)
Judicial    - Judicial district
```

## 🎨 Political Parties (Canonical)

```
Democratic
Republican
Libertarian
Green
Independent
Natural Law
Peace and Freedom
American Independent
Working Families
Forward Party
Socialist Workers
Write-In
Non-Partisan
All Parties (multi-party races)
```

### Party Aliases/Variations Normalized To:

```
D, DEM → Democratic
R, REP, GOP → Republican
L, LIB → Libertarian
G, GRN → Green
I, IND → Independent
etc.
```

## 📊 Data Field Standards

### Header Column Names (Preferred)

```
candidate_name          (aka: name, candidate, contender)
votes                   (aka: vote_count, total_votes)
vote_percentage         (aka: pct, percent, %)
party                   (aka: party_affiliation)
jurisdiction            (aka: county, district)
precinct                (aka: precinct_name, precinct_code)
election_type           (aka: race_type, contest_type)
office                  (aka: office_title, position)
division                (aka: electoral_division)
write_in_votes
registered
```

## ✅ Validation thresholds

```
Name length:            1–100 characters
Vote count:             0 to 10,000,000 (adjustable per jurisdiction)
Vote percentage:        0.0–100.0 or 0.0–1.0
Confidence threshold:   0.5 (minimum for auto-use)
Header match similarity: 80%+ (fuzzy match)
Percentage tolerance:   ±2% (sum validation)
```

## 🌐 Format Types

```
HTML        - Web pages, election result pages
PDF         - Portable document format
CSV         - Comma/tab-separated values
JSON        - JavaScript object notation
XML         - Extensible markup language
```

## 🔍 Extraction Methods

```
Panel       - Contiguous data blocks
Section     - Heading-based extraction
ML/NER      - Machine learning entity recognition
Plugin      - Custom handler plugin
Manual      - User-entered data
```

## 📈 Confidence Score Ranges

```
0.90–1.00   Very High    (auto-use)
0.70–0.89   High         (review recommended)
0.50–0.69   Medium       (significant review needed)
0.30–0.49   Low          (expert review required)
0.00–0.29   Very Low     (reject or reparse)
```

## 🔐 Validation Status Values

```
VALID           - Passed all validations
WARNING_MINOR   - Non-critical issues detected
WARNING_MAJOR   - Significant issues, review recommended
INVALID         - Failed critical validations
NEEDS_REVIEW    - Human review required
```

## 📝 Error Codes

| Code | Meaning | Recovery |
|------|---------|----------|
| E001 | No data found | Reparse with alternative method |
| E002 | Invalid column headers | Run header detection again |
| E003 | Duplicate candidates | Merge duplicate entries |
| E004 | Vote total mismatch | Flag discrepancy |
| E005 | Negative vote count | Treat as data error |
| E006 | Invalid format | Detect and retry |
| E007 | CAPTCHA detected | User intervention required |
| E008 | Network timeout | Retry with backoff |
| E009 | Permission denied | Check credentials |
| E010 | Unsupported election type | Manual handling required |

## 🔄 Common Field Mappings

### Vote Field Variations
```
votes → vote_count, votes_count, total_votes, vote_total
percentage → pct, percent, vote_pct, vote_percentage, %
votes_pct → percentage, pct, vote_pct
```

### Name Field Variations
```
name → candidate_name, candidate, contender, person
office → position, office_title, office_name, race
party → party_affiliation, party_code, party_name
```

## 📅 Date Formats

```
ISO 8601:   2024-11-05, 2024-11-05T14:30:00Z
US Format:  11/05/2024, November 5, 2024
Text:       "Election Day", "General Election 2024"
```

## 🎯 Contest Type Classifications

```
Race            - Head-to-head candidate competition
Proposition     - Ballot measure/question
Referendum      - Yes/No question
Recall          - Recall election
State Question  - State ballot proposition
County Measure  - County ballot measure
```

---

## Configuration Constants

### Parser Defaults
```python
DEFAULT_TIMEOUT = 30  # seconds
DEFAULT_RETRIES = 3
DEFAULT_CONFIDENCE_THRESHOLD = 0.70
MAX_CONCURRENT_BROWSERS = 4
CACHE_TTL = 3600  # seconds
```

### Logging Levels
```
DEBUG   - Detailed diagnostic information
INFO    - General informational messages
WARNING - Warning conditions (non-critical)
ERROR   - Error conditions (functionality affected)
CRITICAL - Critical errors (system failure)
```

---

**See Also**:
- [Data Models & Schema](./DATA_MODELS.md) - Detailed schema definitions
- [System Architecture](./ARCHITECTURE.md) - System design overview

**Reference Source**: [CONSTANTS_INVENTORY.md](../CONSTANTS_INVENTORY.md) - Complete inventory
**Last Updated**: Consolidated constants reference guide

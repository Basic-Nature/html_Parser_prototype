---
layout: default
title: FEC Fuzzy Matching
---

## FEC Fuzzy Matching

Machine learning-powered candidate matching against Federal Election Commission (FEC) records for validation and deduplication.

> **Note**: See [fec_fuzzy.md](../fec_fuzzy.md) for complete technical documentation

## 🎯 Overview

The FEC Fuzzy Matching system:

- Matches extracted candidates against FEC database
- Handles name variations and misspellings
- Identifies duplicate candidates
- Improves data quality and integrity
- Cross-validates with official records

## 📖 Quick Reference

### Matching Process

```tree
Extracted Candidate "John Doe"
    ↓
Query FEC Database
    ↓
Find Similar Candidates
├─ "John Doe" (100% match)
├─ "Jon Doe" (95% match)
└─ "John D." (85% match)
    ↓
Select Best Match (if > threshold)
    ↓
Validate & Link to FEC ID
```

### Usage Example

```python
from utils.fec_fuzzy import match_candidate

result = match_candidate(
    name="John Doe",
    office="Governor",
    state="NY",
    year=2024,
    threshold=0.85
)

# Result:
# {
#    'fec_id': 'C00123456',
#    'official_name': 'John Q Doe',
#    'match_score': 0.92,
#    'confidence': 'high'
# }
```

## 🔧 Configuration

```python
# Fuzzy matching thresholds
FUZZY_MATCH_THRESHOLD = 0.85      # Min similarity score
CONFIDENCE_HIGH = 0.95             # High confidence match
CONFIDENCE_MEDIUM = 0.75           # Medium confidence
CONFIDENCE_LOW = 0.50              # Low confidence

# FEC Database
FEC_DATABASE_URL = "https://api.fec.gov/v1/"
FEC_API_KEY = os.getenv("FEC_API_KEY")
```

## 📊 Algorithm Details

- **String Similarity**: Levenshtein distance with jaro-winkler weighting
- **Phonetic Matching**: Soundex for name variations
- **Context Weighting**: Office, jurisdiction, and party factors
- **Historical Data**: Past elections improve matching

---

See [fec_fuzzy.md](../fec_fuzzy.md) for:

- Complete algorithm documentation
- Performance tuning guide
- Troubleshooting procedures
- API integration examples

**Last Updated**: FEC fuzzy matching reference

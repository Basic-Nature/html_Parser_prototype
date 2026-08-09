# URL Parsing System for Training Data

## Overview

The URL parsing system breaks down election-related URLs into structured components for machine learning training. This enables pattern recognition, vendor identification, and metadata extraction from URL structures.

## Architecture

### Core Module: `webapp/parser/url_parser.py`

The URL parser module extracts:

- **Protocol**: http/https
- **Domain Components**: Full domain, root domain, subdomain
- **Path Breakdown**: Full path, individual segments, path depth
- **Query Parameters**: Parsed key-value pairs
- **Fragment**: Anchor identifiers
- **Election Metadata**: State, county, contest type, year
- **Pattern Indicators**: Election keywords, vendor hints

### API Endpoints

#### 1. `/api/urls/parse` (POST)

Parse single or multiple URLs into structured components.

**Request Body:**

```json
{
  "url": "https://results.enr.clarityelections.com/GA/105940/web.264614/#/summary",
  "store": true
}
```

Or for batch processing:

```json
{
  "urls": [
    "https://elections.example.gov/results/2024",
    "https://county.vote.state.us/precinct"
  ],
  "store": true
}
```

**Response:**

```json
{
  "success": true,
  "parsed": {
    "url": "https://results.enr.clarityelections.com/GA/105940/web.264614/#/summary",
    "protocol": "https",
    "domain": "results.enr.clarityelections.com",
    "root_domain": "clarityelections.com",
    "subdomain": "results.enr",
    "path": "/GA/105940/web.264614/",
    "path_segments": ["GA", "105940", "web.264614"],
    "path_depth": 3,
    "query_params": {},
    "state": "GA",
    "county": "",
    "contest_type": "",
    "year": "",
    "has_election_keywords": true,
    "election_keywords": ["elections", "results", "enr", "election"],
    "vendor_hint": "clarity",
    "parsed_at": "2026-02-17T22:40:43.681957Z"
  }
}
```

**Parameters:**

- `url` (string): Single URL to parse
- `urls` (array): Multiple URLs to parse
- `store` (boolean): Save parsed results to training file (default: false)

**Rate Limit:** 60 requests/minute

---

#### 2. `/api/urls/training_data` (GET)

Retrieve parsed URL training data with filtering.

**Query Parameters:**

- `limit`: Maximum records to return (default: 100, max: 1000)
- `offset`: Skip first N records (default: 0)
- `state`: Filter by state code/name (e.g., "GA", "california")
- `vendor`: Filter by vendor hint (e.g., "clarity", "voteworks")
- `has_county`: Filter to URLs with county data (true/false)

**Example Request:**

```socket
GET /api/urls/training_data?limit=50&state=GA&vendor=clarity
```

**Response:**

```json
{
  "success": true,
  "data": [
    {
      "url": "https://results.enr.clarityelections.com/GA/105940/web.264614/",
      "protocol": "https",
      "domain": "results.enr.clarityelections.com",
      "state": "GA",
      "vendor_hint": "clarity",
      ...
    }
  ],
  "count": 50,
  "total": 247,
  "offset": 0,
  "limit": 50
}
```

**Rate Limit:** 30 requests/minute

---

#### 3. `/api/urls/parse_all` (POST)

Batch parse all URLs from the URL library (`urls.txt`) and store to training file.

**Request:**

```socket
POST /api/urls/parse_all
```

**Response:**

```json
{
  "success": true,
  "parsed_count": 152,
  "failed_count": 3,
  "training_file": "webapp/parser/Context_Integration/Context_Library/log/parsed_urls_training.jsonl",
  "total_urls": 155
}
```

**Rate Limit:** 10 requests/hour (heavy operation)

---

## Storage

Parsed URLs are stored in JSONL format at:

```path
webapp/parser/Context_Integration/Context_Library/log/parsed_urls_training.jsonl
```

Each line is a complete JSON object representing one parsed URL, enabling:

- Streaming processing for large datasets
- Incremental updates without loading entire file
- Easy integration with ML training pipelines
- Simple filtering with line-by-line processing

## URL Component Extraction

### State Detection

Detects states from:

- Domain: `elections.ga.gov` → GA
- Path segments: `/results/california/...` → California
- State codes: `/AL/county/...` → AL

### County Detection

Identifies counties from:

- Path patterns: `/jefferson/county/...` → Jefferson
- Combined words: `/jeffersoncounty/...` → Jefferson
- Query parameters: `?county=Wake` → Wake

### Year Extraction

Finds election years from:

- Path segments: `/results/2024/...` → 2024
- Query params: `?year=2024` → 2024
- Combined patterns: `/election2024/...` → 2024

### Vendor Hints

Identifies platforms:

- **clarity**: Clarity Elections (ENR platform)
- **voteworks**: VoteWorks
- **dominion**: Dominion Voting
- **scytl**: Scytl
- **hart**: Hart InterCivic
- **ess**: ES&S
- **knowink**: KNOWiNK

### Contest Type Detection

Recognizes contests:

- **presidential**: President, POTUS
- **senate**: Senate, Senator
- **house**: House, Congress, Representative
- **governor**: Governor
- **state_leg**: State Assembly, Legislature
- **local**: Mayor, Council, Sheriff, Judge
- **ballot_measure**: Propositions, Amendments, Initiatives

## Usage Examples

### Python

```python
from webapp.parser.url_parser import parse_url_simple

# Parse a URL
url = "https://results.enr.clarityelections.com/GA/105940/web.264614/"
parsed = parse_url_simple(url)

print(f"State: {parsed['state']}")
print(f"Vendor: {parsed['vendor_hint']}")
print(f"Path segments: {parsed['path_segments']}")
```

### API (curl)

```bash
# Parse a single URL
curl -X POST http://localhost:5000/api/urls/parse \
  -H "Content-Type: application/json" \
  -d '{"url": "https://elections.example.gov/results/2024", "store": true}'

# Get training data filtered by state
curl "http://localhost:5000/api/urls/training_data?state=GA&limit=10"

# Parse all URLs from library
curl -X POST http://localhost:5000/api/urls/parse_all
```

### JavaScript

```javascript
// Parse URL via API
async function parseUrl(url) {
  const response = await fetch('/api/urls/parse', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, store: true })
  });

  const data = await response.json();
  console.log('Parsed URL:', data.parsed);
  return data.parsed;
}

// Get training data
async function getTrainingData(filters = {}) {
  const params = new URLSearchParams(filters);
  const response = await fetch(`/api/urls/training_data?${params}`);
  const data = await response.json();
  return data.data;
}
```

## Testing

Run the test suite:

```bash
python tests/test_url_parser.py
```

The test covers:

- Protocol and domain extraction
- Path segmentation
- State/county detection
- Year extraction
- Vendor identification
- Query parameter parsing
- JSON serialization

## ML Training Integration

The parsed URL data is structured for ML training pipelines:

1. **Feature Extraction**: Use path segments, domain components, keywords as features
2. **Pattern Recognition**: Train models to identify election URL patterns
3. **Metadata Prediction**: Predict state/county/year from URL structure
4. **Vendor Classification**: Classify vendor platforms from URL patterns
5. **Contest Type Inference**: Infer contest types from URL paths

### Recommended Features for ML

- `path_depth`: Numeric feature (tree depth)
- `path_segments`: Tokenized text features
- `subdomain`: Categorical feature
- `root_domain`: Categorical feature
- `election_keywords`: Binary/count features
- `vendor_hint`: Categorical target/feature
- `state`: Categorical target/feature
- `year`: Numeric feature

## Future Enhancements

- [ ] Improved county detection (handle city/town variations)
- [ ] Multi-state URL handling (regional results pages)
- [ ] Better contest type inference from deep paths
- [ ] PDF/file extension detection in URLs
- [ ] Tracking URL changes over time (versioning)
- [ ] Integration with navigation recipes for dynamic site parsing
- [ ] Confidence scores for metadata extraction

## Related Files

- **Module**: `webapp/parser/url_parser.py`
- **API Endpoints**: `webapp/Smart_Elections_Parser_Webapp.py` (lines ~2560-2730)
- **Tests**: `tests/test_url_parser.py`
- **Training Data**: `webapp/parser/Context_Integration/Context_Library/log/parsed_urls_training.jsonl`
- **URL Library**: `webapp/parser/urls.txt`

## Integration with Existing Systems

### Ballot Lens

The URL parser complements the Ballot Lens dropdown system by:

- Providing URL-based state/county hints as fallback
- Enhancing warehouse matching with structured URL data
- Supporting URL library metadata enrichment

### Navigation Recipes

URL pattern data can inform:

- Recipe creation for new domains
- Confidence scoring for navigation paths
- Health monitoring via URL structure changes

### Context Integration

Parsed URLs feed into:

- Context library metadata
- Source attribution in results
- Audit trails for data provenance

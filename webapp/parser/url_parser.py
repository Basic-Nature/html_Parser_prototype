"""
URL Parser for Smart Elections Parser

Breaks down URLs into structured components for training and analysis:
- Protocol (http/https)
- Domain root and subdomains  
- Path segments (individual components)
- Query parameters
- Election metadata (state, county, contest type, year)

Used for ML training on election URL patterns.
"""

import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs, unquote

# State codes for pattern matching
STATE_CODES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC"
]

STATE_NAMES = [
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new-hampshire", "new-jersey", "new-mexico", "new-york",
    "north-carolina", "north-dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode-island", "south-carolina", "south-dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west-virginia", "wisconsin", "wyoming"
]

# Common election-related keywords
ELECTION_KEYWORDS = {
    "election", "vote", "ballot", "results", "returns", "precinct",
    "county", "district", "race", "contest", "candidate", "voter",
    "registration", "poll", "referendum", "amendment", "measure",
    "proposition", "initiative", "enr", "sos", "secretary", "state",
    "clerk", "registrar", "board", "elections", "electoral"
}

# Contest type patterns
CONTEST_PATTERNS = {
    "presidential": re.compile(r"president|potus|pres\b", re.I),
    "senate": re.compile(r"senate|senator|sen\b", re.I),
    "house": re.compile(r"house|congress|representative|rep\b", re.I),
    "governor": re.compile(r"governor|gov\b", re.I),
    "state_leg": re.compile(r"assembly|legislature|delegate", re.I),
    "local": re.compile(r"mayor|council|commissioner|sheriff|judge", re.I),
    "ballot_measure": re.compile(r"measure|proposition|amendment|initiative|referendum", re.I)
}


@dataclass
class UrlComponents:
    """Structured breakdown of a URL for training"""
    # Original URL
    original_url: str
    
    # Protocol
    protocol: str  # http, https
    
    # Domain components
    domain: str  # Full domain (e.g., elections.example.gov)
    root_domain: str  # Root only (e.g., example.gov)
    subdomain: Optional[str]  # Subdomain if present (e.g., elections)
    
    # Path breakdown
    path: str  # Full path
    path_segments: List[str]  # Individual path components
    path_depth: int  # Number of segments
    
    # Query parameters
    query_string: str  # Raw query string
    query_params: Dict[str, List[str]]  # Parsed parameters
    
    # Fragment
    fragment: Optional[str]
    
    # Election metadata (extracted from URL)
    state: Optional[str]  # State code or name if found
    county: Optional[str]  # County name if found
    contest_type: Optional[str]  # Type of contest
    year: Optional[str]  # Election year if found
    
    # Pattern indicators
    has_election_keywords: bool  # Contains election-related terms
    election_keywords_found: List[str]  # Specific keywords found
    
    # Vendor/platform hints
    vendor_hint: Optional[str]  # clarity, dominion, voteworks, etc.
    
    # Metadata
    parsed_at: str  # ISO timestamp
    

def extract_root_domain(domain: str) -> Tuple[str, Optional[str]]:
    """
    Extract root domain and subdomain from full domain.
    
    Examples:
        elections.example.gov -> (example.gov, elections)
        www.state.co.us -> (state.co.us, www)
        county.vote -> (county.vote, None)
    """
    parts = domain.split(".")
    
    # Handle common TLDs
    if len(parts) <= 2:
        return domain, None
    
    # Check for .gov, .us, .com, etc.
    if len(parts) >= 3:
        # Special case: state.xx.us or site.gov
        if parts[-1] in {"us", "uk", "au"} and len(parts) >= 3:
            root = ".".join(parts[-3:])
            subdomain = ".".join(parts[:-3]) if len(parts) > 3 else None
        else:
            # Standard case: grab last 2 parts as root
            root = ".".join(parts[-2:])
            subdomain = ".".join(parts[:-2]) if len(parts) > 2 else None
        return root, subdomain
    
    return domain, None


def extract_state_from_url(url_lower: str, path_segments: List[str], domain: str) -> Optional[str]:
    """Extract state code or name from URL components"""
    # Check domain
    for code in STATE_CODES:
        if f".{code.lower()}." in domain or domain.startswith(f"{code.lower()}."):
            return code
    
    # Check path segments
    for segment in path_segments:
        seg_clean = segment.lower().strip()
        if seg_clean in [s.lower() for s in STATE_CODES]:
            return seg_clean.upper()
        if seg_clean in STATE_NAMES:
            return seg_clean
    
    # Check full URL with state names
    for name in STATE_NAMES:
        if name in url_lower or name.replace("-", "") in url_lower:
            return name
    
    return None


def extract_county_from_url(url_lower: str, path_segments: List[str]) -> Optional[str]:
    """Extract county name from URL if present"""
    # Look for "county" keyword in segments
    for i, segment in enumerate(path_segments):
        seg_lower = segment.lower()
        if "county" in seg_lower:
            # Try to get the county name (often previous or same segment)
            if i > 0:
                prev_seg = path_segments[i - 1]
                if prev_seg.lower() != "county" and len(prev_seg) > 2:
                    return prev_seg.title()
            # Or it might be combined like "jeffersoncounty"
            county_match = re.search(r"([a-z]+)county", seg_lower)
            if county_match:
                return county_match.group(1).title()
    
    # Look for specific county names in path
    county_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+County\b")
    match = county_pattern.search(" ".join(path_segments))
    if match:
        return match.group(1)
    
    return None


def extract_year_from_url(url_lower: str, path_segments: List[str], query_params: Dict[str, List[str]]) -> Optional[str]:
    """Extract election year from URL"""
    # Check query parameters first
    for key in ["year", "election", "date", "y"]:
        if key in query_params:
            values = query_params[key]
            for val in values:
                year_match = re.search(r"(20\d{2}|19\d{2})", val)
                if year_match:
                    return year_match.group(1)
    
    # Check path segments
    for segment in path_segments:
        # Direct year match (2024, 2020, etc.)
        if re.match(r"^(20\d{2}|19\d{2})$", segment):
            return segment
        # Year within segment (election2024, 2020-general, etc.)
        year_match = re.search(r"(20\d{2}|19\d{2})", segment)
        if year_match:
            return year_match.group(1)
    
    return None


def detect_contest_type(url_lower: str, path_segments: List[str]) -> Optional[str]:
    """Detect contest type from URL"""
    combined_text = " ".join([url_lower] + [s.lower() for s in path_segments])
    
    for contest_type, pattern in CONTEST_PATTERNS.items():
        if pattern.search(combined_text):
            return contest_type
    
    return None


def detect_vendor_hint(domain: str, url_lower: str) -> Optional[str]:
    """Detect vendor/platform from URL"""
    vendor_patterns = {
        "clarity": r"clarity|clarityelections",
        "voteworks": r"voteworks",
        "dominion": r"dominion",
        "scytl": r"scytl",
        "hart": r"hartintercivic|hart",
        "ess": r"essvote|ess",
        "knowink": r"knowink",
    }
    
    for vendor, pattern in vendor_patterns.items():
        if re.search(pattern, url_lower, re.I):
            return vendor
    
    return None


def find_election_keywords(url_lower: str, path_segments: List[str]) -> Tuple[bool, List[str]]:
    """Find election-related keywords in URL"""
    combined_text = " ".join([url_lower] + [s.lower() for s in path_segments])
    found_keywords = []
    
    for keyword in ELECTION_KEYWORDS:
        if keyword in combined_text:
            found_keywords.append(keyword)
    
    return len(found_keywords) > 0, found_keywords


def parse_url_components(url: str) -> UrlComponents:
    """
    Parse URL into structured components for training.
    
    Args:
        url: URL to parse
        
    Returns:
        UrlComponents with all extracted information
    """
    # Parse URL
    parsed = urlparse(url)
    
    # Extract domain components
    domain = parsed.netloc.lower()
    root_domain, subdomain = extract_root_domain(domain)
    
    # Extract path components
    path = parsed.path
    # Split and clean path segments
    path_segments = [
        unquote(seg) for seg in path.split("/") 
        if seg and seg != ""
    ]
    path_depth = len(path_segments)
    
    # Parse query parameters
    query_string = parsed.query
    query_params = parse_qs(query_string, keep_blank_values=True)
    
    # Fragment
    fragment = parsed.fragment if parsed.fragment else None
    
    # Create lowercase version for pattern matching
    url_lower = url.lower()
    
    # Extract election metadata
    state = extract_state_from_url(url_lower, path_segments, domain)
    county = extract_county_from_url(url_lower, path_segments)
    year = extract_year_from_url(url_lower, path_segments, query_params)
    contest_type = detect_contest_type(url_lower, path_segments)
    vendor_hint = detect_vendor_hint(domain, url_lower)
    
    # Find election keywords
    has_keywords, keywords_found = find_election_keywords(url_lower, path_segments)
    
    return UrlComponents(
        original_url=url,
        protocol=parsed.scheme,
        domain=domain,
        root_domain=root_domain,
        subdomain=subdomain,
        path=path,
        path_segments=path_segments,
        path_depth=path_depth,
        query_string=query_string,
        query_params=query_params,
        fragment=fragment,
        state=state,
        county=county,
        contest_type=contest_type,
        year=year,
        has_election_keywords=has_keywords,
        election_keywords_found=keywords_found,
        vendor_hint=vendor_hint,
        parsed_at=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    )


def format_url_components_for_training(components: UrlComponents) -> Dict:
    """
    Format URL components as a dict suitable for JSONL training data.
    
    Returns flattened structure optimized for ML training.
    """
    return {
        "url": components.original_url,
        "protocol": components.protocol,
        "domain": components.domain,
        "root_domain": components.root_domain,
        "subdomain": components.subdomain or "",
        "path": components.path,
        "path_segments": components.path_segments,
        "path_depth": components.path_depth,
        "query_params": components.query_params,
        "state": components.state or "",
        "county": components.county or "",
        "contest_type": components.contest_type or "",
        "year": components.year or "",
        "has_election_keywords": components.has_election_keywords,
        "election_keywords": components.election_keywords_found,
        "vendor_hint": components.vendor_hint or "",
        "parsed_at": components.parsed_at
    }


def parse_url_simple(url: str) -> Dict:
    """
    Simple interface: parse URL and return dict for API responses.
    
    Args:
        url: URL to parse
        
    Returns:
        Dict with URL components
    """
    components = parse_url_components(url)
    return format_url_components_for_training(components)

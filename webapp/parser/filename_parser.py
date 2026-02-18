"""
Filename Parser for Smart Elections Parser

Extracts election metadata from filenames for manual uploads.
Similar to URL parser but designed for file naming conventions.

Common patterns:
- State_County_Contest_Year.pdf
- Alabama_Jefferson_County_2024_General.csv
- GA_Fulton_President_2024.xlsx
- 2024_CA_Alameda_County_Results.pdf
"""

import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# State codes and names (same as url_parser)
STATE_CODES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC"
]

STATE_NAMES_FULL = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "newhampshire": "NH", "newjersey": "NJ", "newmexico": "NM", "newyork": "NY",
    "northcarolina": "NC", "northdakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhodeisland": "RI", "southcarolina": "SC",
    "southdakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "westvirginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "districtofcolumbia": "DC"
}

# Contest type keywords
CONTEST_KEYWORDS = {
    "presidential": ["president", "potus", "pres"],
    "senate": ["senate", "senator", "sen"],
    "house": ["house", "congress", "representative", "rep"],
    "governor": ["governor", "gov"],
    "state_leg": ["assembly", "legislature", "delegate"],
    "local": ["mayor", "council", "commissioner", "sheriff", "judge"],
    "ballot_measure": ["measure", "proposition", "amendment", "initiative", "referendum", "prop"],
    "general": ["general", "ge"],
    "primary": ["primary", "pp"],
    "special": ["special", "se"]
}


@dataclass
class FilenameComponents:
    """Structured breakdown of a filename"""
    # Original filename
    original_filename: str
    
    # Basic components
    filename: str  # Without extension
    extension: str  # .pdf, .csv, etc.
    
    # Parsed components (split by common delimiters)
    parts: List[str]  # Individual components
    
    # Election metadata
    state: Optional[str]  # State code or name
    county: Optional[str]  # County name
    contest_type: Optional[str]  # Type of contest
    year: Optional[str]  # Election year
    
    # Additional hints
    scope: Optional[str]  # statewide, county, precinct
    format_hint: Optional[str]  # results, canvass, summary
    
    # Metadata
    parsed_at: str  # ISO timestamp


def split_filename_parts(filename: str) -> List[str]:
    """
    Split filename into parts using common delimiters.
    
    Handles: underscores, hyphens, spaces, camelCase
    
    Examples:
        "Alabama_Jefferson_2024.pdf" → ["Alabama", "Jefferson", "2024"]
        "GA-Fulton-President-2024.csv" → ["GA", "Fulton", "President", "2024"]
        "CaliforniaAlamedaResults.xlsx" → ["California", "Alameda", "Results"]
    """
    # Remove extension
    name = Path(filename).stem
    
    # Replace common delimiters with spaces
    name = name.replace('_', ' ').replace('-', ' ')
    
    # Split camelCase (insert space before capital letters)
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    
    # Split by spaces and filter empty
    parts = [p.strip() for p in name.split() if p.strip()]
    
    return parts


def detect_state_from_parts(parts: List[str]) -> Optional[str]:
    """
    Detect state from filename parts.
    
    Checks for:
    - 2-letter state codes (AL, GA, CA, etc.)
    - Full state names (Alabama, Georgia, California, etc.)
    - Common variations (NewYork, North_Carolina, etc.)
    """
    for part in parts:
        part_clean = part.strip().upper()
        
        # Check for state code
        if len(part_clean) == 2 and part_clean in STATE_CODES:
            return part_clean
        
        # Check for full state name
        part_lower = part.lower().replace(' ', '').replace('_', '').replace('-', '')
        if part_lower in STATE_NAMES_FULL:
            return STATE_NAMES_FULL[part_lower]
    
    # Check combinations (e.g., "North Carolina" split into ["North", "Carolina"])
    for i in range(len(parts) - 1):
        combined = (parts[i] + parts[i + 1]).lower().replace(' ', '')
        if combined in STATE_NAMES_FULL:
            return STATE_NAMES_FULL[combined]
    
    return None


def detect_county_from_parts(parts: List[str]) -> Optional[str]:
    """
    Detect county from filename parts.
    
    Looks for:
    - Words followed by "County", "Parish", etc.
    - Common county patterns
    """
    for i, part in enumerate(parts):
        part_lower = part.lower()
        
        # Direct match with "County" suffix
        if part_lower == "county" and i > 0:
            return parts[i - 1].title()
        
        # Combined like "JeffersonCounty"
        if "county" in part_lower:
            county_name = part_lower.replace("county", "").strip()
            if county_name:
                return county_name.title()
        
        # Parish (Louisiana)
        if part_lower == "parish" and i > 0:
            return parts[i - 1].title()
        
        # Borough (Alaska)
        if part_lower == "borough" and i > 0:
            return parts[i - 1].title()
    
    # Check for multi-word counties (e.g., "St Louis")
    for i in range(len(parts) - 1):
        if parts[i].lower() == "st" and i + 1 < len(parts):
            return f"{parts[i]} {parts[i + 1]}".title()
    
    return None


def detect_year_from_parts(parts: List[str]) -> Optional[str]:
    """
    Detect year from filename parts.
    
    Looks for 4-digit years (2020-2030 range)
    """
    for part in parts:
        # Direct 4-digit year
        if re.match(r'^(20[2-3]\d)$', part):
            return part
        
        # Year embedded in string (e.g., "2024Election")
        year_match = re.search(r'(20[2-3]\d)', part)
        if year_match:
            return year_match.group(1)
    
    return None


def detect_contest_type_from_parts(parts: List[str]) -> Optional[str]:
    """
    Detect contest type from filename parts.
    
    Matches against known contest keywords.
    """
    parts_lower = [p.lower() for p in parts]
    combined = ' '.join(parts_lower)
    
    for contest_type, keywords in CONTEST_KEYWORDS.items():
        for keyword in keywords:
            if keyword in parts_lower or keyword in combined:
                return contest_type
    
    return None


def detect_scope_from_parts(parts: List[str]) -> Optional[str]:
    """
    Detect scope (statewide, county, precinct).
    """
    parts_lower = [p.lower() for p in parts]
    
    if "statewide" in parts_lower or "state" in parts_lower:
        return "statewide"
    if "county" in parts_lower or "parish" in parts_lower:
        return "county"
    if "precinct" in parts_lower:
        return "precinct"
    
    return None


def detect_format_hint_from_parts(parts: List[str]) -> Optional[str]:
    """
    Detect format/type hint (results, canvass, summary, etc.).
    """
    format_keywords = {
        "results": ["results", "result"],
        "canvass": ["canvass"],
        "summary": ["summary"],
        "report": ["report"],
        "export": ["export"],
        "data": ["data"],
        "certified": ["certified"]
    }
    
    parts_lower = [p.lower() for p in parts]
    
    for fmt, keywords in format_keywords.items():
        for keyword in keywords:
            if keyword in parts_lower:
                return fmt
    
    return None


def parse_filename(filename: str) -> FilenameComponents:
    """
    Parse filename into structured components.
    
    Args:
        filename: Filename to parse (can include extension)
        
    Returns:
        FilenameComponents with extracted metadata
    """
    # Get extension
    path = Path(filename)
    extension = path.suffix.lower()
    name_only = path.stem
    
    # Split into parts
    parts = split_filename_parts(filename)
    
    # Extract metadata
    state = detect_state_from_parts(parts)
    county = detect_county_from_parts(parts)
    year = detect_year_from_parts(parts)
    contest_type = detect_contest_type_from_parts(parts)
    scope = detect_scope_from_parts(parts)
    format_hint = detect_format_hint_from_parts(parts)
    
    return FilenameComponents(
        original_filename=filename,
        filename=name_only,
        extension=extension,
        parts=parts,
        state=state,
        county=county,
        contest_type=contest_type,
        year=year,
        scope=scope,
        format_hint=format_hint,
        parsed_at=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    )


def parse_filename_simple(filename: str) -> Dict:
    """
    Simple interface: parse filename and return dict.
    
    Args:
        filename: Filename to parse
        
    Returns:
        Dict with filename components
    """
    components = parse_filename(filename)
    return asdict(components)


# Example usage and testing
if __name__ == "__main__":
    test_filenames = [
        "Alabama_Jefferson_County_2024_General.pdf",
        "GA-Fulton-President-2024.csv",
        "California_Alameda_Results_2024.xlsx",
        "2024_Florida_Statewide_Senate.pdf",
        "NewYork_Rockland_County_General_2024.csv",
        "PA_StLouis_Canvass_2024.pdf",
        "Arizona_Maricopa_Primary_2024.xlsx",
        "TX-Harris-General-Election-2024.csv",
        "Washington_King_County_Results.pdf",
        "20241105_GeneralCanvass_Signed.pdf"
    ]
    
    print("=" * 80)
    print("FILENAME PARSER TEST")
    print("=" * 80)
    print()
    
    for filename in test_filenames:
        print(f"Filename: {filename}")
        print("-" * 80)
        
        parsed = parse_filename_simple(filename)
        
        print(f"  Parts: {parsed['parts']}")
        print(f"  State: {parsed['state'] or 'Not detected'}")
        print(f"  County: {parsed['county'] or 'Not detected'}")
        print(f"  Year: {parsed['year'] or 'Not detected'}")
        print(f"  Contest Type: {parsed['contest_type'] or 'Not detected'}")
        print(f"  Scope: {parsed['scope'] or 'Not detected'}")
        print(f"  Format: {parsed['format_hint'] or 'Not detected'}")
        print()

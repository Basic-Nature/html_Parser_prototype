import os, re

from typing import Dict, Set, List
from ..config import CONTEXT_LIBRARY_PATH, PROJECT_ROOT, LOG_DIR, BASE_DIR
import orjson
import subprocess
import sys
import time
from tempfile import NamedTemporaryFile
import shutil
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import time
import threading
import shutil
import tempfile
from ..utils.shared_logger import SharedLogger
logger = SharedLogger()
_CONTEXT_LOCK = threading.Lock()
SCHEMA_VERSION = "1.0"

DEFAULT_STRUCTURE = {
    "schema_version": SCHEMA_VERSION,
    "contests": [],
    "panels": [],
    "tables": [],
    "buttons": [],
    "metadata": {},
}
_context_library_cache = None

#: Maps normalized state names to a sorted list of their counties (all lowercase).
KNOWN_STATE_TO_COUNTY_MAP: Dict[str, List[str]] = {
    "alabama": sorted([
        "autauga", "baldwin", "barbour", "bibb", "blount", "bullock", "butler", "calhoun", "chambers",
        "cherokee", "chilton", "choctaw", "clarke", "cleburne", "coffee", "colbert", "coneucuh", "coosa",
        "covington", "crenshaw", "cullman", "dale", "dallas", "de kalb", "elmore", "escambia", "etowah",
        "fayette", "franklin"
    ]),
    "alaska": sorted([
        "aleutians east", "aleutians west", "anchorage", "bethel", "bristol bay", "denali",
        "fairbanks north star", "haines", "juneau", "kenai peninsula", "ketchikan gateway", "kipnuk",
        "kodiak island", "lake and peninsula", "matanuska-susitna", "nome", "north slope", "northwest arctic",
        "prince of wales-hyder", "sitka", "skagway", "southeast fairbanks", "valdez-cordova"
    ]),
    "arizona": sorted([
        "apache", "coconino", "gila", "graham", "greenlee", "la paz", "maricopa", "mojave", "navajo",
        "pima", "pinal", "santa cruz", "yavapai", "yuma"
    ]),
    "arkansas": sorted([
        "arkansas", "ashley", "baxter", "benton", "boone", "bradley", "calhoun", "carroll", "chicot",
        "clark", "clay", "cleburne", "columbia", "conway", "craighead", "crawford", "crittenden", "cross",
        "dallas", "desha"
    ]),
    "california": sorted([
        "alameda", "alpine", "amador", "butte", "calaveras", "colusa", "contra costa", "del norte",
        "el dorado", "fresno", "glenn", "humboldt", "imperial", "inyo", "kern", "kings", "lake", "lassen",
        "los angeles", "madera"
    ]),
    "colorado": sorted([
        "adams", "alamosa", "arapahoe", "archuleta", "baca", "bent", "boulder", "broomfield", "chaffee",
        "cheyenne", "clear creek", "conejos", "costilla", "crowley", "custer", "delta", "denver", "dolores",
        "douglas", "eagle"
    ]),
    "connecticut": sorted([
        "fairfield", "hartford", "litchfield", "middlesex", "new haven", "new london", "tolland", "windham"
    ]),
    "delaware": sorted([
        "kent", "new castle", "sussex"
    ]),
    "district_of_columbia": ["district of columbia"],
    "florida": sorted([
        "alachua", "baker", "bay", "bradford", "brevard", "broward", "calhoun", "charlotte", "citrus",
        "clay", "collier", "columbia", "de soto", "dixie", "duval", "escambia", "flagler", "franklin"
    ]),
    "georgia": sorted([
        "appling", "atlanta", "bacon", "baker", "baldwin", "banks", "barrow", "bartow", "ben hill", "bibb",
        "bleckley", "brantley", "brooks", "bulloch", "burke", "butts"
    ]),
    "hawaii": sorted([
        "hawaii", "honolulu", "kalaheo", "kauai", "maui"
    ]),
    "idaho": sorted([
        "ada", "adams", "bannock", "bear lake", "benewah", "bingham", "blaine", "boise", "bonner",
        "bonneville", "boundary", "butte", "camas", "canyon", "caribou", "cassia"
    ]),
    "illinois": sorted([
        "adams", "alexander", "bond", "boone", "brown", "bureau", "calhoun", "carroll", "cass",
        "champaign", "christian", "clark", "clay", "clinton", "coles", "cook"
    ]),
    "indiana": sorted([
        "adams", "allen", "bartholomew", "benton", "blackford", "boone", "brown", "carroll", "cass",
        "clark", "clay", "clinton", "crawford", "dearborn", "decatur", "dekalb"
    ]),
    "iowa": sorted([
        "adair", "adams", "allamakee", "appanoose", "audubon", "benton", "black hawk", "boone", "bremer",
        "buchanan", "buena vista", "butler", "calhoun", "carroll", "cass"
    ]),
    "kansas": sorted([
        "allen", "anderson", "atchison", "barber", "barton", "bourbon", "brown", "butler", "chase",
        "chautauqua", "cherokee", "cheyenne", "clark", "clay", "cloud"
    ]),
    "kentucky": sorted([
        "adair", "allen", "anderson", "ballard", "barren", "bell", "boone", "bourbon", "boyd", "boyle",
        "bracken", "breathitt", "breckinridge", "bullitt", "butler"
    ]),
    "louisiana": sorted([
        "acadia", "allen", "ascension", "assumption", "avoyelles", "bienville", "bossier", "caddo",
        "calcasieu", "cameron", "catahoula", "claiborne", "concordia", "de soto", "east baton rouge"
    ]),
    "maine": sorted([
        "androscoggin", "aroostook", "cumberland", "franklin", "hancock", "kennebec", "knox", "lincoln",
        "oxford", "penobscot", "sagadahoc", "somerset", "waldo", "washington", "york"
    ]),
    "maryland": sorted([
        "anne arundel", "baltimore", "calvert", "caroline", "carroll", "cecil", "charles", "dorchester",
        "frederick", "garrett", "harford", "howard", "kent", "montgomery", "prince george's"
    ]),
    "massachusetts": sorted([
        "barnstable", "berkshire", "bristol", "dukes", "essex", "franklin", "hampden", "hampshire",
        "middlesex", "nantucket", "norfolk", "plymouth", "suffolk", "worcester"
    ]),
    "michigan": sorted([
        "alcona", "alger", "allegan", "alpena", "antrim", "arenac", "baraga", "barry", "bay", "benzie",
        "berrien", "branch", "calhoun", "cass", "charlevoix"
    ]),
    "minnesota": sorted([
        "aitkin", "anoka", "becker", "beltrami", "benton", "big stone", "blue earth", "brown", "carver",
        "cass", "chippewa", "chisago", "clay", "clearwater", "cook"
    ]),
    "mississippi": sorted([
        "adams", "alcorn", "amite", "attala", "benton", "bolivar", "calhoun", "carroll", "chickasaw",
        "choctaw", "claiborne", "clarke", "clay", "coahoma", "copiah"
    ]),
    "missouri": sorted([
        "adair", "andrew", "atchison", "audrain", "barry", "barton", "bates", "benton", "bollinger",
        "boone", "buchanan", "butler", "caldwell", "callaway", "camden"
    ]),
    "montana": sorted([
        "beaverhead", "big horn", "blaine", "broadwater", "carbon", "carter", "cascade", "chouteau",
        "custer", "daniels", "dawson", "deer lodge", "fallon", "fergus", "flathead"
    ]),
    "nebraska": sorted([
        "adams", "antelope", "arthur", "banner", "blaine", "boone", "box butte", "boyd", "brown",
        "buffalo", "burke", "butler", "cass", "cedar", "chase"
    ]),
    "nevada": sorted([
        "carson city", "churchill", "clark", "douglas", "elko", "esmeralda", "eureka", "humboldt",
        "lander", "lincoln", "lyon", "mineral", "nye", "pershing", "storey"
    ]),
    "new_hampshire": sorted([
        "belknap", "carroll", "cheshire", "coos", "grafton", "hillsborough", "merrimack", "rockingham",
        "strafford", "sullivan"
    ]),
    "new_jersey": sorted([
        "atlantic", "bergen", "burlington", "camden", "cape may", "cumberland", "essex", "gloucester",
        "hudson", "hunterdon", "mercer", "middlesex", "monmouth", "morris"
    ]),
    "new_mexico": sorted([
        "bernalillo", "catron", "chaves", "cibola", "colfax", "de baca", "doña ana", "eddie", "grant",
        "guadalupe", "harding", "hidalgo", "leonard wood", "los alamos", "luna"
    ]),
    "new_york": sorted([
        "albany", "allegany", "bronx", "broome", "cattaraugus", "cayuga", "chautauqua", "chemung",
        "chenango", "clinton", "columbia", "cortland", "delaware", "dutchess", "erie", "rockland"
    ]),
    "north_carolina": sorted([
        "alamance", "alexander", "alleghany", "anson", "ashe", "avery", "beaufort", "bertie", "bladen",
        "brunswick", "buncombe", "burke", "cabarrus", "caldwell", "camden"
    ]),
    "north_dakota": sorted([
        "adams", "barnes", "burke", "cass", "cavalier", "dickey", "divide", "dunn", "edmunds", "emmons",
        "foster", "golden valley", "grand forks", "grant", "hedinger"
    ]),
}
# --- Central Dynamic Sets (used everywhere) ---

STATE_MODULE_MAP: Dict[str, str] = {
    state: (
        "webapp.parser.handlers.states.dc.dc"
        if state == "district_of_columbia"
        else f"webapp.parser.handlers.states.{state}.{state}"
    )
    for state in KNOWN_STATE_TO_COUNTY_MAP.keys()
}

_CANONICAL_STATE_ABBR: Dict[str, List[str]] = {
    "alabama": ["al", "ala"],
    "alaska": ["ak"],
    "arizona": ["az", "ariz"],
    "arkansas": ["ar", "ark"],
    "california": ["ca", "calif"],
    "colorado": ["co", "colo"],
    "connecticut": ["ct", "conn"],
    "delaware": ["de", "del"],
    "district_of_columbia": ["dc", "d.c."],
    "florida": ["fl", "fla"],
    "georgia": ["ga", "ga."],
    "hawaii": ["hi"],
    "idaho": ["id"],
    "illinois": ["il", "ill"],
    "indiana": ["in", "ind"],
    "iowa": ["ia"],
    "kansas": ["ks", "kans"],
    "kentucky": ["ky", "ky."],
    "louisiana": ["la", "la."],
    "maine": ["me"],
    "maryland": ["md", "md."],
    "massachusetts": ["ma", "mass"],
    "michigan": ["mi", "mich."],
    "minnesota": ["mn", "minn"],
    "mississippi": ["ms", "miss"],
    "missouri": ["mo", "mo."],
    "montana": ["mt", "mont"],
    "nebraska": ["ne", "nebr"],
    "nevada": ["nv", "nev"],
    "new_hampshire": ["nh", "n.h."],
    "new_jersey": ["nj", "n.j."],
    "new_mexico": ["nm", "n. mex."],
    "new_york": ["ny", "n.y."],
    "north_carolina": ["nc", "n.c."],
    "north_dakota": ["nd", "n. dak."],
    "ohio": ["oh"],
    "oklahoma": ["ok", "okla"],
    "oregon": ["or", "ore"],
    "pennsylvania": ["pa", "pa."],
    "rhode_island": ["ri", "r.i."],
    "south_carolina": ["sc", "s.c."],
    "south_dakota": ["sd", "s. dak."],
    "tennessee": ["tn", "tenn"],
    "texas": ["tx", "tex"],
    "utah": ["ut"],
    "vermont": ["vt", "vt."],
    "virginia": ["va", "va."],
    "washington": ["wa", "wash"],
    "west_virginia": ["wv", "w. va."],
    "wisconsin": ["wi", "wis"],
    "wyoming": ["wy", "wyo"],
}

STATE_ABBR: Dict[str, str] = {
    abbr: state
    for state, abbrs in _CANONICAL_STATE_ABBR.items()
    for abbr in abbrs + [state]
}

#: Maps normalized county names to a list of their precincts.
KNOWN_COUNTY_TO_PRECINCTS_MAP: dict[str, list[str]] = {
    "rockland": [
        "haverstraw",
        "clarkstown",
        "ramapo",
        "orangetown",
        "stony point"
    ],
    "kings": [
        "brooklyn",
        "coney island",
        "brownsville"
    ],
    "queens": [
        "astoria",
        "flushing",
        "jamaica",
        "long island city",
        "forest hills"
    ],
    "los angeles": [
        "central la",
        "south la",
        "east la"
    ],
    "cook": [
        "chicago",
        "evanston",
        "oak park"
    ],
    "maricopa": [
        "phoenix",
        "mesa",
        "chandler"
    ],
    # Add more counties and their districts/precincts as needed
}

HTML_TAGS: Set[str] = set([
    "html", "head", "title", "body", "h1", "h2", "h3", "h4", "h5", "h6",
    "b", "i", "center", "ul", "li", "br", "p", "hr", "img", "a", "span", "div", "button", "input", "form", "table"
])
PANEL_TAGS: Set[str] = set([
    "section", "fieldset", "panel", "div", "p-panel", "app-ballot-item-wrapper", "article", "main",
    "aside"
    
])

TABLE_TAGS: Set[str] = set([
    "table", "thead", "tbody", "tfoot", "tr", "th", "td", "caption", "colgroup", "col",
    "table", "results", "summary", "sheet", "spreadsheet", "grid"
])

STATE_TAGS: Set[str] = set([
    "state", "province", "territory", "region"
])

BUTTON_TAGS: Set[str] = set([
    "button", "input", "select", "textarea", "Show Results", "Vote", "Submit", "Summary", "Next", "Continue", "Back",
    "Download", "Print", "Details", "Results", "Ballot", "Cast Vote", "Vote Now", "Vote Here", "Submit Vote", "Confirm Vote",
])

HEADING_TAGS: Set[str] = set([
    "h1", "h2", "h3", "h4", "h5", "h6", "span", "b", "strong"
])

EXTRA_HEADING_TAGS: Set[str] = set([
    "title", "legend", "caption", "summary", "label", "header", "footer", "nav", "article", "section", "aside",
    "div", "p", "hgroup", "dt", "dd", "th", "td", "li", "ul", "ol", "dl", "blockquote", ".ng-star-inserted", ".section-title", ".panel-header", ".fw-bold"
])

CUSTOM_ATTR_PATTERNS: List[re.Pattern] = [
    re.compile(r"^data-"),
    re.compile(r"^aria-"),
    re.compile(r"^role$"),
]

DISTRICT_REGEX: re.Pattern = re.compile(
    r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)*\s*\d{1,3}|District\s*\d{1,3}|Ward\s*\d{1,3}|Precinct\s*\d{1,3}|ED\s*\d{1,3})\b"
)

# --- Table/Entity Keywords (from table_core, dynamic_table_extractor, etc.) ---
BALLOT_TYPES: List[str] = [
    "Election Day", "Early Voting", "Absentee", "Mail", "Provisional", "Affidavit", "Other", "Void"
]
BALLOT_TYPES_SORT_ORDER: List[str] = [
    "Election Day", "Early Voting", "Absentee", "Mail", "Absentee Mail"
]
LOCATION_KEYWORDS: Set[str] = {
    "precinct",
    "ward",
    "district",
    "voting district",
    "county",
    "city",
    "town",
    "township",
    "neighborhood",
    "polling place",
    "election district",
    "voting area",
    "electoral district",
    "community district",
    "voting center",
    "voting location",
    "polling station",
    "polling area",
    "voting precinct",
    "voting place",
    "polling district",
    "electoral area",
    "electoral division",
    "electoral zone",
    "electoral region",
    "electoral section",
    "electoral ward",
    "electoral unit",
    "electoral subdivision",
    "electoral precinct",
    "borough",
    "village",
    "division",
    "subdistrict",
    "ed",  # abbreviation for election district
    "municipality",
    "section",
    "region",
    "zone",
    "subdivision",
    "community",
    "block",
    "site",
    "station",
    "place",
    "locale",
    "sector",
    "unit",
    "assembly district",
    "senate district",
    "school district",
    "congressional district",
    "judicial district",
    "supervisorial district",
    "council district",
    "precinct number",
    "precinct name",
    "district number",
    "district name",
    "polling location",
    "poll site",
    "precinct id",
    "district id"
}
PERCENT_KEYWORDS: Set[str] = {
    "% precincts reporting", "% reported", "percent reported", "fully reported", "precincts reporting"
}
TOTAL_KEYWORDS: Set[str] = {"total", "sum", "votes", "overall", "all", "Percent Reported", "Reporting Status"}
MISC_FOOTER_KEYWORDS: Set[str] = {"undervote", "overvote", "scattering", "write-in", "blank", "void", "spoiled"}
CANDIDATE_KEYWORDS: Set[str] = {
    "candidate", "candidates", "name", "nominee", "person", "individual", "contestant",
    "office", "incumbent", "challenger", "write-in", "write in", "writein", "option", "choice",
    "party", "affiliation", "designation", "slate", "ticket", "representative", "member", "appointee"
}
PARTY_KEYWORDS: Set[str] = {
    "democratic", "republican", "working families", "conservative", "green", "libertarian",
    "independent", "write-in", "write in", "writein", "other", "constitution", "socialist",
    "progressive", "labor", "peace and freedom", "american independent", "no party", "nonpartisan",
    "unaffiliated", "unknown", "blank", "void", "spoiled", "scattering", "undeclared", "unaffiliated",
    "party", "affiliation", "designation", "Democratic", "DEM", "dem", "Republican", "REP", "rep", 
    "Working Families", "WOR", "wor", "Conservative", "CON", "con", "Green", "GRN", "grn", 
    "Libertarian", "LIB", "lib", "Independent", "IND", "ind", "Larouche", "Write-In", "Other" 
}
LOCATION_ABBREVIATIONS: Dict[str, List[str]] = {
    "ed": ["election district"],
    "ward": ["ward"],
    "wd": ["ward"],
    "dist": ["district"],
    "district": ["district"],
    "pct": ["precinct"],
    "prec": ["precinct"],
    "precinct": ["precinct"],
    "muni": ["municipality"],
    "mun": ["municipality"],
    "area": ["area"],
    "city": ["city"],
    "cty": ["county"],
    "munic": ["municipality"],
    "borough": ["borough"],
    "boro": ["borough"],
    "vill": ["village"],
    "vlg": ["village"],
    "village": ["village"],
    "cnty": ["county"],
    "county": ["county"],
    "div": ["division"],
    "division": ["division"],
    "subdist": ["subdistrict"],
    "subdistrict": ["subdistrict"],
    "pollpl": ["polling place"],
    "poll pl": ["polling place"],
    "polling place": ["polling place"],
    "pl": ["place"],
    "section": ["section"],
    "sec": ["section"],
    "region": ["region"],
    "reg": ["region"],
    "zone": ["zone"],
    "zn": ["zone"],
    "subdivision": ["subdivision"],
    "sd": ["subdivision"],
    "comm": ["community"],
    "community": ["community"],
    "neigh": ["neighborhood"],
    "neighborhood": ["neighborhood"],
    "blk": ["block"],
    "block": ["block"],
    "site": ["site"],
    "station": ["station"],
    "stn": ["station"],
    "locale": ["locale"],
    "sector": ["sector"],
    "unit": ["unit"],
    "ad": ["assembly district"],
    "assembly district": ["assembly district"],
    "sd": ["senate district"],
    "senate district": ["senate district"],
    "cd": ["congressional district"],
    "congressional district": ["congressional district"],
    "jd": ["judicial district"],
    "judicial district": ["judicial district"],
    "sup dist": ["supervisorial district"],
    "supervisorial district": ["supervisorial district"],
    "council dist": ["council district"],
    "council district": ["council district"],
    "precinct no": ["precinct number"],
    "precinct num": ["precinct number"],
    "precinct number": ["precinct number"],
    "precinct name": ["precinct name"],
    "district no": ["district number"],
    "district num": ["district number"],
    "district number": ["district number"],
    "district name": ["district name"],
    "poll loc": ["poll location"],
    "poll location": ["poll location"],
    "poll site": ["polling station"],
    "polling station": ["polling station"],
    "precinct id": ["precinct id"],
    "district id": ["district id"]
}
ELECTION_TYPES: set = {"general", "primary", "presidential preference", "special", "runoff", "municipal", "local"}
CONTEST_KEYWORDS: set = {
    "award program",
    "vice president",
    "presidential",
    "senate",
    "senator",
    "congress",
    "representative",
    "electors",
    "house of representatives",
    "proposition",
    "amendment",
    "house",
    "district representative",
    "district delegate",
    "governor",
    "lieutenant governor",
    "attorney general",
    "comptroller",
    "treasurer",
    "secretary of state",
    "state senator",
    "state assembly",
    "state representative",
    "assembly member",
    "member of assembly",
    "state house",
    "state senate",
    "state house of representatives",
    "state delegate",
    "state board of education",
    "state board of elections",
    "state board of equalization",
    "state board of supervisors",
    "state board of trustees",
    "state board of directors",
    "state board of commissioners",
    "state board of assessors",
    "state board of auditors",
    "state board of registrars",
    "state board of supervisors of elections",
    "state board of supervisors of registration",
    "state board of supervisors of voter registration",
    "state board of supervisors of elections and registration",
    "state board of supervisors of elections and voter registration",
    "state board of supervisors of elections and registrars",
    "state board of supervisors of elections and registrars of voters",
    "state board of supervisors of elections and registrars of voter registration",
    "state board of supervisors of elections and voter registration and elections",
    "state board of supervisors of elections and registrars of voter registration and elections",
    "mayor",
    "city council",
    "councilmember",
    "county clerk",
    "sheriff",
    "assessor",
    "district attorney",
    "county commissioner",
    "city auditor",
    "board of supervisors",
    "town council",
    "clerk",
    "judge",
    "justice",
    "supreme court justice",
    "district court",
    "appellate court",
    "trustee",
    "circuit court",
    "magistrate",
    "municipal court",
    "family court",
    "probate court",
    "school board",
    "board of education",
    "superintendent of schools",
    "school committee",
    "school trustee",
    "board of trustees",
    "board of school directors",
    "public utility commissioner",
    "soil and water conservation district supervisor",
    "soil and water conservation board",
    "soil and water conservation district director",
    "soil and water conservation district board",
    "soil and water conservation district commissioner"
}
ALWAYS_IGNORE_TAGS: set = {
        "script", "style", "svg", "path", "defs", "g", "canvas", "noscript", "meta", "link", "base", "title"
    }
ALWAYS_IGNORE_CLASSES: set = {
        "visually-hidden", "sr-only", "skip-link", "screen-reader", "aria-hidden", "d-none", "hidden", "offscreen"
    }
ALWAYS_IGNORE_IDS: set = {
        "skip-link", "hidden", "aria-hidden"
    }
ROOT_CONTAINER_TAGS: set = {"body", "html", "app-root"}

ICON_CLASSES: set = {
        "pi", "bi", "fa", "fas", "far", "fal", "fad", "fab", "glyphicon", "icon", "material-icons",
        "mdi", "octicon", "feather", "ion", "ionicon", "anticon", "euiicon", "p-button-icon", "p-icon",
        "fa-solid", "fa-regular", "fa-light", "fa-duotone", "fa-brands", "fa-stack", "fa-stack-1x", "fa-stack-2x",
        "fa-fw", "fa-li", "fa-border", "fa-spin", "fa-pulse", "fa-inverse", "fa-layers", "fa-layers-text", "fa-layers-counter",
        "oi", "eva", "eva-icon", "remixicon", "ri", "icofont", "icn", "flaticon", "glyph", "iconify", "iconfont",
        "uicon", "uik", "uik-icon", "uik-button-icon", "octicon", "octicon-alert", "octicon-info", "octicon-check",
        "octicon-x", "octicon-star", "octicon-stop", "octicon-download", "octicon-upload", "octicon-arrow", "octicon-chevron",
        "octicon-dot", "octicon-dot-fill", "octicon-dot-outline", "octicon-dot-circle", "octicon-dot-square",
        "icon-label", "icon-btn", "icon-button", "icon-container", "icon-wrapper", "icon-box", "icon-bg", "icon-bg-light",
        "icon-bg-dark", "icon-bg-primary", "icon-bg-secondary", "icon-bg-success", "icon-bg-danger", "icon-bg-warning",
        "icon-bg-info", "icon-bg-white", "icon-bg-black", "icon-bg-gray", "icon-bg-grey", "icon-bg-transparent",
        "icon-bg-gradient", "icon-bg-image", "icon-bg-pattern", "icon-bg-shape", "icon-bg-circle", "icon-bg-square",
        "icon-bg-rectangle", "icon-bg-oval", "icon-bg-round", "icon-bg-pill", "icon-bg-dot", "icon-bg-line",
        "icon-bg-arrow", "icon-bg-chevron", "icon-bg-star", "icon-bg-heart", "icon-bg-check", "icon-bg-x", "icon-bg-plus",
        "icon-bg-minus", "icon-bg-close", "icon-bg-open", "icon-bg-expand", "icon-bg-collapse", "icon-bg-menu", "icon-bg-more",
        "icon-bg-less", "icon-bg-up", "icon-bg-down", "icon-bg-left", "icon-bg-right", "icon-bg-top", "icon-bg-bottom",
        "icon-bg-center", "icon-bg-middle", "icon-bg-end", "icon-bg-start", "icon-bg-first", "icon-bg-last", "icon-bg-prev",
        "icon-bg-next"
    }
ICON_TAGS: set = {"i", "svg", "path", "g", "span"}

NOISY_LABEL_PATTERNS: list = [
    r"(?i)\b(show results|vote|submit|summary|next|continue|back|download|print|details|results|ballot|cast vote)\b",
    r"(?i)\b(vote now|vote here|submit vote|confirm vote)\b"
]

PRECINCT_HEADER_PATTERNS: list = [
    r"(?i)\b(precincts reporting|precincts counted|precincts remaining|precincts total|precincts)\b",
    r"(?i)\b(precinct reporting|precinct count|precinct remaining|precinct total|precinct)\b",
    r"(?i)\b(precincts reporting status|precincts reporting details |precincts reporting information)\b",
]

CONTEST_PANEL_TAGS: set = {
    "contest", "contest-panel", "contest-item", "contest-wrapper", "contest-container",
    "contest-box", "contest-card", "contest-section", "contest-row", "contest-column",
    "contest-header", "contest-title", "contest-name", "contest-info", "contest-details",
    "contest-description", "contest-summary", "contest-results", "contest-votes", "contest-candidates",
    "contest-parties", "contest-positions", "contest-offices", "contest-measures"
}

SELECTORS: dict = {
    "button": {
        "type_": "button",
        "role": "button",
        "aria-pressed": "false"
    },
    "link": {
        "type_": "link",
        "role": "link"
    },
    "checkbox": {
        "type_": "checkbox",
        "role": "checkbox"
    },
    "radio": {
        "type_": "radio",
        "role": "radio"
    }
}

# --- Canonical Segment Labeling & Normalization ---
CANONICAL_SEGMENT_LABELS: dict = {
    # Add common canonical mappings here
    "election results": "results_table",
    "results by precinct": "location_panel",
    "summary": "summary",
    "total votes": "total_votes",
    "precincts reporting": "reporting_status",
    "candidate": "candidate_panel",
    "ballot types": "ballot_types",
    "download": "download_link",
    # Add more as needed
}

BUTTON_CLASSES: set = {"btn", "button", "toggle", "switch", "p-button", "mat-button", "v-btn", "ant-btn", "el-button"}

HEADING_CLASSES: set = {"heading", "header", "title", "h1", "h2", "h3", "h4", "h5", "h6", "section-title", "panel-title"}

PANEL_CLASSES: set = {"panel", "card", "container", "box", "section-panel", "mat-card", "el-card", "ant-card", "v-card"}

TIMESTAMP_CLASSES: set = {
        "time-ago", "timestamp", "last-updated", "results-timestamp", "update-time", "posted", "modified", "date", "datetime"
    }
TIMESTAMP_ID_PATTERNS: list = [
        r"timestamp", r"time[-_]?ago", r"last[-_]?updated", r"update[-_]?time", r"posted", r"modified", r"date", r"datetime"
    ]
TIMESTAMP_ATTRS: list = [
        "timeago", "datetime", "data-timestamp", "data-updated", "data-date", "data-time", "data-last-updated"
    ]

STRUCTURAL_TAGS: set = {"br", "hr", "wbr", "col", "colgroup", "thead", "tbody", "tfoot", "tr", "th", "td"}

VIEW_BY_PHRASES: list = [
    "district", "precinct", "county", "state", "region", "ward", "township", "municipality", "city", "town",
    "village", "area", "location", "polling place", "ballot types", "ballot type", "contest", "candidate", "party", "office",
    "race", "proposal", "referendum", "amendment", "proposition", "measure", "question", "issue", "result",
    "summary", "detail", "breakdown", "group", "section", "table", "chart", "graph", "map", "visualization",
    "export", "download", "print", "share", "email", "sms", "text", "notification", "alert", "update", "change",
    "revision", "correction", "fix", "patch", "release", "version", "build", "deployment", "environment",
    "platform", "device", "browser", "os", "system", "hardware", "software", "application", "program", "tool",
    "utility", "service", "resource", "asset", "inventory", "catalog", "directory", "index", "list", "menu",
    "navigation", "breadcrumb", "path", "route", "link", "reference", "citation", "annotation", "note", "comment",
    "feedback", "suggestion", "recommendation", "tip", "hint", "help", "support", "faq", "guide", "tutorial",
    "documentation", "manual", "instruction", "example", "sample", "template", "snippet", "code", "script",
    "macro", "function", "method", "procedure", "routine", "operation", "process", "workflow", "pipeline", "job",
    "task", "activity", "event", "trigger", "schedule", "calendar", "timeline", "history", "log", "record",
    "entry", "transaction", "audit", "report"
]

UPDATE_PANEL_KEYWORDS: list = [
    "last updated", "auto-refresh", "updated in real time", "posted", "as of", "timestamp", "date:", "time:",
    "reporting status", "election districts reporting", "fully reported", "incoming ballots", "download reports",
    "export", "media export", "powered by", "results last updated", "percent reporting", "no results yet",
    "ballots counted", "ballots remaining", "ballots cast", "precincts reporting", "precincts counted",
    "precincts remaining", "vote method", "followed results", "search"
] + list(PERCENT_KEYWORDS) + list(TOTAL_KEYWORDS)


def atomic_write_json(obj, path):
    """
    Atomically write JSON to path, keeping only the latest .bak and .tmp.
    - Writes to .tmp first, then moves to final path.
    - If path exists, creates a .bak (removing any old .bak).
    - Cleans up any stray .tmp before/after.
    """
    import os
    path = Path(path)
    backup_path = path.with_suffix(path.suffix + ".bak")
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    # Remove any old .tmp file before starting
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except Exception:
            pass

    # Remove any old .bak file before creating new backup
    if backup_path.exists():
        try:
            backup_path.unlink()
        except Exception:
            pass

    # Write to .tmp path
    with open(tmp_path, "wb") as tf:
        tf.write(orjson.dumps(obj, option=orjson.OPT_INDENT_2))

    # If the main file exists, back it up
    if path.exists():
        shutil.copy2(path, backup_path)

    # --- Fix: If the target file exists and is locked, try to close it or retry ---
    import time
    for _ in range(3):
        try:
            shutil.move(str(tmp_path), str(path))
            break
        except (OSError, PermissionError, FileExistsError) as e:
            # Try to remove the target file if possible (only if you are sure it's safe)
            try:
                os.remove(str(path))
            except Exception:
                pass
            time.sleep(0.5)
    else:
        raise RuntimeError(f"Could not move {tmp_path} to {path} after several attempts.")

    # Clean up any stray .tmp (should not exist, but just in case)
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except Exception:
            pass

# --- Extend/Modify Functions ---
def extend_panel_tags(new_tags: List[str]):
    global PANEL_TAGS
    PANEL_TAGS |= set(t.lower() for t in new_tags)

def extend_heading_tags(new_tags: List[str]):
    global HEADING_TAGS
    HEADING_TAGS |= set(t.lower() for t in new_tags)

def extend_html_tags(new_tags: List[str]):
    global HTML_TAGS
    HTML_TAGS |= set(t.lower() for t in new_tags)

def extend_custom_attr_patterns(new_patterns: List[str]):
    global CUSTOM_ATTR_PATTERNS
    for pat in new_patterns:
        if isinstance(pat, str):
            CUSTOM_ATTR_PATTERNS.append(re.compile(pat))
        else:
            CUSTOM_ATTR_PATTERNS.append(pat)

def extend_location_keywords(new_keywords: List[str]):
    global LOCATION_KEYWORDS
    LOCATION_KEYWORDS |= set(k.lower() for k in new_keywords)

def extend_candidate_keywords(new_keywords: List[str]):
    global CANDIDATE_KEYWORDS
    CANDIDATE_KEYWORDS |= set(k.lower() for k in new_keywords)

def extend_ballot_types(new_types: List[str]):
    global BALLOT_TYPES
    BALLOT_TYPES.extend([t for t in new_types if t not in BALLOT_TYPES])

def safe_join(base, *paths):
    final_path = os.path.abspath(os.path.join(base, *paths))
    if not final_path.startswith(os.path.abspath(base)):
        logger.debug(f"DEBUG: Attempted to join {paths} to base {base} -> {final_path}")
        raise ValueError("Attempted Path Traversal Detected!")
    return final_path

# --- Context Library Integration ---
def robust_orjson_loads(val):
    if isinstance(val, bytes):
        return orjson.loads(val)
    elif isinstance(val, str):
        return orjson.loads(val.encode("utf-8"))
    else:
        raise TypeError(f"Cannot decode type {type(val)} with orjson")

def load_context_library(path=CONTEXT_LIBRARY_PATH) -> dict:
    """
    Loads the context library robustly:
    - If missing, creates with default structure.
    - If empty or corrupt, backs up and re-initializes.
    - If missing keys, adds them (preserving existing data).
    - Extends dynamic sets with loaded values.
    """
    safe_path = path
    os.makedirs(os.path.dirname(safe_path), exist_ok=True)

    def merge_defaults(existing, defaults):
        changed = False
        for k, v in defaults.items():
            if k not in existing:
                existing[k] = v
                changed = True
            elif isinstance(v, dict) and isinstance(existing[k], dict):
                if merge_defaults(existing[k], v):
                    changed = True
        return changed

    # If file does not exist or is empty, create with defaults
    if not os.path.exists(safe_path) or os.path.getsize(safe_path) == 0:
        context_lib = {
            "panel_tags": list(PANEL_TAGS),
            "heading_tags": list(HEADING_TAGS),
            "custom_attr_patterns": [pat.pattern for pat in CUSTOM_ATTR_PATTERNS],
            "location_keywords": list(LOCATION_KEYWORDS),
            "candidate_keywords": list(CANDIDATE_KEYWORDS),
            "ballot_types": list(BALLOT_TYPES),
            **DEFAULT_STRUCTURE
        }
        with open(safe_path, "wb") as f:
            f.write(orjson.dumps(context_lib, option=orjson.OPT_INDENT_2))
        return context_lib

    # Try to load, back up and re-init if corrupt
    try:
        with open(safe_path, "rb") as f:
            data = f.read()
            if not data:
                # Empty file, treat as missing
                context_lib = {
                    "panel_tags": list(PANEL_TAGS),
                    "heading_tags": list(HEADING_TAGS),
                    "custom_attr_patterns": [pat.pattern for pat in CUSTOM_ATTR_PATTERNS],
                    "location_keywords": list(LOCATION_KEYWORDS),
                    "candidate_keywords": list(CANDIDATE_KEYWORDS),
                    "ballot_types": list(BALLOT_TYPES),
                    **DEFAULT_STRUCTURE
                }
                with open(safe_path, "wb") as fw:
                    fw.write(orjson.dumps(context_lib, option=orjson.OPT_INDENT_2))
                return context_lib
            context_lib = robust_orjson_loads(data)
    except Exception as e:
        # Backup corrupt file before overwriting
        backup_path = safe_path + ".corrupt"
        try:
            os.rename(safe_path, backup_path)
        except Exception:
            pass
        context_lib = {
            "panel_tags": list(PANEL_TAGS),
            "heading_tags": list(HEADING_TAGS),
            "custom_attr_patterns": [pat.pattern for pat in CUSTOM_ATTR_PATTERNS],
            "location_keywords": list(LOCATION_KEYWORDS),
            "candidate_keywords": list(CANDIDATE_KEYWORDS),
            "ballot_types": list(BALLOT_TYPES),
            **DEFAULT_STRUCTURE
        }
        with open(safe_path, "wb") as f:
            f.write(orjson.dumps(context_lib, option=orjson.OPT_INDENT_2))
        return context_lib

    # Merge in any missing keys from default (preserve existing data)
    if merge_defaults(context_lib, DEFAULT_STRUCTURE):
        save_context_library(context_lib, safe_path)

    # Extend dynamic sets with loaded values
    if "panel_tags" in context_lib:
        extend_panel_tags(context_lib["panel_tags"])
    if "heading_tags" in context_lib:
        extend_heading_tags(context_lib["heading_tags"])
    if "custom_attr_patterns" in context_lib:
        extend_custom_attr_patterns(context_lib["custom_attr_patterns"])
    if "location_keywords" in context_lib:
        extend_location_keywords(context_lib["location_keywords"])
    if "candidate_keywords" in context_lib:
        extend_candidate_keywords(context_lib["candidate_keywords"])
    if "ballot_types" in context_lib:
        extend_ballot_types(context_lib["ballot_types"])

    return context_lib
    
def update_context_library(path, update_fn):
    """
    Safely update the context library at `path` by applying `update_fn(library)`.
    If a dict is passed instead of a function, it will update the library with that dict.
    """
    from ..Context_Integration.context_organizer import clean_for_json  # Import here to avoid circular import at module level
    with _CONTEXT_LOCK:
        lib = load_context_library(path)
        # Accept either a function or a dict
        if isinstance(update_fn, dict):
            lib.update(update_fn)
        else:
            update_fn(lib)
        lib = clean_for_json(lib)  # <-- Ensure all sets are converted before saving
        save_context_library(lib, path)

def file_hash(path):
    """Return SHA256 hash of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
       
def backup_context_library(path=CONTEXT_LIBRARY_PATH, max_backups=5):
    """
    Make a timestamped backup of the context library before overwriting,
    but only if the content has changed. Keep only the most recent `max_backups` backups.
    """
    if not os.path.exists(path):
        return

    dir_ = os.path.dirname(path)
    base = os.path.basename(path)
    # Only match timestamped .bak files
    backups = sorted(
        [f for f in os.listdir(dir_) if f.startswith(base + ".") and f.endswith(".bak")],
        reverse=True
    )
    current_hash = file_hash(path)
    if backups:
        last_backup_path = os.path.join(dir_, backups[0])
        try:
            if file_hash(last_backup_path) == current_hash:
                # No change since last backup
                return
        except Exception:
            pass

    # Remove any non-timestamped .bak (legacy or accidental)
    legacy_bak = os.path.join(dir_, base + ".bak")
    if os.path.exists(legacy_bak):
        try:
            os.remove(legacy_bak)
        except Exception:
            pass

    # Make new backup
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = f"{path}.{timestamp}.bak"
    shutil.copy2(path, backup_path)

    # Prune old backups (keep only the most recent max_backups)
    backups = sorted(
        [f for f in os.listdir(dir_) if f.startswith(base + ".") and f.endswith(".bak")],
        reverse=True
    )
    for old in backups[max_backups:]:
        try:
            os.remove(os.path.join(dir_, old))
        except Exception:
            pass

def save_context_library(lib, path=None):
    """
    Robustly save the context library:
    - Always makes a timestamped backup before writing.
    - Writes atomically (temp file, then replace).
    - Never truncates or loses data on failure.
    """
    if path is None:
        path = CONTEXT_LIBRARY_PATH
    safe_path = safe_join(BASE_DIR, os.path.relpath(path, BASE_DIR))
    backup_context_library(safe_path)
    data = orjson.dumps(lib, option=orjson.OPT_INDENT_2)
    # Write to a temp file first
    dir_name = os.path.dirname(safe_path)
    with tempfile.NamedTemporaryFile("wb", dir=dir_name, delete=False) as tf:
        tf.write(data)
        temp_path = tf.name
    # Atomically replace the original file
    os.replace(temp_path, safe_path)

def merge_and_save_context_library(partial_dict, path=CONTEXT_LIBRARY_PATH):
    """
    Safely merge a partial dict into the context library and save atomically.
    """
    lib = load_context_library(path)
    lib.update(partial_dict)
    save_context_library(lib, path)

def update_context_library_field(key, value, path=CONTEXT_LIBRARY_PATH):
    """
    Safely update a top-level key in the context library.
    """
    lib = load_context_library(path)
    old_value = lib.get(key, None)
    lib[key] = value
    save_context_library(lib, path)
    # Optionally log the change
    logger.info(f"Updated context_library field '{key}': {old_value} -> {value}")

def update_domain_selector_cache(domain, selector, label, success=True):
    lib = load_context_library()
    domain_selectors = lib.setdefault("domain_selectors", {})
    entry = {
        "selector": selector,
        "label": label,
        "success_count": 1 if success else 0,
        "last_used": datetime.now(timezone.utc).isoformat()
    }
    found = False
    for e in domain_selectors.get(domain, []):
        if e["selector"] == selector:
            e["success_count"] += 1 if success else 0
            e["last_used"] = entry["last_used"]
            found = True
            break
    if not found:
        domain_selectors.setdefault(domain, []).append(entry)
    # Only update the domain_selectors field in the context library
    update_context_library_field("domain_selectors", domain_selectors)

def get_domain_selectors(domain):
    lib = load_context_library()
    return lib.get("domain_selectors", {}).get(domain, [])

def log_selector_attempt(domain, selector, label, success):
    lib = load_context_library()
    attempts = lib.setdefault("selector_attempts", [])
    attempts.append({
        "domain": domain,
        "selector": selector,
        "label": label,
        "success": success,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    update_context_library_field("selector_attempts", attempts)

# --- Unknown Tag/Attr Logging for ML/LLM Feedback ---
UNKNOWN_TAGS_LOG = set()
UNKNOWN_ATTRS_LOG = set()

def _get_log_path(filename: str) -> str:
    # Use the centralized LOG_DIR for all logs
    os.makedirs(LOG_DIR, exist_ok=True)
    return os.path.join(LOG_DIR, filename)

def log_unknown_tag(tag: str, context_library):
    known_tags = set(context_library.get("panel_tags", []) + context_library.get("heading_tags", []) + context_library.get("html_tags", []))
    if tag not in known_tags:
        UNKNOWN_TAGS_LOG.add(tag)
        try:
            log_path = _get_log_path("unknown_tags_log.jsonl")
            with open(log_path, "ab") as f:
                f.write(orjson.dumps({"tag": tag}) + b"\n")
        except Exception:
            pass

def log_unknown_attr(attr: str, context_library):
    # Get patterns from context_library or fallback to static set
    pattern_strings = context_library.get("custom_attr_patterns", []) if context_library else []
    patterns = [re.compile(p) for p in pattern_strings] if pattern_strings else CUSTOM_ATTR_PATTERNS

    # Always allow common dynamic attributes
    if attr.startswith("data-") or attr.startswith("aria-") or attr == "role":
        return

    # Only log if it doesn't match any known pattern
    if not any(pat.match(attr) for pat in patterns):
        UNKNOWN_ATTRS_LOG.add(attr)
        try:
            log_path = _get_log_path("unknown_attrs_log.jsonl")
            with open(log_path, "ab") as f:
                f.write(orjson.dumps({"attr": attr}) + b"\n")
        except Exception:
            pass

# --- ML/LLM Feedback Integration Example ---
def integrate_llm_feedback(new_panel_tags=None, new_heading_tags=None, new_attr_patterns=None, new_location_keywords=None, new_candidate_keywords=None, new_ballot_types=None):
    if new_panel_tags:
        extend_panel_tags(new_panel_tags)
    if new_heading_tags:
        extend_heading_tags(new_heading_tags)
    if new_attr_patterns:
        extend_custom_attr_patterns(new_attr_patterns)
    if new_location_keywords:
        extend_location_keywords(new_location_keywords)
    if new_candidate_keywords:
        extend_candidate_keywords(new_candidate_keywords)
    if new_ballot_types:
        extend_ballot_types(new_ballot_types)
    save_context_library()

# --- Load context library at import time ---
load_context_library()

_segment_label_cache = {}

def normalize_segment_text(text: str) -> str:
    """Normalize segment text for canonical lookup (lowercase, strip, collapse spaces)."""
    if not text:
        return ""
    return " ".join(text.lower().strip().split())

def get_canonical_segment_label(text: str) -> str:
    """Return canonical label for normalized segment text, or None if not found."""
    norm = normalize_segment_text(text)
    return CANONICAL_SEGMENT_LABELS.get(norm)

def cache_segment_label(text: str, label: str):
    norm = normalize_segment_text(text)
    _segment_label_cache[norm] = label

def get_cached_segment_label(text: str) -> str:
    norm = normalize_segment_text(text)
    return _segment_label_cache.get(norm)

# --- Self-Heal Mode ---
def self_heal_context_library(max_retries=3, cooldown=2):
    """Self-heal: scan for misaligned NER, run correction bot, reload context library, repeat until clean or max_retries."""
    scan_script = os.path.join(os.path.dirname(__file__), "scan_misaligned_ner.py")
    for attempt in range(1, max_retries + 1):
        logger.warning(f"\n[LIBRARIAN SELF-HEAL] Attempt {attempt}...")
        scan_cmd = [sys.executable, scan_script, "--jsonl", "log/spacy_ner_train_data.jsonl"]
        scan_result = subprocess.run(scan_cmd, check=True, cwd=PROJECT_ROOT)
        if scan_result.returncode == 0:
            logger.info("[LIBRARIAN SELF-HEAL] Data is clean. Exiting self-heal mode.")
            return 0
        logger.warning("[LIBRARIAN SELF-HEAL] Misalignments found. Launching manual_correction_bot...")
        bot_cmd = [sys.executable, "-m", "webapp.parser.bots.manual_correction_bot", "--enhanced"]
        subprocess.run(bot_cmd, check=True, cwd=PROJECT_ROOT)
        logger.warning(f"[LIBRARIAN SELF-HEAL] Sleeping {cooldown}s before rescanning...")
        time.sleep(cooldown)
    logger.info("[LIBRARIAN SELF-HEAL] Max retries reached. Some misalignments may remain.")
    return 2

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Librarian utility for context library management.")
    parser.add_argument("--self-heal", action="store_true", help="Loop: scan -> correct -> rescan until clean or max retries")
    parser.add_argument("--max-retries", type=int, default=3, help="Max self-heal attempts")
    parser.add_argument("--cooldown", type=int, default=2, help="Seconds to wait between self-heal attempts")
    args = parser.parse_args()
    if args.self_heal:
        sys.exit(self_heal_context_library(args.max_retries, args.cooldown))
# --- Export all sets for use in other modules ---
__all__ = [
    "_CANONICAL_STATE_ABBR", "KNOWN_STATE_TO_COUNTY_MAP", "STATE_ABBR", "STATE_MODULE_MAP", "HTML_TAGS", "PANEL_TAGS",
     "TABLE_TAGS", "STATE_TAGS", "HEADING_TAGS", "EXTRA_HEADING_TAGS","CUSTOM_ATTR_PATTERNS", "DISTRICT_REGEX",
    "BALLOT_TYPES", "BALLOT_TYPES_SORT_ORDER", "LOCATION_KEYWORDS", "PERCENT_KEYWORDS", "TOTAL_KEYWORDS",
    "MISC_FOOTER_KEYWORDS", "CANDIDATE_KEYWORDS", "PARTY_KEYWORDS", "LOCATION_ABBREVIATIONS", "ELECTION_TYPES", "CONTEST_KEYWORDS",
    "extend_panel_tags", "extend_heading_tags", "extend_html_tags", "extend_custom_attr_patterns",
    "extend_location_keywords", "extend_candidate_keywords", "extend_ballot_types",
    "log_unknown_tag", "log_unknown_attr", "integrate_llm_feedback", "CANONICAL_SEGMENT_LABELS", 
    "normalize_segment_text", "get_canonical_segment_label", "cache_segment_label", "get_cached_segment_label",
    "ROOT_CONTAINER_TAGS", "ALWAYS_IGNORE_TAGS", "ALWAYS_IGNORE_CLASSES", "ALWAYS_IGNORE_IDS", "ICON_CLASSES", "ICON_TAGS", "BUTTON_CLASSES",
    "HEADING_CLASSES", "PANEL_CLASSES", "TIMESTAMP_CLASSES", "STRUCTURAL_TAGS", "TIMESTAMP_ID_PATTERNS", "TIMESTAMP_ATTRS",
    "STRUCTURAL_TAGS", "VIEW_BY_PHRASES", "UPDATE_PANEL_KEYWORDS", "KNOWN_COUNTY_TO_PRECINCTS_MAP",
    "NOISY_LABEL_PATTERNS", "PRECINCT_HEADER_PATTERNS", "SELECTORS", "CONTEST_PANEL_TAGS",
]

"""B17_REC9B_REC2 shadow-only, pure election-record normalization contracts.

No source acquisition, database access, browser interaction or production wiring.
Non-empty source fields are preserved verbatim; unknown is not numeric zero.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import re
from urllib.parse import urlparse


class NormalizationContractError(ValueError):
    """Input metadata cannot be interpreted without guessing."""


_UNKNOWN_TOKENS = frozenset(("", "-", "--", "—", "N/A", "NA", "NULL"))
_UNSIGNED_INTEGER = re.compile(r"(?:0|[1-9][0-9]*|[1-9][0-9]{0,2}(?:,[0-9]{3})+)")
_KNOWN_FILE_SUFFIXES = {".csv": "CSV", ".tsv": "TSV", ".xlsx": "XLSX", ".xls": "XLS", ".json": "JSON", ".pdf": "PDF", ".txt": "TXT"}


def parse_explicit_vote_cell(value: Any) -> int | None:
    """Return a nonnegative source-supplied integer, or None for unavailable.

    Numeric zero is only returned when the source cell explicitly contains zero.
    No totals, fallback zeros or inferred ballot methods are created.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise NormalizationContractError("BOOLEAN_IS_NOT_A_VOTE_CELL")
    if isinstance(value, int):
        if value < 0:
            raise NormalizationContractError("NEGATIVE_VOTE_CELL")
        return value
    if not isinstance(value, str):
        raise NormalizationContractError("UNSUPPORTED_VOTE_CELL_TYPE")
    token = value.strip()
    if token.upper() in _UNKNOWN_TOKENS:
        return None
    if not _UNSIGNED_INTEGER.fullmatch(token):
        raise NormalizationContractError("VOTE_CELL_UNRECOGNIZED")
    return int(token.replace(",", ""))


def preserve_jurisdiction(*, source_scope: Any, source_state: Any,
                          row_state: Any = None, row_county: Any = None,
                          row_municipality: Any = None, row_ward: Any = None,
                          row_election_district: Any = None, row_precinct: Any = None,
                          navigation_county_hint: Any = None) -> dict[str, Any]:
    """Keep source coverage, explicit row geography, and hints separate.

    A precinct-looking district, navigation county guess, or statewide source
    scope is never promoted to a different explicit row field.
    """
    def explicit(value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise NormalizationContractError("JURISDICTION_FIELD_NOT_TEXT")
        return value if value.strip() else None

    row = {
        "state": explicit(row_state),
        "county": explicit(row_county),
        "municipality": explicit(row_municipality),
        "ward": explicit(row_ward),
        "election_district": explicit(row_election_district),
        "precinct": explicit(row_precinct),
    }
    if row["precinct"] is not None:
        grain = "precinct"
    elif row["election_district"] is not None:
        grain = "election_district"
    elif row["ward"] is not None:
        grain = "ward"
    elif row["municipality"] is not None:
        grain = "municipality"
    elif row["county"] is not None:
        grain = "county"
    elif row["state"] is not None:
        grain = "state"
    else:
        grain = "unknown"
    return {
        "source_scope": explicit(source_scope),
        "source_state": explicit(source_state),
        "row_jurisdiction": row,
        "row_granularity_from_explicit_labels": grain,
        "navigation_county_hint_not_authoritative": explicit(navigation_county_hint),
        "precinct_result_certified": False,
    }


def candidate_method_matrix(*, candidates: Sequence[str], methods: Sequence[str],
                            observed: Mapping[tuple[str, str], Any]) -> dict[str, int | None]:
    """Materialize candidate x source-native method keys without inventing zero.

    This helper is a proposed contract, not a production CSV exporter.
    """
    if isinstance(candidates, (str, bytes)) or isinstance(methods, (str, bytes)):
        raise NormalizationContractError("SEQUENCE_EXPECTED")
    if not candidates or not methods or any(not isinstance(x, str) or not x.strip() for x in (*candidates, *methods)):
        raise NormalizationContractError("EMPTY_CANDIDATE_OR_METHOD")
    if len(candidates) != len(set(candidates)) or len(methods) != len(set(methods)):
        raise NormalizationContractError("DUPLICATE_CANDIDATE_OR_METHOD")
    expected = {(c, m) for c in candidates for m in methods}
    if any(key not in expected for key in observed):
        raise NormalizationContractError("UNRECOGNIZED_CANDIDATE_METHOD")
    return {f"{candidate} - {method}": parse_explicit_vote_cell(observed.get((candidate, method)))
            for candidate in candidates for method in methods}


def propose_source_route(*, registry_source_id: str, source_url: str,
                         catalog_format: str, catalog_state: str,
                         catalog_scope: str) -> dict[str, Any]:
    """Read-only proposal; never activates, publishes, downloads or dispatches.

    URL filename and host are hints, not proof of MIME/content/vendor identity.
    """
    if not isinstance(registry_source_id, str) or not registry_source_id.strip():
        raise NormalizationContractError("SOURCE_ID_REQUIRED")
    if not isinstance(source_url, str) or not source_url.startswith("https://"):
        raise NormalizationContractError("SOURCE_URL_METADATA_INVALID")
    parsed = urlparse(source_url)
    if not parsed.hostname:
        raise NormalizationContractError("SOURCE_HOST_MISSING")
    filename = parsed.path.rsplit("/", 1)[-1]
    lower = filename.lower()
    suffix = next((ext for ext in sorted(_KNOWN_FILE_SUFFIXES, key=len, reverse=True) if lower.endswith(ext)), "")
    extension_format = _KNOWN_FILE_SUFFIXES.get(suffix)
    format_conflict = bool(extension_format and extension_format != catalog_format.upper())
    vendor_hint = "enhanced_voting" if parsed.hostname.lower() == "app.enhancedvoting.com" else None
    return {
        "registry_source_id": registry_source_id,
        "source_url": source_url,
        "catalog_format_unchanged": catalog_format,
        "catalog_state_unchanged": catalog_state,
        "catalog_scope_unchanged": catalog_scope,
        "filename_suffix_observed": suffix or None,
        "filename_format_hint": extension_format,
        "format_conflict_requires_review": format_conflict,
        "vendor_hint_unverified": vendor_hint,
        "source_content_schema_verified": False,
        "precinct_granularity_verified": False,
        "route_activation_authorized": False,
        "proposal_only": True,
    }

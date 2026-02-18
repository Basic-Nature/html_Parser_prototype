"""
Election Data Standardizer
Handles County/District, Candidate Name, Party, Vote Types, Write-In flags, and FEC ID validation.
Flags data for manual review when standardization cannot be automated.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

# Union type for multi-return annotations
try:
    from typing import Tuple
except ImportError:
    pass


class DataQualityFlag(str, Enum):
    """Data quality issues that require manual review"""
    MISSING_FEC_ID = "missing_fec_id"
    MISSING_PARTY = "missing_party_code"
    PARTY_MISMATCH = "party_ballot_vs_fec_mismatch"
    CANDIDATE_NAME_UNCLEAR = "candidate_name_unclear"
    VOTE_TYPE_AMBIGUOUS = "vote_type_ambiguous"
    WRITE_IN_UNCERTAIN = "write_in_uncertain"
    VOTE_TOTAL_MISMATCH = "vote_total_mismatch"
    DUPLICATE_CANDIDATE = "duplicate_candidate"


@dataclass
class StandardizationResult:
    """Result of standardization attempt for a single record"""
    success: bool
    standardized_data: Dict[str, Any] = field(default_factory=dict)
    flags: List[DataQualityFlag] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    original_data: Dict[str, Any] = field(default_factory=dict)
    requires_manual_review: bool = False
    
    def add_flag(self, flag: DataQualityFlag, reason: str = ""):
        """Add a quality flag"""
        self.flags.append(flag)
        if reason:
            self.warnings.append(f"{flag.value}: {reason}")
        self.requires_manual_review = True


class PartyCodeMapper:
    """Map ballot party names to standardized FEC party codes"""
    
    # Ballot party → FEC code mapping (case-insensitive)
    BALLOT_TO_FEC = {
        # Democratic variants
        'democratic': 'DEM',
        'democrat': 'DEM',
        'dem': 'DEM',
        # Republican variants
        'republican': 'REP',
        'gop': 'REP',
        'rep': 'REP',
        # Libertarian variants
        'libertarian': 'LIB',
        'lib': 'LIB',
        # Green variants
        'green': 'GRE',
        'green party': 'GRE',
        'gre': 'GRE',
        # Independent variants
        'independent': 'IND',
        'ind': 'IND',
        'unaffiliated': 'IND',
        # Constitution
        'constitution': 'CON',
        'constitutional': 'CON',
        # American Independent
        'american independent': 'AIP',
        'american ind': 'AIP',
        # Working Families
        'working families': 'WFP',
        'wfp': 'WFP',
        # Progressive
        'progressive': 'PRO',
        # Justice Party
        'justice': 'JUS',
        'justice party': 'JUS',
        # Write-In/None
        'write-in': None,
        'write in': None,
        'writein': None,
        'none': None,
        'n/a': None,
        '': None,
    }
    
    # FEC code validation set
    VALID_FEC_CODES = {
        'DEM', 'REP', 'LIB', 'GRE', 'IND', 'CON', 'AIP', 'WFP', 'PRO', 'JUS',
        'CRV', 'BFA', 'NP', 'NNV', 'AVP', 'GKH', 'CON', 'PRO'
    }
    
    @classmethod
    def map_party(cls, ballot_party: Optional[str]) -> Tuple[Optional[str], bool]:
        """
        Map ballot party to FEC code.
        Returns: (fec_code, is_mapped)
        """
        if not ballot_party or not isinstance(ballot_party, str):
            return None, False
        
        normalized = ballot_party.strip().lower()
        fec_code = cls.BALLOT_TO_FEC.get(normalized)
        
        if fec_code is not None:
            return fec_code, True
        
        # Try to find partial match
        for ballot, code in cls.BALLOT_TO_FEC.items():
            if ballot in normalized or normalized in ballot:
                return code, True
        
        return None, False


class CandidateNameStandardizer:
    """Standardize candidate names to LASTNAME, FIRSTNAME format"""
    
    # Titles and suffixes to handle
    PREFIXES = {'Mr', 'Mrs', 'Ms', 'Dr', 'Rev', 'Hon', 'Gen', 'Col', 'Lt', 'Maj', 'Sgt'}
    SUFFIXES = {'Jr', 'Sr', 'II', 'III', 'IV', 'V', 'Esq', 'PhD', 'MD', 'DDS'}
    
    # Words indicating multiple candidates (joint tickets)
    JOINT_INDICATORS = {'and', '&', 'with', '/'}
    
    @classmethod
    def standardize(cls, name: Optional[str]) -> Tuple[str, bool]:
        """
        Standardize candidate name to LASTNAME, FIRSTNAME format.
        Returns: (standardized_name, is_already_formatted)
        """
        if not name or not isinstance(name, str):
            return "", False
        
        name = name.strip()
        
        if not name:
            return "", False
        
        # Check if already in LASTNAME, FIRSTNAME format
        if ',' in name:
            parts = name.split(',')
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                return name.strip(), True  # Already formatted
        
        # Handle joint tickets (e.g., "OBAMA, BARACK / JOSEPH R. BIDEN")
        if any(indicator in name.upper() for indicator in cls.JOINT_INDICATORS):
            # Return as-is; requires manual review
            return name.strip(), False
        
        # Check if parenthetical (e.g., "ROss C ANDERSON (WRITE IN)")
        if '(' in name and ')' in name:
            # Extract main name
            main_name = name.split('(')[0].strip()
            suffix = f" ({name.split('(')[1]}"
            return cls._format_name(main_name) + suffix, False
        
        return cls._format_name(name), False
    
    @classmethod
    def _format_name(cls, name: str) -> str:
        """Convert FIRSTNAME LASTNAME to LASTNAME, FIRSTNAME format"""
        parts = [p.strip() for p in name.split() if p.strip()]
        
        if not parts:
            return name
        
        if len(parts) == 1:
            # Single word name
            return parts[0]
        
        # Assume last part is surname (unless it's a suffix)
        if len(parts) > 1 and parts[-1] in cls.SUFFIXES:
            if len(parts) > 2:
                surname = parts[-2]
                firstname_parts = parts[:-2] + [parts[-1]]
                return f"{surname}, {' '.join(firstname_parts)}"
        
        surname = parts[-1]
        firstname = ' '.join(parts[:-1])
        return f"{surname}, {firstname}"


class VoteTypeStandardizer:
    """Standardize vote type columns"""
    
    VOTE_TYPE_COLUMNS = {
        'Uncategorized Votes': 'uncategorized',
        'Early Votes': 'early',
        'Early Voting': 'early',
        'EarlyVotes': 'early',
        'Election Day Votes': 'election_day',
        'Election Day': 'election_day',
        'Election Day Vote': 'election_day',
        'Mail in Votes': 'mail',
        'Mail-In Votes': 'mail',
        'Mail': 'mail',
        'Mailing': 'mail',
        'Provisional Votes': 'provisional',
        'Provisional': 'provisional',
    }
    
    @classmethod
    def standardize_vote_types(cls, vote_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[DataQualityFlag]]:
        """
        Standardize vote type columns.
        Marks all-blank as UNCATEGORIZED.
        Marks mixed-blank as N/A (requires manual review).
        
        Returns: (standardized_vote_data, flags)
        """
        flags = []
        standardized = {}
        
        # Extract vote values
        votes = {}
        for col_name, std_name in cls.VOTE_TYPE_COLUMNS.items():
            value = vote_data.get(col_name)
            votes[std_name] = cls._parse_vote_count(value)
        
        # Check if all votes are blank/zero
        non_zero_votes = {k: v for k, v in votes.items() if v is not None and v > 0}
        
        if not non_zero_votes:
            # All blank/zero
            standardized['vote_type_classification'] = 'UNCATEGORIZED'
            standardized['vote_type_note'] = 'All vote type columns are blank'
        else:
            # Check which columns have data
            filled_columns = [k for k, v in votes.items() if v is not None and v > 0]
            
            if len(filled_columns) == 1:
                # Single vote type
                standardized['vote_type_classification'] = filled_columns[0].upper()
            else:
                # Multiple vote types - store as breakdown
                standardized['vote_type_classification'] = 'MIXED'
                standardized['vote_breakdown'] = {k: v for k, v in votes.items() if v is not None}
                flags.append(DataQualityFlag.VOTE_TYPE_AMBIGUOUS)
        
        standardized.update(votes)
        return standardized, flags
    
    @classmethod
    def _parse_vote_count(cls, value: Any) -> Optional[int]:
        """Parse vote count from various formats"""
        if value is None or value == '':
            return None
        
        if isinstance(value, int):
            return value if value > 0 else None
        
        if isinstance(value, str):
            # Remove commas and whitespace
            cleaned = value.strip().replace(',', '')
            if not cleaned:
                return None
            
            try:
                count = int(cleaned)
                return count if count > 0 else None
            except ValueError:
                return None
        
        return None


class CountyDistrictStandardizer:
    """Standardize County/District names"""
    
    # Trailing terms to remove
    TRAILING_TERMS = {' County', ' District', ' County District', ' Parish', ' Borough', ' City'}
    
    @classmethod
    def standardize(cls, location: Optional[str]) -> str:
        """Remove trailing county/district terms"""
        if not location or not isinstance(location, str):
            return ""
        
        result = location.strip()
        
        # Remove trailing terms (case-insensitive)
        for term in cls.TRAILING_TERMS:
            if result.lower().endswith(term.lower()):
                result = result[:-len(term)].strip()
                break
        
        return result


class WriteInFlagStandardizer:
    """Standardize Write-In flags"""
    
    @classmethod
    def standardize(cls, is_write_in: Any, candidate_name: Optional[str] = None) -> Tuple[bool, List[DataQualityFlag]]:
        """
        Standardize write-in flag.
        Default FALSE if blank.
        Flag TRUE if candidate name suggests write-in but flag is FALSE.
        
        Returns: (is_write_in_bool, flags)
        """
        flags = []
        
        # Parse as boolean
        if isinstance(is_write_in, bool):
            result = is_write_in
        elif isinstance(is_write_in, str):
            normalized = is_write_in.strip().lower()
            if normalized in {'true', 'yes', '1', 'y', 'checked', 'x'}:
                result = True
            elif normalized in {'false', 'no', '0', 'n', 'unchecked', ''}:
                result = False
            else:
                result = False  # Default to FALSE
                flags.append(DataQualityFlag.WRITE_IN_UNCERTAIN)
        else:
            result = False  # Default to FALSE
        
        # Cross-check with candidate name
        if candidate_name and '(WRITE IN)' in candidate_name.upper():
            if not result:
                flags.append(DataQualityFlag.WRITE_IN_UNCERTAIN)
                result = True  # Override to TRUE
        
        return result, flags


class ElectionDataStandardizer:
    """Main orchestrator for election data standardization"""
    
    def __init__(self):
        self.party_mapper = PartyCodeMapper()
        self.name_standardizer = CandidateNameStandardizer()
        self.vote_standardizer = VoteTypeStandardizer()
        self.county_standardizer = CountyDistrictStandardizer()
        self.writein_standardizer = WriteInFlagStandardizer()
    
    def standardize_record(self, raw_record: Dict[str, Any]) -> StandardizationResult:
        """
        Standardize a single election record.
        
        Args:
            raw_record: Dict with keys matching the Google Sheets columns
            
        Returns:
            StandardizationResult with standardized data and flags
        """
        result = StandardizationResult(success=True, original_data=raw_record.copy())
        standardized = {}
        
        # 1. County/District
        county = raw_record.get('County') or raw_record.get('County/District', '')
        standardized['county'] = self.county_standardizer.standardize(county)
        
        # 2. Candidate Name
        candidate_raw = raw_record.get('Ballot Candidate Name', '')
        candidate_std, is_formatted = self.name_standardizer.standardize(candidate_raw)
        standardized['candidate_name'] = candidate_std
        
        if not is_formatted and '/' in candidate_std:
            # Joint ticket
            result.add_flag(DataQualityFlag.CANDIDATE_NAME_UNCLEAR, 
                          f"Joint ticket or compound name: {candidate_std}")
        
        # 3. Write-In Flag
        writein_raw = raw_record.get('Write-In', raw_record.get('Is Write In', False))
        is_writein, writein_flags = self.writein_standardizer.standardize(writein_raw, candidate_std)
        standardized['is_write_in'] = is_writein
        result.flags.extend(writein_flags)
        
        # 4. Party Mapping
        ballot_party = raw_record.get('Ballot Party', '')
        fec_party, is_mapped = self.party_mapper.map_party(ballot_party)
        standardized['ballot_party'] = ballot_party
        standardized['fec_party'] = fec_party
        
        if ballot_party and not fec_party:
            result.add_flag(DataQualityFlag.MISSING_PARTY,
                          f"Cannot map ballot party '{ballot_party}' to FEC code")
        
        # 5. FEC ID
        fec_id = raw_record.get('FEC ID', '').strip() if raw_record.get('FEC ID') else ''
        standardized['fec_id'] = fec_id if fec_id else None
        
        # Flag missing FEC ID for non-write-ins with mapped party
        if not fec_id and not is_writein and fec_party:
            result.add_flag(DataQualityFlag.MISSING_FEC_ID,
                          f"No FEC ID for candidate: {candidate_std}, party: {fec_party}")
        
        # 6. Vote Types
        vote_cols = {
            'Uncategorized Votes': raw_record.get('Uncategorized Votes'),
            'Early Votes': raw_record.get('Early Votes') or raw_record.get('EarlyVotes'),
            'Election Day Votes': raw_record.get('Election Day Votes'),
            'Mail in Votes': raw_record.get('Mail in Votes'),
            'Provisional Votes': raw_record.get('Provisional Votes'),
        }
        vote_std, vote_flags = self.vote_standardizer.standardize_vote_types(vote_cols)
        standardized.update(vote_std)
        result.flags.extend(vote_flags)
        
        # 7. Calculate total votes
        total_votes = sum(
            v for k, v in vote_std.items() 
            if k in VoteTypeStandardizer.VOTE_TYPE_COLUMNS.values() and isinstance(v, int)
        )
        standardized['total_votes'] = total_votes
        
        # 8. Copy over other fields
        for key in ['Year', 'Office', 'State', 'RACE ID', 'Source Data URL']:
            if key in raw_record:
                standardized[key.lower().replace(' ', '_')] = raw_record[key]
        
        result.standardized_data = standardized
        
        # Mark for manual review if any flags exist
        if result.flags:
            result.requires_manual_review = True
        
        return result
    
    def standardize_batch(self, raw_records: List[Dict[str, Any]]) -> Tuple[List[StandardizationResult], Dict[str, int]]:
        """
        Standardize multiple records.
        
        Returns:
            (list of results, summary statistics)
        """
        results = []
        stats = {
            'total': len(raw_records),
            'success': 0,
            'flagged_for_review': 0,
            'flags_by_type': {},
        }
        
        for record in raw_records:
            result = self.standardize_record(record)
            results.append(result)
            
            if result.success:
                stats['success'] += 1
            
            if result.requires_manual_review:
                stats['flagged_for_review'] += 1
            
            for flag in result.flags:
                stats['flags_by_type'][flag.value] = stats['flags_by_type'].get(flag.value, 0) + 1
        
        return results, stats

# =====================================================================
# FUZZY MATCHING FOR PRE-QC COMPARISON & ML-ASSISTED FLAGGING
# =====================================================================

class CandidateNameMatcher:
    """Fuzzy matching for candidate names with confidence scoring"""
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return CandidateNameMatcher.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @staticmethod
    def normalized_similarity(s1: str, s2: str) -> Tuple[float, str]:
        """
        Calculate normalized similarity (0.0-1.0) using Levenshtein distance.
        Returns: (confidence, reason)
        """
        if not s1 or not s2:
            return (0.0, "Empty name")
        
        s1_norm = s1.lower().strip()
        s2_norm = s2.lower().strip()
        
        if s1_norm == s2_norm:
            return (1.0, "Exact match")
        
        # Check for partial matches
        if s1_norm in s2_norm or s2_norm in s1_norm:
            return (0.9, "Substring match")
        
        max_len = max(len(s1_norm), len(s2_norm))
        distance = CandidateNameMatcher.levenshtein_distance(s1_norm, s2_norm)
        similarity = 1.0 - (distance / max_len)
        
        if similarity > 0.85:
            reason = "High similarity"
        elif similarity > 0.7:
            reason = "Moderate similarity"
        else:
            reason = "Low similarity"
        
        return (similarity, reason)
    
    @staticmethod
    def extract_parts(name: str) -> Dict[str, str]:
        """Extract (LASTNAME, FIRSTNAME) parts from candidate name"""
        if not name:
            return {}
        
        if ',' in name:
            # Already formatted as LASTNAME, FIRSTNAME
            parts = name.split(',')
            return {
                'lastname': parts[0].strip(),
                'firstname': parts[1].strip() if len(parts) > 1 else '',
            }
        
        # Try to parse unformatted name
        name_parts = name.strip().split()
        if len(name_parts) >= 2:
            return {
                'lastname': name_parts[-1],
                'firstname': ' '.join(name_parts[:-1]),
            }
        
        return {'lastname': name, 'firstname': ''}


@dataclass
class PreQCResult:
    """Result of Pre-QC comparison between DL1 and DL2"""
    race_id: str
    strict_passed: bool
    fuzzy_confidence: float  # Overall confidence 0.0-1.0
    candidate_confidence: float
    party_confidence: float
    fec_id_confidence: float
    discrepancy_count: int
    discrepancies: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    status: str = 'passed'  # passed|failed|review_needed
    summary: str = ''


class PreQCComparisonEngine:
    """
    Pre-QC Auto-check: Strict equality + fuzzy matching between DL1 and DL2
    Generates discrepancy report for QC1 review
    """
    
    FUZZY_CONFIDENCE_THRESHOLD = 0.85  # Confidence above this = pass
    STRICT_COMPARISON_FIELDS = [
        'standardized_candidate_name', 'ballot_party', 'fec_party', 'fec_id',
        'total_votes', 'is_write_in'
    ]
    
    @classmethod
    def compare_records(cls, dl1_record: Dict[str, Any], dl2_record: Dict[str, Any]) -> PreQCResult:
        """
        Compare DL1 and DL2 records with strict and fuzzy matching.
        
        Returns:
            PreQCResult with confidence scores and discrepancies
        """
        race_id = dl1_record.get('race_id', dl2_record.get('race_id', 'unknown'))
        
        # Strict Equality Check
        strict_passed = True
        strict_mismatches = {}
        
        for field in cls.STRICT_COMPARISON_FIELDS:
            dl1_val = dl1_record.get(field)
            dl2_val = dl2_record.get(field)
            
            if dl1_val != dl2_val:
                strict_passed = False
                strict_mismatches[field] = {'dl1': dl1_val, 'dl2': dl2_val}
        
        # Fuzzy Matching Confidence Scores
        candidate_conf, candidate_reason = CandidateNameMatcher.normalized_similarity(
            dl1_record.get('standardized_candidate_name', ''),
            dl2_record.get('standardized_candidate_name', '')
        )
        
        party_conf = cls._compare_parties(
            dl1_record.get('ballot_party'),
            dl2_record.get('ballot_party')
        )
        
        fec_id_conf = cls._compare_fec_ids(
            dl1_record.get('fec_id'),
            dl2_record.get('fec_id')
        )
        
        # Calculate overall fuzzy confidence
        overall_fuzzy = (candidate_conf + party_conf + fec_id_conf) / 3.0
        
        # Determine discrepancies
        discrepancies = {}
        for field, mismatch in strict_mismatches.items():
            if field == 'standardized_candidate_name':
                discrepancies[field] = {
                    **mismatch,
                    'confidence': candidate_conf,
                    'reason': candidate_reason,
                }
            elif field in ('ballot_party', 'fec_party'):
                discrepancies[field] = {
                    **mismatch,
                    'confidence': party_conf,
                    'reason': 'Party code mismatch',
                }
            elif field == 'fec_id':
                discrepancies[field] = {
                    **mismatch,
                    'confidence': fec_id_conf,
                    'reason': 'FEC ID mismatch',
                }
            else:
                discrepancies[field] = mismatch
        
        # Determine status
        if strict_passed:
            status = 'passed'
            summary = f"✓ DL1 and DL2 match exactly"
        elif overall_fuzzy >= cls.FUZZY_CONFIDENCE_THRESHOLD:
            status = 'review_needed'
            summary = f"⚠ Fuzzy match confidence {overall_fuzzy:.1%} - minor discrepancies"
        else:
            status = 'failed'
            summary = f"✗ Confidence {overall_fuzzy:.1%} below threshold - significant differences"
        
        return PreQCResult(
            race_id=race_id,
            strict_passed=strict_passed,
            fuzzy_confidence=overall_fuzzy,
            candidate_confidence=candidate_conf,
            party_confidence=party_conf,
            fec_id_confidence=fec_id_conf,
            discrepancy_count=len(discrepancies),
            discrepancies=discrepancies,
            status=status,
            summary=summary,
        )
    
    @classmethod
    def _compare_parties(cls, party1: Optional[str], party2: Optional[str]) -> float:
        """Compare party codes with fuzzy matching"""
        if not party1 or not party2:
            return 0.5 if (party1 is None) == (party2 is None) else 0.0
        
        if party1 == party2:
            return 1.0
        
        # Check if they map to same FEC code
        fec1 = PartyCodeMapper.map_party(party1)[0]
        fec2 = PartyCodeMapper.map_party(party2)[0]
        
        if fec1 and fec2 and fec1 == fec2:
            return 0.9  # Same FEC party but different ballot representation
        
        return 0.3  # Different parties


    @classmethod
    def _compare_fec_ids(cls, fec_id1: Optional[str], fec_id2: Optional[str]) -> float:
        """Compare FEC IDs"""
        if not fec_id1 and not fec_id2:
            return 1.0  # Both empty = match
        
        if not fec_id1 or not fec_id2:
            return 0.5  # One empty = partial match (one might be enriched)
        
        return 1.0 if fec_id1 == fec_id2 else 0.0  # FEC IDs must match exactly


class QCAutoFlagger:
    """
    ML-Assisted QC Flagging for QC1/QC2
    Detects issues and suggests corrections/actions
    """
    
    @dataclass
    class Flag:
        """Flag with suggested action"""
        flag_type: str
        severity: str  # low|medium|high
        description: str
        suggested_action: str
        field_affected: Optional[str] = None
        dl_source: Optional[str] = None  # DL1|DL2
    
    @classmethod
    def auto_flag_record(cls, record: Dict[str, Any], dl_source: str = 'DL1') -> List['QCAutoFlagger.Flag']:
        """
        Analyze a single record and return list of detected issues.
        
        Args:
            record: Standardization result dict
            dl_source: DL1|DL2 - source of data
            
        Returns:
            List of Flag objects with recommendations
        """
        flags = []
        
        # Check for missing FEC ID
        if not record.get('fec_id') and not record.get('is_write_in'):
            flags.append(cls.Flag(
                flag_type='missing_fec_id',
                severity='high',
                description='FEC ID missing for non-write-in candidate',
                suggested_action='Look up FEC ID in FEC database or flag for manual entry',
                field_affected='fec_id',
                dl_source=dl_source,
            ))
        
        # Check for write-in with FEC ID (might be enrichment error)
        if record.get('is_write_in') and record.get('fec_id'):
            flags.append(cls.Flag(
                flag_type='write_in_with_fec_id',
                severity='medium',
                description='Write-in candidate has FEC ID (may need verification)',
                suggested_action='Verify FEC ID is correct for this write-in',
                field_affected='fec_id',
                dl_source=dl_source,
            ))
        
        # Check for party code issues
        if record.get('ballot_party') and not record.get('fec_party'):
            flags.append(cls.Flag(
                flag_type='unmapped_party',
                severity='medium',
                description=f"Party '{record.get('ballot_party')}' could not be mapped to FEC code",
                suggested_action='Review party code and map to standard FEC party',
                field_affected='ballot_party',
                dl_source=dl_source,
            ))
        
        # Check for candidate name quality
        candidate_name = record.get('standardized_candidate_name', '')
        if '/' in candidate_name:
            flags.append(cls.Flag(
                flag_type='joint_ticket_candidate',
                severity='medium',
                description='Candidate name contains "/" (joint ticket - may need split)',
                suggested_action='Review if this should be split into separate candidates',
                field_affected='standardized_candidate_name',
                dl_source=dl_source,
            ))
        
        # Check for overvote/undervote markers
        ballot_name = record.get('ballot_candidate_name', '')
        if any(marker in ballot_name.upper() for marker in ['OVERVOTE', 'UNDERVOTE', 'SPOILED', 'VOID', 'BLANK']):
            flags.append(cls.Flag(
                flag_type='special_vote_category',
                severity='high',
                description='Special vote category detected (overvote, undervote, blank)',
                suggested_action='Verify this is correctly classified - may need special handling',
                field_affected='ballot_candidate_name',
                dl_source=dl_source,
            ))
        
        # Check for ambiguous vote totals
        uncategorized = record.get('uncategorized', 0) or 0
        total = record.get('total_votes', 0) or 0
        if uncategorized > 0 and total > 0:
            pct = (uncategorized / total) * 100
            if pct > 80:
                flags.append(cls.Flag(
                    flag_type='high_uncategorized_votes',
                    severity='medium',
                    description=f'{pct:.0f}% of votes are uncategorized',
                    suggested_action='Review vote type classification - likely missing vote type data',
                    field_affected='uncategorized_votes',
                    dl_source=dl_source,
                ))
        
        return flags
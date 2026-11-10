"""W24PUBPREVIEW6C3A-REC1B read-only qualified Virginia source adapter.

Not a generic CP1252 policy. Activated only by the server-approved registry
source/URL and a pinned immutable source byte digest in the calling module.
No automatic methods inference, county coercion, summary-to-precinct conversion,
or votes guessed from absent observations. This module is an UNAPPLIED candidate.
"""
from __future__ import annotations
import csv
import hashlib
import io
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

SOURCE_SHA256 = '95e1128a9aa40b72a22ac6be6a5a6b5c1238722b6473070fab5969616c3d43db'
SOURCE_BYTES = 13_269_511
SOURCE_ID_SHA256 = '569cb4cb878d78535d833bda69a24fb484fa0abc2e0f6a17a0a97a3c154a587e'
SOURCE_URL_SHA256 = '6c7aa1279ae1b19f66d46571e7d9b15da4177b0465770c629d2aa445c34582cc'
SOURCE_DATE = '2024-11-05'
SOURCE_OFFICE = 'Member, United States Senate'
SOURCE_HEADERS = ('CandidateId','CandidateName','TOTAL_VOTES','Party','WriteInVote','LocalityCode','LocalityName','PrecinctId','PrecinctName','DistrictId','DistrictType','DistrictName','OfficeId','OfficeTitle','ElectionId','ElectionType','ElectionDate','ElectionName','NumberOfSeats')
METHODS = ('Election Day','Early Voting','Absentee Mail','Provisional')
MAX_ROWS = 150_000
MAX_CELL = 2_000
MAX_CANDIDATES = 200
MAX_VOTE = 2**63-1

class VirginiaSourceError(ValueError):
    """Dedicated adapter failure; do not convert into silently partial results."""

@dataclass(frozen=True)
class VirginiaShape:
    headers: list[str]
    rows: list[dict[str, Any]]
    precinct_rows: int
    candidate_count: int
    source_sha256: str
    missing_method_cells: int
    discrepancy_precincts: int
    year_column_present: bool = True
    state_column_present: bool = False
    excluded_other_contest_rows: int = 0
    excluded_summary_rows: int = 0
    special_voting_unit_rows: int = 0
    incomplete_candidate_observations: int = 0
    office_observations: int = 0
    quarantined_zero_idless_observations: int = 0
    quarantined_zero_idless_groups: int = 0
    quarantined_duplicate_candidate_observations: int = 0
    quarantined_vote_total: int = 0
    quarantine_identity_hashes: tuple[str, ...] = ()
    quarantine_observation_hashes: tuple[str, ...] = ()
    null_method_policy: str = 'UNREPORTED_METHODS_NULL_NOT_ZERO'


def norm(s: str) -> str:
    return ' '.join(re.findall(r'[a-z0-9]+', s.casefold()))


def safe_cell(s: str, *, max_chars: int = 160) -> str:
    value=s.strip()
    if not value or len(value)>max_chars or value[0] in '=+-@' or any(ord(c)<32 or 127<=ord(c)<=159 for c in value):
        raise VirginiaSourceError('VA_UNSAFE_OR_MISSING_SOURCE_IDENTITY')
    return value


def parse_total(s: str) -> int:
    v=s.strip().replace(',','')
    if not re.fullmatch(r'\d+',v):raise VirginiaSourceError('VA_INVALID_VOTE_TOTAL')
    n=int(v)
    if n>MAX_VOTE:raise VirginiaSourceError('VA_VOTE_OVERFLOW')
    return n


def parse_virginia_2024_senate(data: bytes, *, source_url: str, source_sha256: str | None = None,
                               expected_source_sha256: str = SOURCE_SHA256,
                               expected_source_bytes: int = SOURCE_BYTES) -> VirginiaShape:
    """Read-only, deterministic source-specific wide precinct extraction.

    The source pins may be overridden only by the OFFLINE synthetic unit test
    calling this pure function. The runtime integration always passes defaults.
    """
    if not isinstance(data,bytes) or len(data)!=expected_source_bytes:
        raise VirginiaSourceError('VA_SOURCE_LENGTH_DRIFT')
    digest=hashlib.sha256(data).hexdigest()
    if digest!=expected_source_sha256 or (source_sha256 is not None and digest!=source_sha256):
        raise VirginiaSourceError('VA_SOURCE_DIGEST_DRIFT')
    if not source_url.startswith('https://') or hashlib.sha256(source_url.encode('utf-8')).hexdigest()!=SOURCE_URL_SHA256 and expected_source_sha256==SOURCE_SHA256:
        raise VirginiaSourceError('VA_SOURCE_URL_IDENTITY_DRIFT')
    try:decoded=data.decode('cp1252',errors='strict')
    except UnicodeDecodeError as exc:raise VirginiaSourceError('VA_PINNED_CP1252_DECODING_FAILED') from exc
    if any((ord(c)<32 and c not in '\r\n\t') or 127<=ord(c)<=159 for c in decoded):
        raise VirginiaSourceError('VA_SOURCE_CONTROL_CHARACTER')
    if '\ufffd' in decoded or '\x00' in decoded:raise VirginiaSourceError('VA_SOURCE_REPLACEMENT_OR_BINARY')
    reader=csv.DictReader(io.StringIO(decoded,newline=''),restkey='__extra__',restval=None)
    if tuple(reader.fieldnames or [])!=SOURCE_HEADERS:
        raise VirginiaSourceError('VA_19_COLUMN_HEADER_DRIFT')
    # Ordered by source appearance; every contest/precinct identity remains separate.
    precincts: OrderedDict[tuple[str,str], dict[str,Any]]=OrderedDict()
    candidates: dict[str,tuple[str,str]]= {} # candidate ID -> candidate name + party
    candidate_label_ids: dict[str,str]={}
    locality_by_code: dict[str,str]={}
    other=summary=senate=0
    zero_idless=0
    idless_dup=0
    quarantined_record_hashes: list[str] = []
    quarantine: dict[tuple[str,str],dict[str,Any]] = {}
    senate_office_ids: set[str]=set()
    senate_election_ids: set[str]=set()
    for count,rec in enumerate(reader,1):
        if count>MAX_ROWS:raise VirginiaSourceError('VA_RECORD_BUDGET_EXCEEDED')
        if not isinstance(rec,dict) or '__extra__' in rec or any(v is None or len(v)>MAX_CELL for v in rec.values()):
            raise VirginiaSourceError('VA_RAGGED_OR_OVERSIZED_RECORD')
        if rec['ElectionDate'].strip()!=SOURCE_DATE or rec['ElectionType'].strip()!='General':
            raise VirginiaSourceError('VA_ELECTION_DATE_OR_TYPE_SCOPE_DRIFT')
        office=rec['OfficeTitle'].strip()
        if office!=SOURCE_OFFICE:
            other+=1;continue
        senate+=1
        senate_office_ids.add(safe_cell(rec['OfficeId']))
        senate_election_ids.add(safe_cell(rec['ElectionId']))
        if len(senate_office_ids)>1 or len(senate_election_ids)>1:
            raise VirginiaSourceError('VA_MULTIPLE_SENATE_CONTEST_IDENTITIES')
        if rec['DistrictType'].strip().casefold()!='state':
            raise VirginiaSourceError('VA_SENATE_DISTRICT_SCOPE_DRIFT')
        locid=safe_cell(rec['LocalityCode']);loc=safe_cell(rec['LocalityName'])
        if locid in locality_by_code and locality_by_code[locid]!=loc:
            raise VirginiaSourceError('VA_LOCALITY_CODE_COLLISION')
        locality_by_code[locid]=loc
        precinct_id=rec['PrecinctId'].strip();precinct_name=rec['PrecinctName'].strip()
        candidateid=safe_cell(rec['CandidateId']);candidate=safe_cell(rec['CandidateName'])
        party=rec['Party'].strip()
        if party and (len(party)>40 or party[0] in '=+-@' or any(ord(c)<32 or 127<=ord(c)<=159 for c in party)):
            raise VirginiaSourceError('VA_UNSAFE_PARTY_VALUE')
        existing=candidates.setdefault(candidateid,(candidate,party))
        if existing!=(candidate,party):raise VirginiaSourceError('VA_CANDIDATE_IDENTITY_COLLISION')
        label=candidate + (f' ({party})' if party else '')
        if label in candidate_label_ids and candidate_label_ids[label]!=candidateid:
            raise VirginiaSourceError('VA_CANDIDATE_OUTPUT_LABEL_COLLISION')
        candidate_label_ids[label]=candidateid
        if len(candidates)>MAX_CANDIDATES:raise VirginiaSourceError('VA_CANDIDATE_LIMIT_EXCEEDED')
        if not precinct_id:
            # Preserve *every* official zero-vote ID-less observation in an
            # explicitly disclosed exception ledger, never as a fake precinct.
            # A single nonzero record or a changed group structure is a HOLD.
            precinct_name=safe_cell(precinct_name)
            if re.search(r'\b(total|totals|summary|statewide|countywide|localitywide|all precincts|all precinct|aggregate|absentee|provisional|curbside|early voting|early vote|mail|vote center|voting center)\b',precinct_name,re.I):
                raise VirginiaSourceError('VA_IDLESS_LABEL_CLASSIFICATION_CHANGED')
            v=parse_total(rec['TOTAL_VOTES'])
            if v!=0:raise VirginiaSourceError('VA_NONZERO_IDLESS_OBSERVATION_REQUIRES_ADJUDICATION')
            key=(locid,' '.join(precinct_name.casefold().split()))
            group=quarantine.setdefault(key,{'candidate_ids':set(),'candidate_counts':{},'count':0,'duplicates':0,'votes':0,'name':precinct_name})
            if group['name']!=precinct_name:raise VirginiaSourceError('VA_IDLESS_NAME_IDENTITY_COLLISION')
            if candidateid in group['candidate_ids']:
                group['duplicates']+=1
                idless_dup+=1
            group['candidate_ids'].add(candidateid)
            group['candidate_counts'][candidateid]=group['candidate_counts'].get(candidateid,0)+1
            group['count']+=1
            group['votes']+=v
            # One source-row-position fingerprint per original zero observation.
            # Keeps all 48 records attributable, including duplicates, without
            # persisting identifying source row text or individual vote records.
            quarantined_record_hashes.append(hashlib.sha256(('VA_IDLESS_SOURCE_OBSERVATION_V1\0'+str(count)+'\0'+locid+'\0'+key[1]+'\0'+candidateid).encode('utf-8')).hexdigest()[:20])
            zero_idless+=1
            continue
        precinct_id=safe_cell(precinct_id);precinct_name=safe_cell(precinct_name)
        k=(locid,precinct_id)
        special=bool(re.search(r'\b(absentee|provisional|central absentee|curbside|early voting)\b',precinct_name,re.I))
        obj=precincts.setdefault(k,{'locality':loc,'locality_code':locid,'precinct_name':precinct_name,'precinct_id':precinct_id,'special':special,'votes':{},'writein_metadata':set()})
        if (obj['locality'],obj['precinct_name'],obj['special'])!=(loc,precinct_name,special):
            raise VirginiaSourceError('VA_PRECINCT_IDENTITY_COLLISION')
        if candidateid in obj['votes']:raise VirginiaSourceError('VA_DUPLICATE_PRECINCT_CANDIDATE')
        obj['votes'][candidateid]=parse_total(rec['TOTAL_VOTES'])
        # Keep WriteInVote as metadata only, never an inferred voting method.
        writein=rec['WriteInVote'].strip()
        if writein and (len(writein)>32 or any(ord(c)<32 for c in writein)):
            raise VirginiaSourceError('VA_INVALID_WRITEIN_METADATA')
        obj['writein_metadata'].add(writein)
    if not precincts or not senate:raise VirginiaSourceError('VA_NO_VERIFIED_SENATE_PRECINCTS')
    if len(quarantine)!=8 or zero_idless!=48 or idless_dup!=24:
        raise VirginiaSourceError('VA_IDLESS_QUARANTINE_COUNT_OR_DUPLICATE_DRIFT')
    if any(g['count']!=6 or g['duplicates']!=3 or len(g['candidate_ids'])!=len(candidates) or g['votes']!=0 or any(n!=2 for n in g['candidate_counts'].values()) for g in quarantine.values()):
        raise VirginiaSourceError('VA_IDLESS_QUARANTINE_CANDIDATE_PATTERN_DRIFT')
    # Reproduce the REC1A audit's group fingerprints, without persisting names.
    qhashes=tuple(sorted(hashlib.sha256(('VIRGINIA_IDLESS_AUDIT_V1\0' + locid + '\0' + normalized).encode('utf-8')).hexdigest()[:20] for locid,normalized in quarantine))
    if len(quarantined_record_hashes)!=48 or len(set(quarantined_record_hashes))!=48:
        raise VirginiaSourceError('VA_IDLESS_QUARANTINE_RECORD_FINGERPRINT_DRIFT')
    if len(qhashes)!=8 or len(set(qhashes))!=8:
        raise VirginiaSourceError('VA_IDLESS_QUARANTINE_FINGERPRINT_COLLISION')
    ordering=sorted(candidates, key=lambda cid:(candidates[cid][1],candidates[cid][0],cid))
    prefix=['Precinct','% Precincts Reporting','Election Day Total','Early Voting Total','Absentee Mail Total','Provisional Total',
            'Locality','Locality Code','Precinct ID','Source Precinct Name','Voting Unit Type','Official Source Link','Source SHA-256']
    headers=prefix+[f'{candidates[cid][0]}{(" ("+candidates[cid][1]+")") if candidates[cid][1] else ""} - {m}'
                    for cid in ordering for m in METHODS+('Total Votes',)]+['Method Coverage Flag','Discrepancy Flag','Grand Total']
    rows=[];method_nulls=0;incomplete=0;discrepancies=0;special_count=0
    for obj in precincts.values():
        row=dict.fromkeys(headers)
        row.update({'Precinct':f'{obj["locality"]} / {obj["precinct_name"]} [ID:{obj["precinct_id"]}]',
                    'Locality':obj['locality'],'Locality Code':obj['locality_code'],'Precinct ID':obj['precinct_id'],
                    'Source Precinct Name':obj['precinct_name'],
                    'Voting Unit Type':'SPECIAL_VOTING_UNIT' if obj['special'] else 'PRECINCT_IDENTITY',
                    'Official Source Link':source_url,'Source SHA-256':digest,
                    'Method Coverage Flag':'SOURCE_TOTALS_ONLY_METHODS_UNREPORTED'})
        if obj['special']:special_count+=1
        missing=[];vsum=0
        for cid in ordering:
            cand,party=candidates[cid];label=cand+(f' ({party})' if party else '')
            # All methods are NULL because the source contains only totals.
            for method in METHODS:row[f'{label} - {method}']=None;method_nulls+=1
            value=obj['votes'].get(cid)
            row[f'{label} - Total Votes']=value
            if value is None:missing.append(label)
            else:vsum+=value
        if missing:
            incomplete+=len(missing)
            row['Grand Total']=None
            row['Discrepancy Flag']='INCOMPLETE_CANDIDATE_OBSERVATIONS'
            discrepancies+=1
        else:
            if vsum>MAX_VOTE:raise VirginiaSourceError('VA_PRECINCT_TOTAL_OVERFLOW')
            row['Grand Total']=vsum
            row['Discrepancy Flag']=''
        rows.append(row)
    grand=dict.fromkeys(headers)
    grand['Precinct']='Grand Total (summary, not a precinct)'
    grand['Official Source Link']=source_url
    grand['Source SHA-256']=digest
    for h in headers:
        if h in ('Grand Total',) or h.endswith(' - Total Votes'):
            vals=[r[h] for r in rows]
            if vals and all(type(v) is int for v in vals):grand[h]=sum(vals)
    grand['Method Coverage Flag']='SOURCE_TOTALS_ONLY_METHODS_UNREPORTED'
    grand['Discrepancy Flag']='SUMMARY_NOT_A_PRECINCT'
    rows.append(grand)
    return VirginiaShape(headers,rows,len(rows)-1,len(candidates),digest,method_nulls,discrepancies,
                         excluded_other_contest_rows=other,excluded_summary_rows=summary,
                         special_voting_unit_rows=special_count,incomplete_candidate_observations=incomplete,
                         office_observations=senate,
                         quarantined_zero_idless_observations=zero_idless,
                         quarantined_zero_idless_groups=len(quarantine),
                         quarantined_duplicate_candidate_observations=idless_dup,
                         quarantined_vote_total=0,
                         quarantine_identity_hashes=qhashes,
                         quarantine_observation_hashes=tuple(quarantined_record_hashes))

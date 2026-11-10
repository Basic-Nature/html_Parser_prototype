"""P6C4B review candidate: compact ordered-column contest pages; NOT ROUTED.

A result is a server-session-bound in-memory snapshot. Headers are returned
ONCE in a bounded manifest. Page envelopes contain ordered value arrays only.
Safe retry: the *current* page may be requested again until the next page is
requested; old page replay is rejected. End-of-stream retained until explicitly
acknowledged or TTL expiry (lost final-page response can be retried).

NOT a public endpoint. Server-side authorization, request throttles, origin/CSRF
checks, worker pinning, cancellation, client UI, and process RSS must be
qualified separately. No Redis or filesystem persistence.
"""
from __future__ import annotations
import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from typing import Any

CONTRACT = 'electionpulse_contest_compact_v1'
EVENT_HARD_LIMIT = 65536
PAGE_BUDGET = 49152
TTL_SECONDS = 600
SNAPSHOT_BYTE_LIMIT = 16 * 1024 * 1024
MAX_LIVE_SNAPSHOTS = 4
MAX_ROWS = 30000
MAX_PAGES = 4096

class CompactDeliveryError(ValueError):
    pass


def packed(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')


def _b64(x:bytes)->str:
    return base64.urlsafe_b64encode(x).decode('ascii').rstrip('=')


def _unb64(x:str)->bytes:
    return base64.urlsafe_b64decode(x+'='*(-len(x)%4))


@dataclass
class Snapshot:
    session_hash: str
    expires_at: int
    scope: dict[str, str]
    digest: str
    headers: list[str]
    rows: list[list[Any]]
    boundaries: list[tuple[int, int]]
    highest_page_seen: int = -1
    request_count: int = 0


class CompactContestPager:
    def __init__(self, secret: bytes, *, page_budget: int = PAGE_BUDGET,
                 frame_limit: int = EVENT_HARD_LIMIT, ttl_seconds: int = TTL_SECONDS,
                 snapshot_limit: int = SNAPSHOT_BYTE_LIMIT, max_live_snapshots: int = MAX_LIVE_SNAPSHOTS):
        if not isinstance(secret,bytes) or len(secret)<32:
            raise CompactDeliveryError('PAGER_SECRET_REQUIRED')
        if not (8192 <= page_budget <= frame_limit-2048 and frame_limit <= EVENT_HARD_LIMIT):
            raise CompactDeliveryError('PAGER_FRAME_BUDGET_UNSAFE')
        if not (30 <= ttl_seconds <= 900 and 1024*1024 <= snapshot_limit <= 32*1024*1024):
            raise CompactDeliveryError('PAGER_SNAPSHOT_LIMIT_UNSAFE')
        if not (1 <= max_live_snapshots <= 8):
            raise CompactDeliveryError('PAGER_CAPACITY_UNSAFE')
        self.secret=secret
        self.page_budget=page_budget
        self.frame_limit=frame_limit
        self.ttl_seconds=ttl_seconds
        self.snapshot_limit=snapshot_limit
        self.max_live_snapshots=max_live_snapshots
        self.snapshots:dict[str,Snapshot]={}

    def _sid(self,session_id:str)->str:
        if not isinstance(session_id,str) or not session_id or len(session_id)>512:
            raise CompactDeliveryError('PAGER_SERVER_SESSION_REQUIRED')
        return hmac.new(self.secret,b'sid\0'+session_id.encode(),hashlib.sha256).hexdigest()

    def _token(self,snapshot_id:str,page:int,expires_at:int,session_hash:str)->str:
        message=packed({'i':snapshot_id,'p':page,'x':expires_at,'s':session_hash,'v':1})
        return _b64(message)+'.'+_b64(hmac.new(self.secret,message,hashlib.sha256).digest())

    def _read_token(self,token:str,session_hash:str,now:int)->dict[str,Any]:
        try:
            if not isinstance(token,str) or len(token)>1200 or token.count('.')!=1:
                raise ValueError()
            a,b=token.split('.')
            message=_unb64(a)
            if not hmac.compare_digest(hmac.new(self.secret,message,hashlib.sha256).digest(),_unb64(b)):
                raise ValueError()
            item=json.loads(message)
            if (not isinstance(item,dict) or set(item)!={'i','p','x','s','v'}
                or item['s']!=session_hash or type(item['p']) is not int or type(item['x']) is not int
                or item['v']!=1 or now>=item['x'] or len(item['i'])>64):
                raise ValueError()
            return item
        except Exception as exc:
            raise CompactDeliveryError('PAGER_INVALID_EXPIRED_OR_FOREIGN_CURSOR') from exc

    def purge(self, *, now:int|None=None)->int:
        now=int(time.time()) if now is None else now
        stale=[key for key,val in self.snapshots.items() if val.expires_at<=now]
        for key in stale:del self.snapshots[key]
        return len(stale)

    @staticmethod
    def _scope(scope:dict[str,str])->dict[str,str]:
        if not isinstance(scope,dict) or set(scope)!={'state','year','contest','registry_source_id'}:
            raise CompactDeliveryError('PAGER_SERVER_SCOPE_REQUIRED')
        if any(not isinstance(v,str) or not v or len(v)>200 for v in scope.values()):
            raise CompactDeliveryError('PAGER_SCOPE_INVALID')
        return dict(scope)

    def _page(self, *, ident:str, index:int, start:int, end:int, total:int,
              rows:list[list[Any]], expiry:int, sid_hash:str)->dict[str,Any]:
        next_token=self._token(ident,index+1,expiry,sid_hash) if index+1<total else None
        return {'v':1,'i':ident,'p':index,'n':total,'d':rows[start:end],'next':next_token}

    def register(self, *, server_session_id:str, scope:dict[str,str], source_sha256:str,
                 headers:list[str], rows:list[dict[str,Any]], now:int|None=None)->tuple[dict[str,Any],str]:
        now=int(time.time()) if now is None else now
        self.purge(now=now)
        if len(self.snapshots)>=self.max_live_snapshots:
            raise CompactDeliveryError('PAGER_STORE_CAPACITY_REACHED')
        sid_hash=self._sid(server_session_id)
        scope=self._scope(scope)
        if not isinstance(source_sha256,str) or len(source_sha256)!=64 or any(ch not in '0123456789abcdef' for ch in source_sha256):
            raise CompactDeliveryError('PAGER_SOURCE_DIGEST_INVALID')
        if (not isinstance(headers,list) or not headers or len(headers)>500 or len(set(headers))!=len(headers)
            or any(not isinstance(h,str) or not h or len(h)>200 for h in headers)):
            raise CompactDeliveryError('PAGER_HEADERS_INVALID')
        if not isinstance(rows,list) or not 1<=len(rows)<=MAX_ROWS:
            raise CompactDeliveryError('PAGER_ROW_COUNT_INVALID')
        names=set(headers)
        if any(not isinstance(row,dict) or set(row)!=names for row in rows):
            raise CompactDeliveryError('PAGER_ROW_SCHEMA_INVALID')
        # Snapshot freeze eliminates post-registration object mutation and prevents
        # future server code from accidentally changing signed page boundaries.
        frozen=json.loads(packed({'headers':headers,'scope':scope,'rows':rows}))
        columns=frozen['headers'];scope=frozen['scope']
        matrix=[[row[h] for h in columns] for row in frozen['rows']]
        if len(packed({'h':columns,'r':matrix,'s':scope,'d':source_sha256}))>self.snapshot_limit:
            raise CompactDeliveryError('PAGER_SERIALIZED_SNAPSHOT_TOO_LARGE')
        ident=secrets.token_urlsafe(16)
        expiry=now+self.ttl_seconds
        # Build boundaries with conservative worst-case total page count and
        # exact envelope serialization; never exceed the socket event ceiling.
        boundaries=[];start=0
        while start<len(matrix):
            end=start+1
            while True:
                frame=self._page(ident=ident,index=len(boundaries),start=start,end=end,
                                 total=MAX_PAGES,rows=matrix,expiry=expiry,sid_hash=sid_hash)
                if len(packed(frame))>self.page_budget:
                    if end==start+1:raise CompactDeliveryError('PAGER_SINGLE_ROW_TOO_LARGE')
                    end-=1
                    break
                if end==len(matrix):break
                end+=1
            boundaries.append((start,end))
            if len(boundaries)>MAX_PAGES:raise CompactDeliveryError('PAGER_TOO_MANY_PAGES')
            start=end
        manifest={'contract':CONTRACT,'v':1,'scope':scope,'source_sha256':source_sha256,
                  'headers':columns,'rows':len(matrix),'page_count':len(boundaries),
                  'expires_at':expiry,'snapshot_id':ident,'page_budget_bytes':self.page_budget}
        if len(packed(manifest))>self.page_budget:
            raise CompactDeliveryError('PAGER_MANIFEST_TOO_LARGE')
        first=self._token(ident,0,expiry,sid_hash)
        self.snapshots[ident]=Snapshot(sid_hash,expiry,scope,source_sha256,columns,matrix,boundaries)
        return manifest,first

    def fetch(self, token:str, *, server_session_id:str, now:int|None=None)->dict[str,Any]:
        now=int(time.time()) if now is None else now
        sid_hash=self._sid(server_session_id)
        item=self._read_token(token,sid_hash,now)
        snap=self.snapshots.get(item['i'])
        if snap is None or snap.session_hash!=sid_hash or snap.expires_at!=item['x']:
            raise CompactDeliveryError('PAGER_SNAPSHOT_NOT_FOUND_OR_FOREIGN')
        index=item['p']
        if not 0<=index<len(snap.boundaries) or index<snap.highest_page_seen or index>snap.highest_page_seen+1:
            raise CompactDeliveryError('PAGER_STALE_OR_OUT_OF_ORDER_CURSOR')
        snap.request_count+=1
        if snap.request_count>min(10000,4*len(snap.boundaries)+8):
            raise CompactDeliveryError('PAGER_RETRY_BUDGET_EXCEEDED')
        snap.highest_page_seen=max(snap.highest_page_seen,index)
        start,end=snap.boundaries[index]
        frame=self._page(ident=item['i'],index=index,start=start,end=end,total=len(snap.boundaries),
                         rows=snap.rows,expiry=snap.expires_at,sid_hash=sid_hash)
        length=len(packed(frame))
        if length>self.page_budget or length>self.frame_limit:
            raise CompactDeliveryError('PAGER_EVENT_SIZE_EXCEEDED')
        return frame

    def acknowledge_complete(self,last_page_token:str, *, server_session_id:str, now:int|None=None)->bool:
        now=int(time.time()) if now is None else now
        item=self._read_token(last_page_token,self._sid(server_session_id),now)
        snap=self.snapshots.get(item['i'])
        if snap is None or snap.session_hash!=item['s']:
            raise CompactDeliveryError('PAGER_SNAPSHOT_NOT_FOUND_OR_FOREIGN')
        if item['p']!=len(snap.boundaries)-1 or snap.highest_page_seen!=item['p']:
            raise CompactDeliveryError('PAGER_ACK_BEFORE_LAST_PAGE')
        del self.snapshots[item['i']]
        return True

    def cancel(self, snapshot_id:str, *, server_session_id:str)->bool:
        snap=self.snapshots.get(snapshot_id)
        if snap is None:return False
        if snap.session_hash!=self._sid(server_session_id):
            raise CompactDeliveryError('PAGER_CANCEL_FOREIGN_SESSION')
        del self.snapshots[snapshot_id]
        return True


def reconstruct_rows(headers:list[str],frames:list[dict[str,Any]])->list[dict[str,Any]]:
    if not frames or not headers or len(set(headers))!=len(headers):
        raise CompactDeliveryError('PAGER_RECONSTRUCTION_INPUT_INVALID')
    expected=len(frames)
    if any(frame.get('n')!=expected or frame.get('p')!=i or frame.get('v')!=1 for i,frame in enumerate(frames)):
        raise CompactDeliveryError('PAGER_MISSING_DUPLICATE_OR_REORDERED_PAGE')
    rows=[]
    for frame in frames:
        for values in frame['d']:
            if not isinstance(values,list) or len(values)!=len(headers):
                raise CompactDeliveryError('PAGER_ROW_WIDTH_MISMATCH')
            rows.append(dict(zip(headers,values)))
    return rows

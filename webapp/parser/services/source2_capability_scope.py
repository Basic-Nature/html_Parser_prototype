"""Pure capability decision candidate. Not wired into the public runtime.

Call only with server-resolved source IDs and the approved execution set from
public_ballot_lens_policy. Re-evaluate at each capability issuance boundary.
"""
from dataclasses import dataclass
import re
from collections.abc import Mapping

MD = 'blsrc_v1_fd0d8be9659f0c22e4c03cca55b52742aa809ae8650729e63bc186235cb64837'
UTAH = 'blsrc_v1_267d575a499b910deacc0c4ee660e696aeeab9c162126fe17625239be15f615c'
ID = re.compile(r'\Ablsrc_v1_[0-9a-f]{64}\Z')
MD_COMPACT = 'BALLOT_LENS_PUBLIC_MARYLAND_COMPACT_PAGES_ENABLED'
MD_CSV = 'BALLOT_LENS_PUBLIC_MARYLAND_CSV_DOWNLOAD_ENABLED'
LEGACY_COMPACT = 'ELECTIONPULSE_P6C5_COMPACT_PAGES_ENABLED'
LEGACY_CSV = 'BALLOT_LENS_PUBLIC_CSV_DOWNLOAD_ENABLED'

@dataclass(frozen=True)
class SourceCapabilities:
    compact: bool = False
    csv: bool = False


def enabled(env: Mapping[str, str], key: str) -> bool:
    value = env.get(key)
    return isinstance(value, str) and value.strip().lower() in {'1','true','yes','on'}


def source_capabilities(*, source_id: str, primary_id: str | None,
                        approved_execution_ids: tuple[str, ...],
                        environ: Mapping[str, str]) -> SourceCapabilities:
    deny = SourceCapabilities()
    if not isinstance(source_id, str) or not ID.fullmatch(source_id):
        return deny
    if not isinstance(primary_id, str) or not ID.fullmatch(primary_id):
        return deny
    if (not isinstance(approved_execution_ids, tuple) or not 1 <= len(approved_execution_ids) <= 2
        or any(not isinstance(v,str) or not ID.fullmatch(v) for v in approved_execution_ids)
        or len(set(approved_execution_ids)) != len(approved_execution_ids)
        or source_id not in approved_execution_ids or primary_id not in approved_execution_ids):
        return deny
    if source_id == primary_id:
        # Preserve independent legacy flags for the historical primary source.
        return SourceCapabilities(enabled(environ, LEGACY_COMPACT), enabled(environ, LEGACY_CSV))
    if primary_id != UTAH or source_id != MD:
        return deny
    compact = enabled(environ, MD_COMPACT)
    return SourceCapabilities(compact, compact and enabled(environ, MD_CSV))

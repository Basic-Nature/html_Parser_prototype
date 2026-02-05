"""
Helper functions for dataset promotion with verification gating.
"""

from webapp.parser.utils.logger_singleton import logger


def check_exact_duplicate(session, state: str, county: str, contest: str,
                          candidate: str, party: str, votes: int,
                          precinct: str = None, election_date = None) -> bool:
    """
    Check for exact match (all fields must match). Returns True if duplicate exists.
    Excludes 'unverified' or 'rejected' records to allow re-processing.
    """
    from webapp.parser.utils.models import WarehouseElectionResult
    
    query = session.query(WarehouseElectionResult).filter(
        WarehouseElectionResult.state == state,
        WarehouseElectionResult.county == county,
        WarehouseElectionResult.contest == contest,
        WarehouseElectionResult.candidate == candidate,
        WarehouseElectionResult.party == party,
        WarehouseElectionResult.votes == votes,
        WarehouseElectionResult.verification_status.in_(['verified', 'pending']),
    )
    if precinct:
        query = query.filter(WarehouseElectionResult.precinct == precinct)
    if election_date:
        query = query.filter(WarehouseElectionResult.election_date == election_date)
    return query.first() is not None


def get_url_verification_tier(source_url: str) -> str:
    """
    Return 'trusted' | 'pending' | 'blocked' based on trust score.
    
    - score >= 85: 'trusted' (verified source, auto-import as verified)
    - score 60-84: 'pending' (questionable, requires manual review)
    - score < 60: 'blocked' (reject entirely)
    """
    if not source_url:
        return 'pending'
    
    try:
        from webapp.parser.utils.url_trust_scorer import compute_trust_score
        score, factors = compute_trust_score(source_url, {})
        if score >= 85:
            return 'trusted'
        elif score >= 60:
            return 'pending'
        else:
            return 'blocked'
    except Exception as exc:
        logger.warning(f"[URL_TIER] Failed to compute trust score: {exc}")
        return 'pending'

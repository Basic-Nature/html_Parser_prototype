"""
4-Tier Privilege System for Election Results Parser

Tiers:
0. STANDARD_USER: Regular election officials (85/70/35 thresholds)
1. ADMIN_REVIEWER: Quarantine reviewers (75/55/25 thresholds, +5 boost)
2. ADMIN_FULL_TRUST: Full administrators (70/50/20 thresholds, +10 boost)
3. ROOT_ADMIN: Root admin via Linux UID=0 (tokenized session, no thresholds)

Admin boost applied only to gov/verified/allowlist domains (security boundary).
Root admin always audited, never bypasses logging.
"""

from __future__ import annotations

import os
from enum import IntEnum
from webapp.parser.auth.authorization import tier_satisfies
from typing import Dict, Optional

from ..utils.logger_singleton import logger


class PrivilegeTier(IntEnum):
    """4-tier privilege model."""
    STANDARD_USER = 0
    ADMIN_REVIEWER = 1
    ADMIN_FULL_TRUST = 2
    ROOT_ADMIN = 3
    
    @property
    def name_display(self) -> str:
        """Human-readable tier name."""
        names = {
            0: "Standard User",
            1: "Admin Reviewer",
            2: "Admin Full Trust",
            3: "Root Admin"
        }
        return names.get(int(self), "Unknown")
    
    @property
    def boost_amount(self) -> int:
        """Trust score boost for admin tiers."""
        boosts = {0: 0, 1: 5, 2: 10, 3: 10}
        return boosts.get(int(self), 0)


def get_tier_trust_thresholds(tier: PrivilegeTier | int) -> Dict[str, int]:
    """
    Get per-tier trust score thresholds.
    
    Returns:
    {
        "snapshot": int,      # >threshold: use DOM snapshot mode (vs direct navigation)
        "quarantine": int,    # >threshold: quarantine for review
        "reject": int         # <threshold: reject outright
    }
    """
    tier = int(tier)
    
    thresholds = {
        0: {"snapshot": 90, "quarantine": 70, "reject": 30},
        1: {"snapshot": 75, "quarantine": 55, "reject": 25},
        2: {"snapshot": 70, "quarantine": 50, "reject": 20},
        3: {"snapshot": 0, "quarantine": 0, "reject": 0},  # Root admin: no thresholds (tokenized session)
    }
    
    return thresholds.get(tier, thresholds[0])


def get_principal_tier(principal: Optional[str], principal_source: str = "") -> PrivilegeTier:
    """
    Resolve principal string to privilege tier.
    
    Args:
        principal: Principal identifier (e.g., "sso:OID" or "cert:CN")
        principal_source: Where principal came from ("sso_oid", "cert", "dev_bypass", etc.)
    
    Returns:
        PrivilegeTier (0=standard, 1=reviewer, 2=full, 3=root)
    """
    if not principal:
        return PrivilegeTier.STANDARD_USER
    
    # ===== SSO OID PRINCIPALS =====
    if principal.startswith("sso:"):
        oid = principal[4:].strip()
        
        # Root admin OIDs (highest privilege)
        root_oids = _parse_env_list("ROOT_ADMIN_PRINCIPALS")
        if oid in root_oids:
            logger.info({
                "level": "INFO",
                "type": "auth",
                "message": f"Principal {principal} mapped to ROOT_ADMIN tier",
                "principal": principal,
                "session_id": None
            })
            return PrivilegeTier.ROOT_ADMIN
        
        # Full trust OIDs
        full_trust_oids = _parse_env_list("ADMIN_FULL_TRUST_PRINCIPALS")
        if oid in full_trust_oids:
            logger.info({
                "level": "INFO",
                "type": "auth",
                "message": f"Principal {principal} mapped to ADMIN_FULL_TRUST tier",
                "principal": principal,
                "session_id": None
            })
            return PrivilegeTier.ADMIN_FULL_TRUST
        
        # Reviewer OIDs
        reviewer_oids = _parse_env_list("ADMIN_REVIEWER_PRINCIPALS")
        if oid in reviewer_oids:
            logger.info({
                "level": "INFO",
                "type": "auth",
                "message": f"Principal {principal} mapped to ADMIN_REVIEWER tier",
                "principal": principal,
                "session_id": None
            })
            return PrivilegeTier.ADMIN_REVIEWER
    
    # ===== CLIENT CERTIFICATE PRINCIPALS =====
    # ===== CLIENT CERTIFICATE PRINCIPALS =====
    if principal.startswith("cert:"):
        cert_identity = principal[5:].strip()

        from webapp.parser.auth.cert_trust import (
            normalize_sha256_fingerprint,
        )

        fingerprint = normalize_sha256_fingerprint(cert_identity)

        if fingerprint:
            def _matches_fingerprint(
                current_fingerprint: str,
                *env_names: str,
            ) -> bool:
                for env_name in env_names:
                    for value in _parse_env_list(env_name):
                        candidate = normalize_sha256_fingerprint(value)
                        if candidate == current_fingerprint:
                            return True
                return False

            if _matches_fingerprint(
                fingerprint,
                "ROOT_ADMIN_CERT_FINGERPRINTS",
                "ROOT_ADMIN_CERTS",
            ):
                logger.info({
                    "level": "INFO",
                    "type": "auth",
                    "message": (
                        f"Principal {principal} mapped to ROOT_ADMIN tier "
                        "via certificate fingerprint"
                    ),
                    "principal": principal,
                    "session_id": None,
                })
                return PrivilegeTier.ROOT_ADMIN

            if _matches_fingerprint(
                fingerprint,
                "ADMIN_FULL_TRUST_CERT_FINGERPRINTS",
                "ADMIN_FULL_TRUST_CERTS",
            ):
                logger.info({
                    "level": "INFO",
                    "type": "auth",
                    "message": (
                        f"Principal {principal} mapped to ADMIN_FULL_TRUST tier "
                        "via certificate fingerprint"
                    ),
                    "principal": principal,
                    "session_id": None,
                })
                return PrivilegeTier.ADMIN_FULL_TRUST

            if _matches_fingerprint(
                fingerprint,
                "ADMIN_REVIEWER_CERT_FINGERPRINTS",
                "ADMIN_REVIEWER_CERTS",
            ):
                logger.info({
                    "level": "INFO",
                    "type": "auth",
                    "message": (
                        f"Principal {principal} mapped to ADMIN_REVIEWER tier "
                        "via certificate fingerprint"
                    ),
                    "principal": principal,
                    "session_id": None,
                })
                return PrivilegeTier.ADMIN_REVIEWER

        else:
            # Compatibility only for manually constructed legacy cert:CN
            # principals. Real cert_utils principals are SHA-256 fingerprints.
            legacy_cn = cert_identity.lower()

            root_certs = _parse_env_list("ROOT_ADMIN_CERTS")
            if any(rc.lower() in legacy_cn for rc in root_certs):
                logger.info({
                    "level": "INFO",
                    "type": "auth",
                    "message": (
                        f"Legacy principal {principal} mapped to ROOT_ADMIN tier "
                        "via certificate CN compatibility"
                    ),
                    "principal": principal,
                    "session_id": None,
                })
                return PrivilegeTier.ROOT_ADMIN

            full_certs = _parse_env_list("ADMIN_FULL_TRUST_CERTS")
            if any(fc.lower() in legacy_cn for fc in full_certs):
                logger.info({
                    "level": "INFO",
                    "type": "auth",
                    "message": (
                        f"Legacy principal {principal} mapped to "
                        "ADMIN_FULL_TRUST tier via certificate CN compatibility"
                    ),
                    "principal": principal,
                    "session_id": None,
                })
                return PrivilegeTier.ADMIN_FULL_TRUST

            reviewer_certs = _parse_env_list("ADMIN_REVIEWER_CERTS")
            if any(rc.lower() in legacy_cn for rc in reviewer_certs):
                logger.info({
                    "level": "INFO",
                    "type": "auth",
                    "message": (
                        f"Legacy principal {principal} mapped to "
                        "ADMIN_REVIEWER tier via certificate CN compatibility"
                    ),
                    "principal": principal,
                    "session_id": None,
                })
                return PrivilegeTier.ADMIN_REVIEWER
    
    # ===== DEV BYPASS =====
    if principal_source == "dev_bypass":
        logger.warning({
            "level": "WARNING",
            "type": "auth",
            "message": "Dev bypass principal (localhost development only)",
            "principal": principal,
            "session_id": None
        })
        return PrivilegeTier.STANDARD_USER
    
    # Default
    logger.debug({
        "level": "DEBUG",
        "type": "auth",
        "message": f"Principal {principal} mapped to STANDARD_USER tier (no elevation found)",
        "principal": principal,
        "session_id": None
    })
    return PrivilegeTier.STANDARD_USER


def should_apply_admin_boost(
    trust_factors: Dict[str, float],
    tier: PrivilegeTier | int,
    domain: str
) -> bool:
    """Compatibility wrapper for the live admin trust-boost policy."""
    from webapp.parser.trust_authority import (
        should_apply_admin_boost_policy,
    )

    return should_apply_admin_boost_policy(
        trust_factors,
        tier,
        domain,
        domain_allowlisted=is_domain_in_allowlist(domain),
    )


def is_domain_in_allowlist(domain: str) -> bool:
    """Check if domain is in verified allowlist."""
    allowlist = _parse_env_list("URL_ALLOWLIST_HOSTS")
    domain_lower = domain.lower()
    
    # Exact match or suffix match (e.g., *.sos.ca.gov)
    for allowed in allowlist:
        allowed_lower = allowed.lower()
        if domain_lower == allowed_lower or domain_lower.endswith("." + allowed_lower):
            return True
    
    return False


def _parse_env_list(env_var: str) -> list[str]:
    """Parse comma-separated env var into list of trimmed strings."""
    raw = os.environ.get(env_var, "")
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


# ============================================================================
# PRIVILEGE TIER DECORATORS & VALIDATORS
# ============================================================================

def require_minimum_tier(minimum_tier: int):
    """Decorator to enforce minimum privilege tier for a function."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            tier = kwargs.get("privilege_tier")
            if tier is not None and not tier_satisfies(tier, minimum_tier):
                tier_name = PrivilegeTier(int(tier)).name_display
                required_name = PrivilegeTier(minimum_tier).name_display
                raise PermissionError(
                    f"Function {func.__name__} requires {required_name} tier, "
                    f"but got {tier_name}"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# ENVIRONMENT VARIABLE EXAMPLES (For .env.example)
# ============================================================================

ENV_EXAMPLES = """
# Privilege Tier Configuration

# Root admin OID principals (comma-separated SSO OIDs)
# Example: root@elections.maryland.gov, root@sos.ca.gov
# These get FULL ROOT_ADMIN access (tier 3)
ROOT_ADMIN_PRINCIPALS=

# Full trust admin OIDs (tier 2)
# Example: admin@elections.maryland.gov
ADMIN_FULL_TRUST_PRINCIPALS=

# Reviewer OIDs (tier 1)
# Example: reviewer@elections.maryland.gov
ADMIN_REVIEWER_PRINCIPALS=

# Root admin certificate CNs (comma-separated substrings)
# If cert CN contains any of these, grant ROOT_ADMIN tier
# Example: root-admin, root_signer
ROOT_ADMIN_CERTS=

# Full trust admin certificate CNs (tier 2)
ADMIN_FULL_TRUST_CERTS=

# Reviewer certificate CNs (tier 1)
ADMIN_REVIEWER_CERTS=

# Verified allowlist hosts (comma-separated domains)
# Admin boost will only apply to these domains + gov/verified domains
# Example: trusted.elections.org, sos.ca.gov, elections.maryland.gov
URL_ALLOWLIST_HOSTS=
"""

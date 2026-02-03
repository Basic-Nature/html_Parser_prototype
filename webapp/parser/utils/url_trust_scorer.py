"""URL Trust Scoring System for Smart Elections Parser

Implements intelligent URL verification using verified data references,
domain pattern analysis, and phishing detection to ensure election data
integrity while preventing SSRF and malicious site exploitation.

Trust Score Scale:
  90-100: Verified government sites (direct navigation)
  70-89:  Known government sites (direct navigation)
  50-69:  Medium-trust sites (DOM snapshot mode)
  30-49:  Low-trust sites (quarantine for review)
  0-29:   Blocked/suspicious (reject)

Author: Smart Elections Team
Date: February 2026
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

try:
    import Levenshtein  # type: ignore[import-not-found]
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False

from ..config import (
    LOG_DIR,
    PROJECT_ROOT,
    URL_ALLOWLIST_HOSTS,
    URL_ALLOWLIST_SUFFIXES,
)
from .logger_singleton import logger
from .privilege_tiers import (
    PrivilegeTier,
    get_principal_tier,
    should_apply_admin_boost,
)
from .telemetry import emit_telemetry_event

# Trust score thresholds
TRUST_THRESHOLD_HIGH = 80      # Direct navigation allowed
TRUST_THRESHOLD_MEDIUM = 50    # DOM snapshot mode
TRUST_THRESHOLD_LOW = 30       # Quarantine for manual review
# Below LOW threshold = blocked/rejected

# Verified data cache path (synced from Google Drive)
VERIFIED_DATA_DIR = Path(PROJECT_ROOT) / "webapp" / "parser" / "Context_Integration" / "verified_data"
VERIFIED_DOMAINS_FILE = VERIFIED_DATA_DIR / "verified_domains.json"
TRUST_HISTORY_FILE = LOG_DIR / "trust_history.jsonl"

# Domain trust patterns (government sites)
GOV_DOMAIN_PATTERNS = [
    r"\.gov$",
    r"\.state\.[a-z]{2}\.us$",
    r"\.co\.[a-z]{2}\.us$",
    r"[a-z]{2}\.gov$",
    r"elections?\.[a-z]{2}\.gov$",
    r"sos\.[a-z]{2}\.gov$",
]

# Suspicious TLDs (common in phishing)
SUSPICIOUS_TLDS = {
    ".xyz", ".top", ".loan", ".click", ".win", ".date", ".download",
    ".stream", ".racing", ".bid", ".trade", ".science", ".party",
    ".cricket", ".accountant", ".faith", ".review", ".country",
}

# Typosquat detection patterns
PHISHING_INDICATORS = [
    (r"goo+gle", "google"),           # Extra letters
    (r"e1ections?", "elections"),     # L33t speak
    (r"gov\.com$", ".gov"),           # Wrong TLD
    (r"gov-.*\.com$", ".gov"),        # Gov prefix on commercial domain
    (r"secure-.*\.com$", ""),         # Fake security prefix
]


def _load_verified_domains() -> Dict[str, Any]:
    """Load verified domains from cached file (synced from Google Drive).
    
    Returns dict with structure:
    {
        "domains": ["elections.maryland.gov", "sos.ca.gov", ...],
        "patterns": [r".*\\.maryland\\.gov$", ...],
        "last_synced": "2026-02-02T10:00:00Z"
    }
    """
    if not VERIFIED_DOMAINS_FILE.exists():
        return {"domains": [], "patterns": [], "last_synced": None}
    try:
        with open(VERIFIED_DOMAINS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"domains": [], "patterns": [], "last_synced": None}
        return data
    except Exception as e:
        logger.warning({
            "level": "WARNING",
            "type": "trust_scorer",
            "message": f"Failed to load verified domains: {e}",
            "session_id": None
        })
        return {"domains": [], "patterns": [], "last_synced": None}


def _load_trust_history(url: str, lookback_days: int = 30) -> Dict[str, Any]:
    """Load historical trust/success data for a URL domain from JSONL log.
    
    Returns:
    {
        "total_attempts": int,
        "successful_parses": int,
        "failed_parses": int,
        "success_rate": float (0.0-1.0),
        "last_seen": str (ISO timestamp)
    }
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
    except Exception:
        return {"total_attempts": 0, "successful_parses": 0, "failed_parses": 0, "success_rate": 0.0, "last_seen": None}
    
    if not TRUST_HISTORY_FILE.exists():
        return {"total_attempts": 0, "successful_parses": 0, "failed_parses": 0, "success_rate": 0.0, "last_seen": None}
    
    cutoff_ts = time.time() - (lookback_days * 86400)
    total = 0
    success = 0
    failed = 0
    last_seen = None
    
    try:
        with open(TRUST_HISTORY_FILE, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    import orjson
                    entry = orjson.loads(line)
                except Exception:
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                
                entry_domain = entry.get("domain", "").lower()
                if entry_domain != domain:
                    continue
                
                entry_ts = entry.get("timestamp")
                if isinstance(entry_ts, (int, float)) and entry_ts < cutoff_ts:
                    continue
                
                total += 1
                if entry.get("status") == "success":
                    success += 1
                elif entry.get("status") in ("error", "fail"):
                    failed += 1
                
                if not last_seen or (isinstance(entry_ts, str) and entry_ts > last_seen):
                    last_seen = entry_ts
    except Exception:
        pass
    
    success_rate = (success / total) if total > 0 else 0.0
    return {
        "total_attempts": total,
        "successful_parses": success,
        "failed_parses": failed,
        "success_rate": success_rate,
        "last_seen": last_seen
    }


def _log_trust_decision(url: str, score: int, factors: Dict[str, Any], action: str, session_id: str | None = None) -> None:
    """Append trust scoring decision to JSONL audit log."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
    except Exception:
        domain = "unknown"
    
    entry = {
        "timestamp": time.time(),
        "url": url,
        "domain": domain,
        "trust_score": score,
        "action": action,  # "allow_direct", "use_snapshot", "quarantine", "reject"
        "factors": factors,
        "session_id": session_id
    }
    
    try:
        import orjson
        line = orjson.dumps(entry) + b"\n"
        with open(TRUST_HISTORY_FILE, "ab") as f:
            f.write(line)
    except Exception:
        pass


def get_domain_trust_factors(url: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Analyze URL and return breakdown of trust score components.
    
    Args:
        url: URL to analyze
        context: Optional context with state/county/contest hints
    
    Returns:
        Dict with trust factor breakdown:
        {
            "verified_domain": bool,          # In verified data cache
            "gov_domain": bool,               # .gov or state.us TLD
            "allowlist_match": bool,          # In URL_ALLOWLIST_SUFFIXES
            "historical_success": float,      # 0.0-1.0 success rate
            "suspicious_tld": bool,           # In SUSPICIOUS_TLDS set
            "phishing_indicators": list[str], # Detected phishing patterns
            "domain_age_days": int | None,    # Estimated domain age (if available)
            "ssl_valid": bool | None,         # SSL cert validity (if checked)
        }
    """
    factors = {
        "verified_domain": False,
        "gov_domain": False,
        "allowlist_match": False,
        "historical_success": 0.0,
        "suspicious_tld": False,
        "phishing_indicators": [],
        "domain_age_days": None,
        "ssl_valid": None,
    }
    
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        scheme = parsed.scheme.lower()
    except Exception:
        return factors
    
    # Check verified domains cache
    verified_data = _load_verified_domains()
    if domain in verified_data.get("domains", []):
        factors["verified_domain"] = True
    else:
        # Check against verified patterns
        for pattern in verified_data.get("patterns", []):
            try:
                if re.search(pattern, domain, re.IGNORECASE):
                    factors["verified_domain"] = True
                    break
            except Exception:
                continue
    
    # Check government domain patterns
    for pattern in GOV_DOMAIN_PATTERNS:
        try:
            if re.search(pattern, domain, re.IGNORECASE):
                factors["gov_domain"] = True
                break
        except Exception:
            continue
    
    # Check URL allowlist
    if URL_ALLOWLIST_SUFFIXES:
        for suffix in URL_ALLOWLIST_SUFFIXES:
            if domain.endswith(suffix.lower()):
                factors["allowlist_match"] = True
                break
    if URL_ALLOWLIST_HOSTS and domain in [h.lower() for h in URL_ALLOWLIST_HOSTS]:
        factors["allowlist_match"] = True
    
    # Load historical success rate
    history = _load_trust_history(url)
    factors["historical_success"] = history.get("success_rate", 0.0)
    
    # Check suspicious TLD
    try:
        tld = "." + domain.split(".")[-1]
        if tld in SUSPICIOUS_TLDS:
            factors["suspicious_tld"] = True
    except Exception:
        pass
    
    # Check phishing indicators
    for pattern, expected in PHISHING_INDICATORS:
        try:
            if re.search(pattern, domain, re.IGNORECASE):
                factors["phishing_indicators"].append(f"Pattern: {pattern} (expected: {expected})")
        except Exception:
            continue
    
    # SSL validation (basic check - HTTPS required)
    if scheme == "https":
        factors["ssl_valid"] = True  # Basic check - full cert validation requires network call
    elif scheme == "http":
        factors["ssl_valid"] = False
    
    return factors


def detect_domain_mimicry(url: str, verified_urls: List[str] | None = None) -> Tuple[bool, str | None]:
    """Detect if URL domain mimics a verified government domain.
    
    Uses Levenshtein distance to find near-matches that could be typosquatting.
    
    Args:
        url: URL to check
        verified_urls: Optional list of verified URLs to compare against
                      (defaults to verified_domains.json cache)
    
    Returns:
        (is_mimic, suspected_target_domain)
        - is_mimic: True if domain is suspiciously similar to a verified domain
        - suspected_target_domain: The verified domain being mimicked (if detected)
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
    except Exception:
        return False, None
    
    # Load verified domains if not provided
    if verified_urls is None:
        verified_data = _load_verified_domains()
        verified_domains = [urlparse(u).netloc.lower() if u.startswith("http") else u.lower()
                           for u in verified_data.get("domains", [])]
    else:
        verified_domains = [urlparse(u).netloc.lower() if u.startswith("http") else u.lower()
                           for u in verified_urls]
    
    if not verified_domains:
        return False, None
    
    # Quick exact match check (not a mimic if exact match)
    if domain in verified_domains:
        return False, None
    
    # Levenshtein distance check (if library available)
    if HAS_LEVENSHTEIN:
        for verified_domain in verified_domains:
            try:
                distance = Levenshtein.distance(domain, verified_domain)
                # Threshold: 1-3 character difference suggests typosquatting
                # Adjust based on domain length (longer domains allow slightly higher threshold)
                threshold = min(3, max(1, len(verified_domain) // 10))
                if 0 < distance <= threshold:
                    return True, verified_domain
            except Exception:
                continue
    else:
        # Fallback: simple character-by-character comparison
        for verified_domain in verified_domains:
            if len(domain) == len(verified_domain):
                diff_count = sum(1 for a, b in zip(domain, verified_domain) if a != b)
                if 0 < diff_count <= 2:
                    return True, verified_domain
            elif abs(len(domain) - len(verified_domain)) == 1:
                # One character added/removed
                shorter, longer = (domain, verified_domain) if len(domain) < len(verified_domain) else (verified_domain, domain)
                for i in range(len(longer)):
                    if longer[:i] + longer[i+1:] == shorter:
                        return True, verified_domain
    
    return False, None


def compute_trust_score(url: str, context: Dict[str, Any] | None = None, session_id: str | None = None, principal: str | None = None, principal_source: str | None = None) -> Tuple[int, Dict[str, Any]]:
    """Compute trust score for a URL (0-100 scale) with tier-aware admin boost.
    
    Args:
        url: URL to score
        context: Optional context with state/county/contest hints
        session_id: Optional session ID for logging
        principal: Optional principal identifier (for admin boost eligibility)
        principal_source: Optional source of principal (e.g., "sso_oid", "cert_cn")
    
    Returns:
        (trust_score, factors_dict)
        - trust_score: Integer 0-100 indicating trust level
        - factors_dict: Breakdown of scoring components (from get_domain_trust_factors)
        - privilege_tier: Tier of principal (if provided)
    
    Scoring Algorithm:
        Base score: 0
        + 50 if verified_domain (in verified data cache)
        + 40 if gov_domain (.gov or state.us)
        + 20 if allowlist_match
        + 20 * historical_success (0-20 based on success rate)
        - 30 if suspicious_tld
        - 50 if phishing_indicators detected
        - 40 if domain_mimicry detected
        - 20 if no SSL (http://)
        + [5-10] if admin boost applies (tier-dependent)
        
        Final score clamped to 0-100
    """
    factors = get_domain_trust_factors(url, context)
    
    # Base score calculation
    score = 0
    
    # Verified domain (highest trust)
    if factors["verified_domain"]:
        score += 50
        logger.debug({
            "level": "DEBUG",
            "type": "trust_scorer",
            "message": "[TrustScore] Verified domain: +50",
            "session_id": session_id,
            "url": url
        })
    
    # Government domain pattern
    if factors["gov_domain"]:
        score += 40
        logger.debug({
            "level": "DEBUG",
            "type": "trust_scorer",
            "message": "[TrustScore] Gov domain: +40",
            "session_id": session_id,
            "url": url
        })
    
    # Allowlist match
    if factors["allowlist_match"]:
        score += 20
        logger.debug({
            "level": "DEBUG",
            "type": "trust_scorer",
            "message": "[TrustScore] Allowlist match: +20",
            "session_id": session_id,
            "url": url
        })
    
    # Historical success rate (0-20 points)
    historical_points = int(factors["historical_success"] * 20)
    if historical_points > 0:
        score += historical_points
        logger.debug({
            "level": "DEBUG",
            "type": "trust_scorer",
            "message": f"[TrustScore] Historical success ({factors['historical_success']:.1%}): +{historical_points}",
            "session_id": session_id,
            "url": url
        })
    
    # Suspicious TLD (penalty)
    if factors["suspicious_tld"]:
        score -= 30
        logger.warning({
            "level": "WARNING",
            "type": "trust_scorer",
            "message": "[TrustScore] Suspicious TLD: -30",
            "session_id": session_id,
            "url": url
        })
    
    # Phishing indicators (penalty)
    if factors["phishing_indicators"]:
        score -= 50
        logger.warning({
            "level": "WARNING",
            "type": "trust_scorer",
            "message": "[TrustScore] Phishing indicators detected: -50",
            "session_id": session_id,
            "url": url,
            "indicators": factors["phishing_indicators"]
        })
    
    # Domain mimicry check (penalty)
    is_mimic, target_domain = detect_domain_mimicry(url)
    if is_mimic:
        score -= 40
        factors["domain_mimicry"] = {"detected": True, "target": target_domain}
        logger.warning({
            "level": "WARNING",
            "type": "trust_scorer",
            "message": f"[TrustScore] Domain mimicry detected (mimics {target_domain}): -40",
            "session_id": session_id,
            "url": url
        })
    else:
        factors["domain_mimicry"] = {"detected": False, "target": None}
    
    # SSL validation (penalty for http://)
    if factors["ssl_valid"] is False:
        score -= 20
        logger.warning({
            "level": "WARNING",
            "type": "trust_scorer",
            "message": "[TrustScore] No SSL (http://): -20",
            "session_id": session_id,
            "url": url
        })
    
    # Admin boost (tier-aware, security boundary enforced)
    privilege_tier = None
    admin_boost_applied = False
    if principal and principal_source:
        try:
            privilege_tier = get_principal_tier(principal, principal_source)
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Check if admin boost should apply (security boundary enforced in should_apply_admin_boost)
            if should_apply_admin_boost(factors, privilege_tier, domain):
                # Boost amount depends on tier (REVIEWER: +5, FULL_TRUST/ROOT_ADMIN: +10)
                boost_amount = 10 if privilege_tier in (PrivilegeTier.ADMIN_FULL_TRUST, PrivilegeTier.ROOT_ADMIN) else 5
                score += boost_amount
                admin_boost_applied = True
                logger.info({
                    "level": "INFO",
                    "type": "trust_scorer",
                    "message": f"[TrustScore] Admin boost applied (tier={privilege_tier.name}): +{boost_amount}",
                    "session_id": session_id,
                    "url": url,
                    "principal": principal,
                    "privilege_tier": privilege_tier.value
                })
        except Exception as e:
            logger.debug({
                "level": "DEBUG",
                "type": "trust_scorer",
                "message": f"[TrustScore] Admin boost check failed: {e}",
                "session_id": session_id,
                "principal": principal
            })
    
    factors["admin_boost_applied"] = admin_boost_applied
    factors["privilege_tier"] = privilege_tier.value if privilege_tier else None
    
    # Clamp score to 0-100
    final_score = max(0, min(100, score))
    
    # Determine action based on score
    if final_score >= TRUST_THRESHOLD_HIGH:
        action = "allow_direct"
    elif final_score >= TRUST_THRESHOLD_MEDIUM:
        action = "use_snapshot"
    elif final_score >= TRUST_THRESHOLD_LOW:
        action = "quarantine"
    else:
        action = "reject"
    
    # Log trust decision
    _log_trust_decision(url, final_score, factors, action, session_id)
    
    # Emit telemetry
    try:
        emit_telemetry_event("trust_score_computed", {
            "url": url,
            "score": final_score,
            "action": action,
            "session_id": session_id,
            "verified_domain": factors["verified_domain"],
            "gov_domain": factors["gov_domain"],
            "suspicious_tld": factors["suspicious_tld"],
            "phishing_indicators_count": len(factors["phishing_indicators"]),
            "domain_mimicry": is_mimic
        })
    except Exception:
        pass
    
    logger.info({
        "level": "INFO",
        "type": "trust_scorer",
        "message": f"[TrustScore] Final score: {final_score}/100 → Action: {action}",
        "session_id": session_id,
        "url": url,
        "trust_score": final_score,
        "action": action
    })
    
    return final_score, factors


def should_use_snapshot_mode(trust_score: int, url: str) -> bool:
    """Determine if URL should use DOM snapshot mode instead of full navigation.
    
    Args:
        trust_score: Computed trust score (0-100)
        url: URL being assessed
    
    Returns:
        True if DOM snapshot mode should be used (medium-trust range)
    """
    return TRUST_THRESHOLD_MEDIUM <= trust_score < TRUST_THRESHOLD_HIGH


def should_quarantine(trust_score: int, url: str, privilege_tier: PrivilegeTier | None = None) -> bool:
    """Determine if URL should be quarantined for manual review.
    
    Tier-aware: ROOT_ADMIN/ADMIN_FULL_TRUST bypass quarantine for low-trust URLs.
    
    Args:
        trust_score: Computed trust score (0-100)
        url: URL being assessed
        privilege_tier: Optional privilege tier (if provided, tier-specific logic applies)
    
    Returns:
        True if URL should be quarantined (low-trust range)
    
    Tier Logic:
        - ROOT_ADMIN: Never quarantine (bypass to direct processing)
        - ADMIN_FULL_TRUST: Stricter thresholds, but still allowed to process
        - REVIEWER/USER: Standard thresholds
    """
    # ROOT_ADMIN bypasses quarantine (but isolation tracking still applies)
    if privilege_tier == PrivilegeTier.ROOT_ADMIN:
        return False
    
    # ADMIN_FULL_TRUST: stricter quarantine threshold (40 instead of 30)
    if privilege_tier == PrivilegeTier.ADMIN_FULL_TRUST:
        admin_quarantine_low = 40
        return admin_quarantine_low <= trust_score < TRUST_THRESHOLD_MEDIUM
    
    # Standard quarantine range for REVIEWER and USER
    return TRUST_THRESHOLD_LOW <= trust_score < TRUST_THRESHOLD_MEDIUM


def should_reject(trust_score: int, url: str, privilege_tier: PrivilegeTier | None = None) -> bool:
    """Determine if URL should be rejected outright.
    
    Tier-aware: ROOT_ADMIN/ADMIN_FULL_TRUST bypass rejection for very-low-trust URLs.
    
    Args:
        trust_score: Computed trust score (0-100)
        url: URL being assessed
        privilege_tier: Optional privilege tier (if provided, tier-specific logic applies)
    
    Returns:
        True if URL should be rejected (below low threshold)
    
    Tier Logic:
        - ROOT_ADMIN: Never reject (but logged + isolated)
        - ADMIN_FULL_TRUST: Stricter rejection (< 20 instead of < 30)
        - REVIEWER/USER: Standard rejection threshold
    
    Security Note:
        Rejection bypasses do NOT apply to:
        - Phishing indicators
        - Domain mimicry + suspicious indicators
        - Known malware/ransomware domains
        
        Those must be handled by security team.
    """
    # ROOT_ADMIN bypasses rejection (but isolation & audit logging apply)
    if privilege_tier == PrivilegeTier.ROOT_ADMIN:
        logger.warning({
            "level": "WARNING",
            "type": "trust_scorer",
            "message": "[TrustScore] ROOT_ADMIN bypassing rejection for very-low-trust URL",
            "url": url,
            "trust_score": trust_score,
            "privilege_tier": "ROOT_ADMIN"
        })
        return False
    
    # ADMIN_FULL_TRUST: stricter rejection threshold (20 instead of 30)
    if privilege_tier == PrivilegeTier.ADMIN_FULL_TRUST:
        admin_reject_low = 20
        return trust_score < admin_reject_low
    
    # Standard rejection for REVIEWER and USER
    return trust_score < TRUST_THRESHOLD_LOW

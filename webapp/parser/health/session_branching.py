"""Session Branching and Multi-Tenant Isolation for Smart Elections Parser

Implements per-principal URL isolation to prevent cross-tenant data leakage.
Each principal maintains a SessionBranch with:
  - Quarantined URLs (under review)
  - Rejected URLs (blocked)
  - Can-access validation (multi-tenant enforcement)

Author: Smart Elections Team
Date: February 2026
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Set

from ..utils.logger_singleton import logger
from ..utils.privilege_tiers import PrivilegeTier, get_principal_tier


class SessionBranch:
    """Per-principal URL isolation container.
    
    Maintains per-tenant lists of quarantined and rejected URLs to prevent
    cross-principal access to analysis results.
    """
    
    def __init__(self, principal: str):
        """Initialize session branch for principal.
        
        Args:
            principal: Principal identifier (e.g., "sso:oid-value")
        """
        self.principal = principal
        self.quarantined_urls: Set[str] = set()
        self.rejected_urls: Set[str] = set()
        self.accessed_urls: Dict[str, float] = {}  # URL -> timestamp
        self._lock = threading.RLock()
    
    def add_quarantined_url(self, url: str) -> bool:
        """Add URL to quarantine list.
        
        Args:
            url: URL to quarantine
        
        Returns:
            True if added, False if already present.
        """
        with self._lock:
            if url in self.quarantined_urls:
                return False
            self.quarantined_urls.add(url)
            return True
    
    def add_rejected_url(self, url: str) -> bool:
        """Add URL to rejection list.
        
        Args:
            url: URL to reject
        
        Returns:
            True if added, False if already present.
        """
        with self._lock:
            if url in self.rejected_urls:
                return False
            self.rejected_urls.add(url)
            return True
    
    def is_quarantined(self, url: str) -> bool:
        """Check if URL is quarantined for this principal.
        
        Args:
            url: URL to check
        
        Returns:
            True if quarantined, False otherwise.
        """
        with self._lock:
            return url in self.quarantined_urls
    
    def is_rejected(self, url: str) -> bool:
        """Check if URL is rejected for this principal.
        
        Args:
            url: URL to check
        
        Returns:
            True if rejected, False otherwise.
        """
        with self._lock:
            return url in self.rejected_urls
    
    def can_access_url(self, url: str, access_type: str = "view") -> bool:
        """Check if principal can access URL (multi-tenant isolation enforcement).
        
        Args:
            url: URL to check access for
            access_type: Type of access ("view", "quarantine", "reject")
        
        Returns:
            True if principal can access, False if isolation breach.
        """
        with self._lock:
            # Quarantine access
            if access_type == "quarantine" and url in self.quarantined_urls:
                return True
            
            # Reject access
            if access_type == "reject" and url in self.rejected_urls:
                return True
            
            # Default: no access
            return False
    
    def record_access(self, url: str, timestamp: float | None = None) -> None:
        """Record URL access for audit trail.
        
        Args:
            url: URL accessed
            timestamp: Access timestamp (defaults to now)
        """
        import time
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            self.accessed_urls[url] = timestamp
    
    def get_summary(self) -> Dict[str, Any]:
        """Get isolation summary for audit purposes.
        
        Returns:
            Dict with quarantined/rejected count and accessed URLs.
        """
        with self._lock:
            return {
                "principal": self.principal,
                "quarantined_count": len(self.quarantined_urls),
                "rejected_count": len(self.rejected_urls),
                "accessed_count": len(self.accessed_urls),
                "quarantined_urls": list(self.quarantined_urls)[:20],  # Limit for logging
                "rejected_urls": list(self.rejected_urls)[:20],
            }


# Global isolation map: principal -> SessionBranch
_BRANCH_ISOLATION_MAP: Dict[str, SessionBranch] = {}
_ISOLATION_MAP_LOCK = threading.RLock()


def get_isolated_branch(principal: str | None) -> SessionBranch | None:
    """Retrieve or create isolation branch for principal.
    
    Args:
        principal: Principal identifier
    
    Returns:
        SessionBranch for principal, or None if principal is invalid.
    """
    if not principal or not isinstance(principal, str):
        return None
    
    with _ISOLATION_MAP_LOCK:
        if principal not in _BRANCH_ISOLATION_MAP:
            _BRANCH_ISOLATION_MAP[principal] = SessionBranch(principal)
        return _BRANCH_ISOLATION_MAP[principal]


def validate_url_access(principal: str | None, url: str, access_type: str = "view", principal_source: str | None = None) -> tuple[bool, str]:
    """Validate multi-tenant URL access with privilege consideration.
    
    Args:
        principal: Principal requesting access
        url: URL to access
        access_type: Type of access ("view", "quarantine", "reject")
        principal_source: Source of principal (for tier resolution)
    
    Returns:
        (allowed: bool, reason: str)
    
    Logic:
        - ROOT_ADMIN tier: Always allowed (audit logged)
        - Other tiers: Must have URL in their isolation branch
    """
    if not principal:
        return False, "missing_principal"
    
    branch = get_isolated_branch(principal)
    if not branch:
        return False, "invalid_principal"
    
    # ROOT_ADMIN bypass (but still audited)
    if principal_source:
        try:
            tier = get_principal_tier(principal, principal_source)
            if tier == PrivilegeTier.ROOT_ADMIN:
                # Log root admin access for audit
                logger.info({
                    "level": "INFO",
                    "type": "isolation",
                    "message": "Root admin accessed URL from different isolation branch",
                    "principal": principal,
                    "url": url,
                    "access_type": access_type
                })
                branch.record_access(url)
                return True, "root_admin_override"
        except Exception:
            pass  # Fall through to standard isolation check
    
    # Standard multi-tenant enforcement
    can_access = branch.can_access_url(url, access_type)
    if can_access:
        branch.record_access(url)
        return True, "authorized"
    
    logger.warning({
        "level": "WARNING",
        "type": "isolation",
        "message": f"Isolation breach attempted: {access_type}",
        "principal": principal,
        "url": url
    })
    return False, "isolation_breach"


def add_url_to_isolation(principal: str, url: str, status: str) -> bool:
    """Add URL to principal's isolation branch (quarantine or reject).
    
    Args:
        principal: Principal identifier
        url: URL to isolate
        status: Isolation status ("quarantine" or "reject")
    
    Returns:
        True if added, False otherwise.
    """
    branch = get_isolated_branch(principal)
    if not branch:
        return False
    
    if status == "quarantine":
        added = branch.add_quarantined_url(url)
    elif status == "reject":
        added = branch.add_rejected_url(url)
    else:
        return False
    
    if added:
        logger.info({
            "level": "INFO",
            "type": "isolation",
            "message": f"URL added to {status} isolation",
            "principal": principal,
            "url": url
        })
    
    return added


def get_isolation_summary(principal: str) -> Dict[str, Any] | None:
    """Get isolation summary for principal (admin audit).
    
    Args:
        principal: Principal identifier
    
    Returns:
        Isolation branch summary dict, or None if not found.
    """
    branch = get_isolated_branch(principal)
    if not branch:
        return None
    return branch.get_summary()


def list_all_isolation_branches() -> list[Dict[str, Any]]:
    """List all active isolation branches (admin use only).
    
    Returns:
        List of isolation branch summaries.
    
    WARNING:
        Should only be called by trusted admins for auditing.
    """
    with _ISOLATION_MAP_LOCK:
        return [branch.get_summary() for branch in _BRANCH_ISOLATION_MAP.values()]


def cleanup_principal_isolation(principal: str) -> bool:
    """Remove isolation branch for principal (on logout).
    
    Args:
        principal: Principal identifier
    
    Returns:
        True if removed, False if not found.
    """
    with _ISOLATION_MAP_LOCK:
        if principal in _BRANCH_ISOLATION_MAP:
            _BRANCH_ISOLATION_MAP.pop(principal)
            logger.info({
                "level": "INFO",
                "type": "isolation",
                "message": "Isolation branch cleaned up",
                "principal": principal
            })
            return True
    return False

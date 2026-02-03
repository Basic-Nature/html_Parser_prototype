"""Root Admin Session Management for Smart Elections Parser

Implements tokenized root admin session handling with Linux UID=0 verification.
Prevents privilege escalation through session manipulation by requiring both
valid token AND root UID confirmation.

Token Strategy:
  - 64-char hex token generated on first root admin login
  - Stored in ROOT_ADMIN_TOKEN environment variable
  - Verified on each /root_admin_login request
  - Session bound to sess_root_admin_TIMESTAMP format
  - Audit logged to admin_full_trust_decisions.jsonl

Author: Smart Elections Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import os
import secrets
import time
from typing import Any, Dict

from .logger_singleton import logger

# Root admin token (64-char hex, set via environment)
ROOT_ADMIN_TOKEN = os.environ.get("ROOT_ADMIN_TOKEN")

# Timeout for root admin sessions (seconds)
ROOT_ADMIN_SESSION_TIMEOUT = 3600

# Storage for active root admin sessions (session_id -> expiry_time)
_ROOT_ADMIN_SESSIONS: Dict[str, float] = {}
_ROOT_ADMIN_LOCK = __import__("threading").Lock()


def generate_root_admin_token() -> str:
    """Generate new 64-char hex root admin token.
    
    Returns:
        64-character hex string suitable for ROOT_ADMIN_TOKEN env var.
    
    Raises:
        RuntimeError: If entropy source unavailable.
    """
    try:
        # Use secrets module for cryptographically strong random
        token_bytes = secrets.token_bytes(32)  # 32 bytes = 64 hex chars
        return token_bytes.hex()
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "root_admin",
            "message": f"Failed to generate root admin token: {e}",
            "session_id": None
        })
        raise RuntimeError(f"Cannot generate root admin token: {e}")


def hash_token(token: str) -> str:
    """Hash token for safe comparison (constant-time).
    
    Args:
        token: Plain text token to hash
    
    Returns:
        SHA256 hex digest
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def verify_root_admin_token(provided_token: str, stored_token: str | None = None) -> bool:
    """Verify root admin token using constant-time comparison.
    
    Args:
        provided_token: Token provided by user
        stored_token: Token to compare against (defaults to ROOT_ADMIN_TOKEN env var)
    
    Returns:
        True if tokens match, False otherwise.
    
    Note:
        Uses hmac.compare_digest() to prevent timing attacks.
    """
    if not provided_token or not isinstance(provided_token, str):
        return False
    
    stored = stored_token or ROOT_ADMIN_TOKEN
    if not stored:
        return False
    
    try:
        import hmac
        return hmac.compare_digest(hash_token(provided_token), hash_token(stored))
    except Exception:
        return False


def check_is_root_uid() -> bool:
    """Check if current process is running as root (UID=0).
    
    Returns:
        True if UID=0, False otherwise.
    
    Note:
        On Windows, always returns False (root concept not applicable).
    """
    try:
        if os.name == "nt":  # Windows
            return False
        current_uid = os.getuid()
        return current_uid == 0
    except Exception:
        return False


def create_root_admin_session(principal: str, principal_source: str, session_id: str | None = None) -> str:
    """Create new root admin session bound to principal identity.
    
    Args:
        principal: Principal identifier (e.g., "sso:oid-value" or "cert:CN=root")
        principal_source: Source of principal (e.g., "sso_oid", "cert_cn")
        session_id: Optional explicit session ID (for testing)
    
    Returns:
        Root admin session ID (format: sess_root_admin_TIMESTAMP_RANDOM)
    
    Raises:
        PermissionError: If not running as UID=0
        ValueError: If principal invalid
    
    Side Effects:
        - Adds session to _ROOT_ADMIN_SESSIONS
        - Logs to logger (admin_full_trust_decisions audit trail)
    """
    # Verify UID=0
    if not check_is_root_uid():
        logger.error({
            "level": "ERROR",
            "type": "root_admin",
            "message": "Root admin session creation failed: Not running as UID=0",
            "principal": principal,
            "session_id": None
        })
        raise PermissionError("Root admin sessions require UID=0")
    
    # Validate principal
    if not principal or not isinstance(principal, str):
        logger.error({
            "level": "ERROR",
            "type": "root_admin",
            "message": "Root admin session creation failed: Invalid principal",
            "principal": principal,
            "session_id": None
        })
        raise ValueError("Principal must be non-empty string")
    
    # Generate session ID
    if session_id is None:
        timestamp = int(time.time())
        random_suffix = secrets.token_hex(8)
        session_id = f"sess_root_admin_{timestamp}_{random_suffix}"
    
    # Store session with expiry
    expiry_time = time.time() + ROOT_ADMIN_SESSION_TIMEOUT
    with _ROOT_ADMIN_LOCK:
        _ROOT_ADMIN_SESSIONS[session_id] = expiry_time
    
    # Log creation
    logger.info({
        "level": "INFO",
        "type": "root_admin",
        "message": f"Root admin session created: {session_id}",
        "session_id": session_id,
        "principal": principal,
        "principal_source": principal_source,
        "expiry_timestamp": expiry_time
    })
    
    return session_id


def is_root_admin_session(session_id: str | None) -> bool:
    """Check if session_id is valid root admin session.
    
    Args:
        session_id: Session ID to check
    
    Returns:
        True if valid (not expired), False otherwise.
    
    Side Effects:
        - Removes expired sessions from _ROOT_ADMIN_SESSIONS
    """
    if not session_id or not isinstance(session_id, str):
        return False
    
    now = time.time()
    with _ROOT_ADMIN_LOCK:
        if session_id not in _ROOT_ADMIN_SESSIONS:
            return False
        
        expiry = _ROOT_ADMIN_SESSIONS[session_id]
        if now > expiry:
            # Session expired, remove it
            _ROOT_ADMIN_SESSIONS.pop(session_id, None)
            return False
        
        return True


def get_root_admin_session_info(session_id: str) -> Dict[str, Any] | None:
    """Retrieve root admin session metadata.
    
    Args:
        session_id: Session ID to look up
    
    Returns:
        Dict with session metadata if valid, None otherwise.
    """
    if not is_root_admin_session(session_id):
        return None
    
    expiry = _ROOT_ADMIN_SESSIONS.get(session_id)
    if expiry is None:
        return None
    
    return {
        "session_id": session_id,
        "type": "root_admin",
        "expiry_timestamp": expiry,
        "time_remaining_sec": max(0, expiry - time.time())
    }


def revoke_root_admin_session(session_id: str, reason: str = "unknown") -> bool:
    """Revoke root admin session immediately.
    
    Args:
        session_id: Session ID to revoke
        reason: Reason for revocation (logged to audit trail)
    
    Returns:
        True if session was revoked, False if not found.
    
    Side Effects:
        - Removes session from _ROOT_ADMIN_SESSIONS
        - Logs to audit trail
    """
    with _ROOT_ADMIN_LOCK:
        if session_id not in _ROOT_ADMIN_SESSIONS:
            return False
        
        _ROOT_ADMIN_SESSIONS.pop(session_id)
    
    logger.warning({
        "level": "WARNING",
        "type": "root_admin",
        "message": f"Root admin session revoked: {reason}",
        "session_id": session_id
    })
    
    return True


def cleanup_expired_root_admin_sessions() -> int:
    """Remove all expired root admin sessions.
    
    Returns:
        Count of sessions removed.
    """
    now = time.time()
    removed = 0
    
    with _ROOT_ADMIN_LOCK:
        expired = [sid for sid, expiry in _ROOT_ADMIN_SESSIONS.items() if now > expiry]
        for sid in expired:
            _ROOT_ADMIN_SESSIONS.pop(sid, None)
        removed = len(expired)
    
    if removed > 0:
        logger.info({
            "level": "INFO",
            "type": "root_admin",
            "message": f"Cleaned up {removed} expired root admin sessions",
            "session_id": None
        })
    
    return removed


def list_active_root_admin_sessions() -> list[Dict[str, Any]]:
    """List all active root admin sessions (admin use only).
    
    Returns:
        List of session metadata dicts.
    
    WARNING:
        This should only be called by trusted admins for auditing.
    """
    now = time.time()
    sessions = []
    
    with _ROOT_ADMIN_LOCK:
        for session_id, expiry in _ROOT_ADMIN_SESSIONS.items():
            if now <= expiry:
                sessions.append({
                    "session_id": session_id,
                    "expiry_timestamp": expiry,
                    "time_remaining_sec": expiry - now
                })
    
    return sessions

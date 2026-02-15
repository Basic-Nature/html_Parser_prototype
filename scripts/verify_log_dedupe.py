#!/usr/bin/env python3
"""
Verification script for log deduplication behavior.

Tests the security log dedupe logic to ensure repeated auth/isolation/security
messages are suppressed within the configured window.

Usage:
    python scripts/verify_log_dedupe.py
"""

import time
from typing import Any, Dict


class MockSessionManager:
    """Mock session manager for testing dedupe logic."""
    
    def __init__(self):
        self._recent_cache: Dict[str, Dict[str, Any]] = {}
    
    def should_emit_message(
        self, 
        session_id: str, 
        cache_key: str, 
        *, 
        now: float, 
        window: float, 
        max_entries: int
    ) -> bool:
        """
        Check if message should be emitted (not a duplicate).
        
        Returns:
            True if message should be logged, False if duplicate within window.
        """
        cache = self._recent_cache.setdefault(session_id, {"seen": {}, "order": []})
        last_ts = cache["seen"].get(cache_key)
        
        if last_ts and (now - last_ts) < window:
            return False  # Duplicate within window
        
        # Update cache
        cache["seen"][cache_key] = now
        cache["order"].append(cache_key)
        
        # Trim cache
        if len(cache["order"]) > max_entries:
            overflow = len(cache["order"]) - max_entries
            for _ in range(overflow):
                old = cache["order"].pop(0)
                cache["seen"].pop(old, None)
        
        return True


def test_security_log_dedupe():
    """Test security log deduplication behavior."""
    print("=" * 60)
    print("Security Log Deduplication Test")
    print("=" * 60)
    
    session_mgr = MockSessionManager()
    session_id = "test_session_001"
    
    # Configuration
    SECURITY_LOG_DEDUPE_WINDOW = 12.0
    MAX_CACHE_PER_SESSION = 120
    
    # Simulate repeated isolation breach logs
    test_cases = [
        {
            "type": "isolation",
            "message": "URL blocked due to isolation: isolation_breach",
            "principal": "cert:abc123",
            "reason": "isolation_breach",
            "url": "https://example1.com"
        },
        {
            "type": "isolation",
            "message": "URL blocked due to isolation: isolation_breach",
            "principal": "cert:abc123",
            "reason": "isolation_breach",
            "url": "https://example2.com"
        },
        {
            "type": "isolation",
            "message": "URL blocked due to isolation: isolation_breach",
            "principal": "cert:abc123",
            "reason": "isolation_breach",
            "url": "https://example3.com"
        },
    ]
    
    print("\n1. Testing repeated isolation breach (same principal/reason, different URLs)")
    print("-" * 60)
    
    emitted_count = 0
    suppressed_count = 0
    
    now = time.time()
    for i, log_obj in enumerate(test_cases):
        msg_type = log_obj["type"]
        msg = log_obj["message"]
        principal = log_obj.get("principal", "")
        reason = log_obj.get("reason", "")
        
        # Build cache key (excludes URL to dedupe across multiple URLs)
        key = f"{msg_type}|{msg}|{principal}|{reason}"
        
        should_emit = session_mgr.should_emit_message(
            session_id,
            key,
            now=now,
            window=SECURITY_LOG_DEDUPE_WINDOW,
            max_entries=MAX_CACHE_PER_SESSION,
        )
        
        if should_emit:
            emitted_count += 1
            print(f"  ✓ Log {i+1}: EMITTED (url={log_obj['url']})")
        else:
            suppressed_count += 1
            print(f"  ✗ Log {i+1}: SUPPRESSED (duplicate within {SECURITY_LOG_DEDUPE_WINDOW}s window)")
    
    print(f"\nResult: {emitted_count} emitted, {suppressed_count} suppressed")
    assert emitted_count == 1, f"Expected 1 emission, got {emitted_count}"
    assert suppressed_count == 2, f"Expected 2 suppressions, got {suppressed_count}"
    print("✓ PASS: Only first log emitted, subsequent duplicates suppressed\n")
    
    # Test 2: Different principals should not be deduped
    print("\n2. Testing different principals (should NOT dedupe)")
    print("-" * 60)
    
    different_principal_cases = [
        {
            "type": "auth",
            "message": "Client certificate is expired.",
            "principal": "cert:user1",
            "reason": "cert_expired",
        },
        {
            "type": "auth",
            "message": "Client certificate is expired.",
            "principal": "cert:user2",
            "reason": "cert_expired",
        },
    ]
    
    emitted_count = 0
    now = time.time()
    
    for i, log_obj in enumerate(different_principal_cases):
        msg_type = log_obj["type"]
        msg = log_obj["message"]
        principal = log_obj.get("principal", "")
        reason = log_obj.get("reason", "")
        
        key = f"{msg_type}|{msg}|{principal}|{reason}"
        
        should_emit = session_mgr.should_emit_message(
            session_id,
            key,
            now=now,
            window=SECURITY_LOG_DEDUPE_WINDOW,
            max_entries=MAX_CACHE_PER_SESSION,
        )
        
        if should_emit:
            emitted_count += 1
            print(f"  ✓ Log {i+1}: EMITTED (principal={principal})")
    
    print(f"\nResult: {emitted_count} emitted")
    assert emitted_count == 2, f"Expected 2 emissions, got {emitted_count}"
    print("✓ PASS: Different principals not deduped\n")
    
    # Test 3: Window expiry allows re-emission
    print("\n3. Testing window expiry (should re-emit after window)")
    print("-" * 60)
    
    log_obj = {
        "type": "security",
        "message": "Upload blocked by guarded gate: guard_key_invalid",
        "principal": "sso:test_user",
        "reason": "guard_key_invalid",
    }
    
    msg_type = log_obj["type"]
    msg = log_obj["message"]
    principal = log_obj.get("principal", "")
    reason = log_obj.get("reason", "")
    key = f"{msg_type}|{msg}|{principal}|{reason}"
    
    # First emission
    now1 = time.time()
    should_emit_1 = session_mgr.should_emit_message(
        session_id,
        key,
        now=now1,
        window=SECURITY_LOG_DEDUPE_WINDOW,
        max_entries=MAX_CACHE_PER_SESSION,
    )
    print(f"  First attempt (t=0s): {'EMITTED' if should_emit_1 else 'SUPPRESSED'}")
    
    # Second emission within window
    now2 = now1 + 5.0  # 5 seconds later (within 12s window)
    should_emit_2 = session_mgr.should_emit_message(
        session_id,
        key,
        now=now2,
        window=SECURITY_LOG_DEDUPE_WINDOW,
        max_entries=MAX_CACHE_PER_SESSION,
    )
    print(f"  Second attempt (t=5s): {'EMITTED' if should_emit_2 else 'SUPPRESSED'}")
    
    # Third emission after window expiry
    now3 = now1 + 13.0  # 13 seconds later (outside 12s window)
    should_emit_3 = session_mgr.should_emit_message(
        session_id,
        key,
        now=now3,
        window=SECURITY_LOG_DEDUPE_WINDOW,
        max_entries=MAX_CACHE_PER_SESSION,
    )
    print(f"  Third attempt (t=13s): {'EMITTED' if should_emit_3 else 'SUPPRESSED'}")
    
    assert should_emit_1 is True, "First emission should be allowed"
    assert should_emit_2 is False, "Second emission should be suppressed (within window)"
    assert should_emit_3 is True, "Third emission should be allowed (window expired)"
    print("\n✓ PASS: Window expiry allows re-emission\n")
    
    print("=" * 60)
    print("All tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_security_log_dedupe()

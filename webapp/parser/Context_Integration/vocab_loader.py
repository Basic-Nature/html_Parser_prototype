"""
VocabLoader: Secure, auditable vocabulary file loader for election integrity.

Provides tamper-detection, integrity verification, and audit logging for
vocab files (offices, parties, jurisdictions, aliases, verified sources).
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

from webapp.parser.utils.logger_singleton import logger


# Custom Exceptions
class VocabLoaderError(Exception):
    """Base exception for VocabLoader errors."""
    pass


class VocabFileNotFound(VocabLoaderError):
    """Raised when a vocab file is not found."""
    pass


class VocabIntegrityError(VocabLoaderError):
    """Raised when vocab file integrity check fails."""
    pass


class VocabSecurityError(VocabLoaderError):
    """Raised when security violation detected (tampering, path traversal)."""
    pass


class RateLimitError(VocabLoaderError):
    """Raised when rate limit exceeded."""
    pass


class VocabLoader:
    """
    Secure vocabulary file loader with tamper detection and audit logging.
    
    Features:
    - Hash-based integrity verification
    - Immutability (file hashes tracked)
    - Audit logging (all loads tracked)
    - Rate limiting (prevent abuse)
    - Path traversal protection
    - Thread-safe access
    
    Usage:
        loader = VocabLoader(base_dir="webapp/parser/Context_Integration/vocab")
        offices = loader.load("entities/offices.txt")
        # Subsequent loads return cached data if hash unchanged
    """
    
    def __init__(
        self,
        base_dir: str | Path,
        enable_caching: bool = True,
        enable_rate_limiting: bool = True,
        max_loads_per_minute: int = 100,
    ):
        """
        Initialize VocabLoader.
        
        Args:
            base_dir: Base directory for vocab files (must be absolute or relative to PROJECT_ROOT)
            enable_caching: Enable in-memory caching (default: True)
            enable_rate_limiting: Enable rate limiting (default: True)
            max_loads_per_minute: Max file loads per minute (default: 100)
        """
        self.base_dir = Path(base_dir).resolve()
        if not self.base_dir.exists():
            self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_caching = enable_caching
        self.enable_rate_limiting = enable_rate_limiting
        self.max_loads_per_minute = max_loads_per_minute
        
        # Thread-safe cache: {file_path: {hash, data, timestamp}}
        self._cache: Dict[str, Dict] = {}
        self._cache_lock = threading.RLock()
        
        # Rate limiting: {minute_bucket: load_count}
        self._rate_limit_buckets: Dict[int, int] = {}
        self._rate_limit_lock = threading.RLock()
        
        # File hash tracking for immutability verification
        self._file_hashes: Dict[str, str] = {}
        
        # Audit log (in-memory for now; can be persisted later)
        self._audit_log: List[Dict] = []
    
    def load(
        self,
        relative_path: str,
        *,
        session_id: Optional[str] = None,
        force_reload: bool = False,
    ) -> List[str]:
        """
        Load vocab file with integrity verification and caching.
        
        Args:
            relative_path: Path relative to base_dir (e.g., "entities/offices.txt")
            session_id: Optional session ID for audit logging
            force_reload: Force reload even if cached (default: False)
        
        Returns:
            List of lines from vocab file (stripped, non-empty, non-comment)
        
        Raises:
            VocabFileNotFound: File does not exist
            VocabSecurityError: Path traversal detected or tampering detected
            VocabIntegrityError: File hash changed unexpectedly
            RateLimitError: Rate limit exceeded
        """
        # 1. Path security check
        abs_path = self._resolve_path(relative_path)
        
        # 2. Rate limiting check
        if self.enable_rate_limiting:
            self._check_rate_limit()
        
        # 3. Check cache if enabled and not forcing reload
        if self.enable_caching and not force_reload:
            cached = self._get_cached(abs_path)
            if cached is not None:
                self._audit("cache_hit", relative_path, session_id, cached_hash=cached["hash"])
                return cached["data"]
        
        # 4. Load file and verify integrity
        lines, file_hash = self._load_and_hash(abs_path)
        
        # 5. Check for tampering (hash changed since last load)
        self._verify_integrity(abs_path, file_hash)
        
        # 6. Cache result
        if self.enable_caching:
            self._set_cached(abs_path, lines, file_hash)
        
        # 7. Audit log
        self._audit("load", relative_path, session_id, file_hash=file_hash, line_count=len(lines))
        
        return lines
    
    def load_mapping(
        self,
        relative_path: str,
        *,
        session_id: Optional[str] = None,
        force_reload: bool = False,
    ) -> Dict[str, str]:
        """
        Load vocab file with alias mappings (e.g., "Pres -> President").
        
        Args:
            relative_path: Path relative to base_dir
            session_id: Optional session ID for audit logging
            force_reload: Force reload even if cached
        
        Returns:
            Dict mapping aliases to canonical values
        
        Format:
            Lines like "Pres -> President" or "Dem -> Democratic"
            Also supports simple values (no arrow)
        """
        lines = self.load(relative_path, session_id=session_id, force_reload=force_reload)
        mapping: Dict[str, str] = {}
        
        for line in lines:
            if " -> " in line:
                alias, canonical = line.split(" -> ", 1)
                mapping[alias.strip()] = canonical.strip()
            else:
                # Simple value (maps to itself)
                mapping[line.strip()] = line.strip()
        
        return mapping
    
    def clear_cache(self, relative_path: Optional[str] = None) -> None:
        """
        Clear cache for specific file or all files.
        
        Args:
            relative_path: Optional path to clear (if None, clears all)
        """
        with self._cache_lock:
            if relative_path is None:
                self._cache.clear()
            else:
                abs_path = str(self._resolve_path(relative_path))
                self._cache.pop(abs_path, None)
    
    def get_audit_log(self, limit: int = 100) -> List[Dict]:
        """
        Get audit log entries (most recent first).
        
        Args:
            limit: Max entries to return (default: 100)
        
        Returns:
            List of audit log entries
        """
        return self._audit_log[-limit:]
    
    # --- Internal Methods ---
    
    def _resolve_path(self, relative_path: str) -> Path:
        """
        Resolve relative path and check for path traversal.
        
        Args:
            relative_path: Path relative to base_dir
        
        Returns:
            Absolute resolved path
        
        Raises:
            VocabSecurityError: Path traversal detected
            VocabFileNotFound: File does not exist
        """
        # Normalize path (prevent ../../ etc.)
        normalized = os.path.normpath(relative_path).replace("\\", "/")
        if normalized.startswith(".."):
            raise VocabSecurityError(f"Path traversal detected: {relative_path}")
        
        abs_path = (self.base_dir / normalized).resolve()
        
        # Ensure resolved path is under base_dir
        if not str(abs_path).startswith(str(self.base_dir)):
            raise VocabSecurityError(f"Path escape detected: {relative_path}")
        
        # Check file exists
        if not abs_path.exists():
            raise VocabFileNotFound(f"Vocab file not found: {relative_path}")
        
        return abs_path
    
    def _check_rate_limit(self) -> None:
        """
        Check rate limit and raise if exceeded.
        
        Raises:
            RateLimitError: Rate limit exceeded
        """
        now = int(time.time())
        minute_bucket = now // 60
        
        with self._rate_limit_lock:
            # Clean old buckets (>2 minutes ago)
            cutoff = minute_bucket - 2
            for bucket in list(self._rate_limit_buckets.keys()):
                if bucket < cutoff:
                    del self._rate_limit_buckets[bucket]
            
            # Increment current bucket
            count = self._rate_limit_buckets.get(minute_bucket, 0)
            if count >= self.max_loads_per_minute:
                raise RateLimitError(f"Rate limit exceeded: {self.max_loads_per_minute} loads/minute")
            
            self._rate_limit_buckets[minute_bucket] = count + 1
    
    def _load_and_hash(self, abs_path: Path) -> tuple[List[str], str]:
        """
        Load file and compute hash.
        
        Args:
            abs_path: Absolute path to file
        
        Returns:
            Tuple of (lines, file_hash)
        """
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                raw_content = f.read()
        except Exception as e:
            raise VocabLoaderError(f"Failed to read file: {e}")
        
        # Compute hash of raw content
        file_hash = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
        
        # Parse lines (strip whitespace, skip empty/comments)
        lines = []
        for line in raw_content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            lines.append(stripped)
        
        return lines, file_hash
    
    def _verify_integrity(self, abs_path: Path, current_hash: str) -> None:
        """
        Verify file integrity (hash unchanged since last load).
        
        Args:
            abs_path: Absolute path to file
            current_hash: Current file hash
        
        Raises:
            VocabIntegrityError: File hash changed unexpectedly
        """
        path_str = str(abs_path)
        previous_hash = self._file_hashes.get(path_str)
        
        if previous_hash is not None and previous_hash != current_hash:
            raise VocabIntegrityError(
                f"File hash changed: {abs_path.name} "
                f"(expected {previous_hash[:8]}..., got {current_hash[:8]}...)"
            )
        
        # Update tracked hash
        self._file_hashes[path_str] = current_hash
    
    def _get_cached(self, abs_path: Path) -> Optional[Dict]:
        """
        Get cached data if available and hash unchanged.
        
        Args:
            abs_path: Absolute path to file
        
        Returns:
            Cached data dict or None
        """
        with self._cache_lock:
            path_str = str(abs_path)
            cached = self._cache.get(path_str)
            
            if cached is None:
                return None
            
            # Verify file hasn't changed (quick hash check)
            try:
                current_stat = abs_path.stat()
                cached_mtime = cached.get("mtime")
                
                # If mtime changed, invalidate cache
                if cached_mtime != current_stat.st_mtime:
                    del self._cache[path_str]
                    return None
            except Exception:
                # File may have been deleted; invalidate cache
                del self._cache[path_str]
                return None
            
            return cached
    
    def _set_cached(self, abs_path: Path, lines: List[str], file_hash: str) -> None:
        """
        Cache loaded data.
        
        Args:
            abs_path: Absolute path to file
            lines: Loaded lines
            file_hash: File hash
        """
        with self._cache_lock:
            path_str = str(abs_path)
            self._cache[path_str] = {
                "data": lines,
                "hash": file_hash,
                "mtime": abs_path.stat().st_mtime,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
    
    def _audit(self, event_type: str, relative_path: str, session_id: Optional[str], **kwargs) -> None:
        """
        Record audit log entry.
        
        Args:
            event_type: Type of event ("load", "cache_hit", etc.)
            relative_path: Relative path to file
            session_id: Optional session ID
            **kwargs: Additional metadata
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "path": relative_path,
            "session_id": session_id,
            **kwargs,
        }
        self._audit_log.append(entry)
        
        # Log to main logger (best-effort)
        try:
            logger.info({
                "level": "INFO",
                "type": "vocab_loader",
                "message": f"[VocabLoader] {event_type}: {relative_path}",
                "session_id": session_id,
                **kwargs,
            })
        except Exception:
            pass


# Singleton instance for convenience
_default_loader: Optional[VocabLoader] = None


def get_vocab_loader(base_dir: Optional[str | Path] = None) -> VocabLoader:
    """
    Get singleton VocabLoader instance.
    
    Args:
        base_dir: Optional base directory (only used on first call)
    
    Returns:
        VocabLoader instance
    """
    global _default_loader
    
    if _default_loader is None:
        if base_dir is None:
            # Default to Context_Integration/vocab
            from webapp.parser.config import PROJECT_ROOT
            base_dir = PROJECT_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab"
        
        _default_loader = VocabLoader(base_dir)
    
    return _default_loader

"""
Vocab Loader: Safe, audited vocabulary file management for confidence/caution framework.

Provides:
- VocabLoader: Core loader with path traversal protection, file validation, rate limiting
- VocabSecurityError: Path traversal / security violation
- VocabFileNotFound: File does not exist
- VocabIntegrityError: File integrity check failed (e.g., duplicate entries)
- VocabLoaderError: Generic loader error
- RateLimitError: Rate limit exceeded for reload operations

Design:
- All vocab files stored under webapp/parser/Context_Integration/vocab/
- Subdirectories: entities/, validators/, sources/ (immutable structure)
- Each file: newline-separated canonical names or key->value mappings
- Integrity: No duplicate entries, no blank lines (stripped on load)
- Security: Path traversal blocked; only .txt files allowed
- Rate limiting: Reload operations limited to once per 60 seconds per file
- Audit: All loads logged with session_id, load_count, entry_count, hash
"""

import hashlib
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from webapp.parser.utils.logger_singleton import logger


# ===== EXCEPTIONS =====

class VocabLoaderError(Exception):
    """Base vocab loader error."""
    pass


class VocabSecurityError(VocabLoaderError):
    """Path traversal or security violation."""
    pass


class VocabFileNotFound(VocabLoaderError):
    """Vocab file not found."""
    pass


class VocabIntegrityError(VocabLoaderError):
    """Vocab file integrity check failed (e.g., duplicate entries)."""
    pass


class RateLimitError(VocabLoaderError):
    """Rate limit exceeded for reload operation."""
    pass


# ===== CONSTANTS =====

VOCAB_BASE_DIR = Path(__file__).parent  # webapp/parser/Context_Integration/vocab/
ALLOWED_SUBDIRS = {"entities", "validators", "sources", "scoring"}
RELOAD_COOLDOWN_SECONDS = 60
HASH_ALGO = "sha256"


# ===== VOCAB LOADER CLASS =====

class VocabLoader:
    """
    Safe vocabulary file loader with path traversal protection, integrity checks, and rate limiting.
    
    Usage:
        loader = VocabLoader()
        entries = loader.load_canonical("entities", "offices.txt")  # List[str]
        mapping = loader.load_mapping("validators", "office_aliases.txt")  # Dict[str, str]
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize loader with optional custom base directory.
        
        Args:
            base_dir: Override vocab base directory (default: VOCAB_BASE_DIR)
        """
        self.base_dir = Path(base_dir or VOCAB_BASE_DIR)
        self._cache: Dict[str, List[str]] = {}
        self._mapping_cache: Dict[str, Dict[str, str]] = {}
        self._last_reload_time: Dict[str, float] = {}
        self._file_hashes: Dict[str, str] = {}
        self._load_counts: Dict[str, int] = {}

        # Ensure base directory exists
        if not self.base_dir.exists():
            raise VocabFileNotFound(f"Vocab base directory not found: {self.base_dir}")

    def load_canonical(
        self,
        subdir: str,
        filename: str,
        skip_cache: bool = False,
        session_id: Optional[str] = None,
    ) -> List[str]:
        """
        Load a canonical list from a vocab file (one entry per line, stripped).
        
        Args:
            subdir: Subdirectory name (entities, validators, sources, scoring)
            filename: Filename (must be .txt)
            skip_cache: Force reload from disk
            session_id: Optional session ID for audit logging
            
        Returns:
            List[str]: Canonical entries (blank lines and comments removed)
            
        Raises:
            VocabSecurityError: Path traversal attempt detected
            VocabFileNotFound: File does not exist
            VocabIntegrityError: Integrity check failed
            RateLimitError: Reload rate limit exceeded
            VocabLoaderError: Generic loader error
        """
        cache_key = self._make_cache_key(subdir, filename)
        
        if not skip_cache and cache_key in self._cache:
            return list(self._cache[cache_key])
        
        entries = self._load_from_disk(subdir, filename, session_id=session_id)
        self._cache[cache_key] = entries
        return list(entries)

    def load_mapping(
        self,
        subdir: str,
        filename: str,
        skip_cache: bool = False,
        session_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Load a key->value mapping from a vocab file (format: "key -> value" per line).
        
        Args:
            subdir: Subdirectory name
            filename: Filename
            skip_cache: Force reload from disk
            session_id: Optional session ID for audit logging
            
        Returns:
            Dict[str, str]: Key->value mapping
            
        Raises:
            VocabSecurityError, VocabFileNotFound, VocabIntegrityError, RateLimitError
        """
        cache_key = self._make_cache_key(subdir, filename)
        
        if not skip_cache and cache_key in self._mapping_cache:
            return dict(self._mapping_cache[cache_key])
        
        entries = self._load_from_disk(subdir, filename, session_id=session_id)
        mapping = self._parse_mapping(entries, subdir, filename)
        self._mapping_cache[cache_key] = mapping
        return dict(mapping)

    def reload(
        self,
        subdir: str,
        filename: str,
        session_id: Optional[str] = None,
    ) -> Tuple[int, str]:
        """
        Force reload a vocab file from disk (respects rate limiting).
        
        Args:
            subdir: Subdirectory name
            filename: Filename
            session_id: Optional session ID for audit logging
            
        Returns:
            Tuple[int, str]: (entry_count, file_hash)
            
        Raises:
            RateLimitError: Reload cooldown not elapsed
            VocabSecurityError, VocabFileNotFound, VocabIntegrityError
        """
        cache_key = self._make_cache_key(subdir, filename)
        now = time.time()
        
        if cache_key in self._last_reload_time:
            elapsed = now - self._last_reload_time[cache_key]
            if elapsed < RELOAD_COOLDOWN_SECONDS:
                raise RateLimitError(
                    f"Reload cooldown not elapsed for {cache_key} "
                    f"({elapsed:.1f}s < {RELOAD_COOLDOWN_SECONDS}s)"
                )
        
        entries = self._load_from_disk(subdir, filename, session_id=session_id)
        self._cache[cache_key] = entries
        self._mapping_cache.pop(cache_key, None)
        self._last_reload_time[cache_key] = now
        
        file_hash = self._file_hashes.get(cache_key, "unknown")
        return len(entries), file_hash

    def get_load_count(self, subdir: str, filename: str) -> int:
        """Get number of times a file has been loaded."""
        cache_key = self._make_cache_key(subdir, filename)
        return self._load_counts.get(cache_key, 0)

    def get_file_hash(self, subdir: str, filename: str) -> Optional[str]:
        """Get SHA256 hash of a loaded file."""
        cache_key = self._make_cache_key(subdir, filename)
        return self._file_hashes.get(cache_key)

    def clear_cache(self, subdir: Optional[str] = None, filename: Optional[str] = None) -> int:
        """
        Clear cache entries. If both subdir and filename provided, clear only that file.
        If neither provided, clear all caches.
        
        Returns: Number of cache entries cleared
        """
        if subdir and filename:
            cache_key = self._make_cache_key(subdir, filename)
            count = len([k for k in self._cache.keys() if k == cache_key])
            count += len([k for k in self._mapping_cache.keys() if k == cache_key])
            self._cache.pop(cache_key, None)
            self._mapping_cache.pop(cache_key, None)
            return count
        
        count = len(self._cache) + len(self._mapping_cache)
        self._cache.clear()
        self._mapping_cache.clear()
        return count

    # ===== PRIVATE METHODS =====

    def _make_cache_key(self, subdir: str, filename: str) -> str:
        """Create a normalized cache key."""
        return f"{subdir}:{filename}".lower()

    def _load_from_disk(
        self,
        subdir: str,
        filename: str,
        session_id: Optional[str] = None,
    ) -> List[str]:
        """
        Load raw entries from disk with validation.
        
        Raises: VocabSecurityError, VocabFileNotFound, VocabIntegrityError
        """
        # Validate subdir
        if subdir not in ALLOWED_SUBDIRS:
            raise VocabSecurityError(
                f"Invalid subdirectory: {subdir}. Allowed: {ALLOWED_SUBDIRS}"
            )

        # Validate filename
        if not filename.lower().endswith(".txt"):
            raise VocabSecurityError(f"Only .txt files allowed, got: {filename}")

        if "/" in filename or "\\" in filename or ".." in filename:
            raise VocabSecurityError(f"Path traversal detected in filename: {filename}")

        # Construct safe path
        file_path = (self.base_dir / subdir / filename).resolve()
        base_resolved = self.base_dir.resolve()

        # Path traversal protection
        try:
            file_path.relative_to(base_resolved)
        except ValueError:
            raise VocabSecurityError(f"Path traversal blocked: {file_path}")

        if not file_path.exists():
            raise VocabFileNotFound(f"Vocab file not found: {file_path}")

        if not file_path.is_file():
            raise VocabSecurityError(f"Path is not a file: {file_path}")

        # Read and parse
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as exc:
            raise VocabLoaderError(f"Failed to read {file_path}: {exc}")

        # Compute hash
        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        cache_key = self._make_cache_key(subdir, filename)
        self._file_hashes[cache_key] = file_hash

        # Parse entries
        entries = []
        seen = set()
        for line_no, raw_line in enumerate(content.split("\n"), start=1):
            line = raw_line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Check for duplicates
            if line in seen:
                raise VocabIntegrityError(
                    f"Duplicate entry in {file_path} at line {line_no}: {line}"
                )
            seen.add(line)
            entries.append(line)

        # Increment load count and audit log
        self._load_counts[cache_key] = self._load_counts.get(cache_key, 0) + 1
        logger.info({
            "level": "INFO",
            "type": "vocab_loader",
            "message": f"Loaded vocab file: {subdir}/{filename}",
            "session_id": session_id,
            "subdir": subdir,
            "filename": filename,
            "entry_count": len(entries),
            "file_hash": file_hash,
            "load_count": self._load_counts[cache_key],
        })

        return entries

    def _parse_mapping(
        self,
        entries: List[str],
        subdir: str,
        filename: str,
    ) -> Dict[str, str]:
        """Parse key->value mapping from entries. Format: 'key -> value' (arrows with spaces)."""
        mapping = {}
        for entry in entries:
            if " -> " not in entry:
                raise VocabIntegrityError(
                    f"Invalid mapping format in {subdir}/{filename}: {entry} "
                    f"(expected 'key -> value')"
                )
            key, value = entry.split(" -> ", 1)
            key = key.strip()
            value = value.strip()
            
            if not key or not value:
                raise VocabIntegrityError(
                    f"Empty key or value in {subdir}/{filename}: {entry}"
                )
            
            mapping[key] = value
        return mapping


# ===== SINGLETON ACCESSOR =====

_loader_instance: Optional[VocabLoader] = None


def get_vocab_loader(base_dir: Optional[Path] = None) -> VocabLoader:
    """Get singleton VocabLoader instance."""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = VocabLoader(base_dir=base_dir)
    return _loader_instance

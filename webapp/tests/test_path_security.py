"""
Path Security Test Suite
Tests path traversal prevention and file operation security
"""
import os
import tempfile
from pathlib import Path

import pytest

from webapp.parser.utils.shared_logic import (
    is_path_safe,
    safe_filename,
    safe_join_path,
    safe_resolve_path,
    validate_directory_path,
)


class TestSafeFilename:
    """Test suite for safe_filename function"""

    def test_basic_sanitization(self):
        """Test basic character sanitization"""
        assert safe_filename("test file.txt") == "test_file.txt"
        assert safe_filename("file@#$%.txt") == "file_txt"  # Special chars replaced and collapsed
        
    def test_path_separator_removal(self):
        """Test that path separators are removed"""
        assert safe_filename("../../../etc/passwd") == "etcpasswd"
        assert "/" not in safe_filename("../../../etc/passwd")
        assert safe_filename("test/file.txt") == "testfile.txt"
        assert safe_filename("test\\file.txt") == "testfile.txt"
        
    def test_path_traversal_prevention(self):
        """Test that path traversal attempts are blocked"""
        assert safe_filename("..") == "file"
        assert safe_filename("../") == "file"
        assert safe_filename("../../") == "file"
        assert safe_filename("....//") == "file"
        
    def test_null_byte_removal(self):
        """Test that null bytes are removed"""
        assert safe_filename("test\x00file.txt") == "testfile.txt"
        assert safe_filename("\x00malicious") == "malicious"
        
    def test_reserved_names(self):
        """Test Windows reserved device names are handled"""
        assert safe_filename("CON") == "_CON_"
        assert safe_filename("PRN") == "_PRN_"
        assert safe_filename("COM1") == "_COM1_"
        assert safe_filename("LPT1") == "_LPT1_"
        
    def test_empty_input(self):
        """Test empty input returns default"""
        assert safe_filename("") == "file"
        assert safe_filename("   ") == "file"
        assert safe_filename("...", default="empty") == "empty"
        
    def test_length_limit(self):
        """Test max length enforcement"""
        long_name = "a" * 300
        result = safe_filename(long_name, max_length=255)
        assert len(result) <= 255
        
    def test_strict_mode(self):
        """Test strict mode enforces stricter rules"""
        # With strict_mode=True, traversal dots are stripped
        result = safe_filename("test/../file.txt", strict_mode=True)
        assert ".." not in result
        assert result == "test_file.txt"

    def test_unicode_handling(self):
        """Test Unicode character handling"""
        # With allow_unicode=False (default)
        assert safe_filename("test.txt") == "test.txt"
        # With allow_unicode=True  
        assert safe_filename("test.txt", allow_unicode=True) == "test.txt"


class TestSafeResolvePath:
    """Test suite for safe_resolve_path function"""

    def test_normal_path_resolution(self, tmp_path):
        """Test normal path resolution within base"""
        base = tmp_path
        target = base / "subdir" / "file.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
        
        resolved = safe_resolve_path(target, base)
        assert resolved == target
        assert resolved.is_relative_to(base)
        
    def test_relative_path_resolution(self, tmp_path):
        """Test relative path resolution"""
        base = tmp_path
        (base / "subdir").mkdir()
        
        resolved = safe_resolve_path("subdir", base)
        assert resolved == base / "subdir"
        
    def test_path_traversal_blocked(self, tmp_path):
        """Test that path traversal is blocked"""
        base = tmp_path / "safe"
        base.mkdir()
        
        # Try to escape via ../
        with pytest.raises(ValueError, match="Path traversal detected"):
            safe_resolve_path("../../../etc/passwd", base)
            
    def test_absolute_path_outside_base(self, tmp_path):
        """Test that absolute paths outside base are blocked"""
        base = tmp_path / "safe"
        base.mkdir()
        
        outside = tmp_path / "unsafe"
        outside.mkdir()
        
        with pytest.raises(ValueError, match="Path traversal detected"):
            safe_resolve_path(outside, base)
            
    def test_symbolic_link_escape(self, tmp_path):
        """Test that symbolic links cannot escape base"""
        if os.name == "nt":
            pytest.skip("Symbolic link test not applicable on Windows")
            
        base = tmp_path / "safe"
        base.mkdir()
        outside = tmp_path / "unsafe"
        outside.mkdir()
        
        link = base / "escape"
        link.symlink_to(outside)
        
        # Should still be caught after resolution
        with pytest.raises(ValueError, match="Path traversal detected"):
            safe_resolve_path(link, base)
            
    def test_must_exist_flag(self, tmp_path):
        """Test must_exist flag"""
        base = tmp_path
        
        # Non-existent path with must_exist=True should fail
        with pytest.raises(ValueError, match="does not exist"):
            safe_resolve_path("nonexistent", base, must_exist=True)
            
        # Existent path with must_exist=True should pass
        existing = base / "exists"
        existing.touch()
        resolved = safe_resolve_path("exists", base, must_exist=True)
        assert resolved == existing


class TestIsPathSafe:
    """Test suite for is_path_safe function"""

    def test_path_within_allowed(self, tmp_path):
        """Test path within allowed directory"""
        base = tmp_path
        safe_path = base / "subdir" / "file.txt"
        
        assert is_path_safe(safe_path, [base]) == True
        
    def test_path_outside_allowed(self, tmp_path):
        """Test path outside allowed directories"""
        base1 = tmp_path / "allowed1"
        base2 = tmp_path / "allowed2"
        outside = tmp_path / "forbidden"
        
        for d in [base1, base2, outside]:
            d.mkdir(exist_ok=True)
            
        assert is_path_safe(outside, [base1, base2]) == False
        
    def test_multiple_allowed_dirs(self, tmp_path):
        """Test with multiple allowed directories"""
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        
        for d in [dir1, dir2]:
            d.mkdir(exist_ok=True)
            
        path1 = dir1 / "file.txt"
        path2 = dir2 / "file.txt"
        
        assert is_path_safe(path1, [dir1, dir2]) == True
        assert is_path_safe(path2, [dir1, dir2]) == True
        
    def test_traversal_attempt(self, tmp_path):
        """Test that traversal attempts are caught"""
        base = tmp_path / "safe"
        base.mkdir()
        
        # Even with traversal in the string, is_path_safe should catch it
        traversal_path = base / ".." / ".." / "etc" / "passwd"
        assert is_path_safe(traversal_path, [base]) == False


class TestSafeJoinPath:
    """Test suite for safe_join_path function"""

    def test_normal_join(self, tmp_path):
        """Test normal path joining"""
        base = tmp_path
        result = safe_join_path(base, "subdir", "file.txt")
        
        assert result == base / "subdir" / "file.txt"
        assert result.is_relative_to(base)
        
    def test_sanitization_applied(self, tmp_path):
        """Test that sanitization is applied to components"""
        base = tmp_path
        # Unsafe component should be sanitized
        result = safe_join_path(base, "sub/dir", "file.txt")
        
        # Path separators should be removed from components
        assert "sub/dir" not in str(result)
        
    def test_traversal_blocked(self, tmp_path):
        """Test that traversal is blocked"""
        base = tmp_path
        
        # Components with .. get sanitized; ensure final path stays inside base
        result = safe_join_path(base, "..", "..", "etc", "passwd")
        assert result.is_relative_to(base)
        assert ".." not in str(result)
            
    def test_empty_components(self, tmp_path):
        """Test handling of empty components"""
        base = tmp_path
        result = safe_join_path(base, "", "subdir", "")
        # Empty components are ignored; subdir remains
        assert result == base / "subdir"


class TestValidateDirectoryPath:
    """Test suite for validate_directory_path function"""

    def test_existing_directory(self, tmp_path):
        """Test validation of existing directory"""
        directory = tmp_path / "existing"
        directory.mkdir()
        
        validated = validate_directory_path(directory)
        assert validated == directory.resolve()
        
    def test_nonexistent_without_create(self, tmp_path):
        """Test non-existent directory without create flag"""
        directory = tmp_path / "nonexistent"
        
        with pytest.raises(ValueError, match="does not exist"):
            validate_directory_path(directory, create_if_missing=False)
            
    def test_nonexistent_with_create(self, tmp_path):
        """Test non-existent directory with create flag"""
        directory = tmp_path / "new" / "nested" / "dir"
        
        validated = validate_directory_path(directory, create_if_missing=True)
        assert validated.exists()
        assert validated.is_dir()
        
    def test_file_not_directory(self, tmp_path):
        """Test that files are rejected"""
        file_path = tmp_path / "file.txt"
        file_path.touch()
        
        with pytest.raises(ValueError, match="not a directory"):
            validate_directory_path(file_path)


class TestPathTraversalAttacks:
    """Test suite for common path traversal attack vectors"""

    @pytest.fixture
    def safe_base(self, tmp_path):
        """Create a safe base directory for testing"""
        base = tmp_path / "safe"
        base.mkdir()
        return base

    def test_dot_dot_slash(self, safe_base):
        """Test ../attack"""
        with pytest.raises(ValueError):
            safe_resolve_path("../../../etc/passwd", safe_base)
            
    def test_dot_dot_backslash(self, safe_base):
        """Test ..\\attack"""
        with pytest.raises(ValueError):
            safe_resolve_path("..\\..\\..\\windows\\system32", safe_base)
            
    def test_url_encoded_traversal(self, safe_base):
        """Test URL-encoded traversal"""
        # URL decode happens before our validation
        decoded = "..%2F..%2F..%2Fetc%2Fpasswd"
        # After URL decode: "../../../etc/passwd"
        with pytest.raises(ValueError):
            safe_resolve_path(decoded.replace("%2F", "/"), safe_base)
            
    def test_double_encoded_traversal(self, safe_base):
        """Test double-encoded traversal"""
        # Literal %252F sequences are treated as characters; ensure path stays within base
        result = safe_resolve_path("..%252F..%252F..%252Fetc", safe_base)
        assert result.is_relative_to(safe_base)
            
    def test_unicode_traversal(self, safe_base):
        """Test Unicode traversal attempts"""
        # Unicode variants of ..
        with pytest.raises(ValueError):
            safe_resolve_path("\u002e\u002e/\u002e\u002e/etc", safe_base)
            
    def test_null_byte_injection(self, safe_base):
        """Test null byte injection"""
        # Null bytes should be stripped by safe_filename
        filename = safe_filename("safe.txt\x00../../etc/passwd")
        assert "\x00" not in filename
        assert "../" not in filename
        
    def test_mixed_separators(self, safe_base):
        """Test mixed path separators"""
        with pytest.raises(ValueError):
            safe_resolve_path("..\\..//etc/passwd", safe_base)


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios"""

    def test_output_path_construction(self, tmp_path):
        """Test safe output path construction"""
        output_base = tmp_path / "outputs"
        output_base.mkdir()
        
        # User-provided contest name (potentially malicious)
        contest = "../../../etc/passwd"
        state = "../../windows/system32"
        
        # Sanitize components
        safe_contest = safe_filename(contest, strict_mode=True)
        safe_state = safe_filename(state, strict_mode=True)
        
        # Build path safely
        output_path = safe_join_path(output_base, safe_state, safe_contest)
        
        # Verify it's safe
        assert is_path_safe(output_path, [output_base])
        assert "../" not in str(output_path)
        assert "../../" not in str(output_path)
        
    def test_cache_path_construction(self, tmp_path):
        """Test safe cache path construction"""
        cache_base = tmp_path / "cache"
        cache_base.mkdir()
        
        # User-provided cache key (potentially malicious)
        cache_key = "../../tmp/malicious.cache"
        
        # Sanitize
        safe_key = safe_filename(cache_key, strict_mode=True)
        
        # Build path
        cache_path = safe_join_path(cache_base, safe_key)
        
        # Verify safe
        assert is_path_safe(cache_path, [cache_base])
        
    def test_log_path_construction(self, tmp_path):
        """Test safe log path construction"""
        log_base = tmp_path / "logs"
        log_base.mkdir()
        
        # Session ID from user (could be crafted)
        session_id = "sess_../../etc/passwd"
        
        # Sanitize
        safe_session = safe_filename(session_id, strict_mode=True)
        
        # Build log path
        log_path = safe_join_path(log_base, f"{safe_session}.log")
        
        # Verify safe
        assert is_path_safe(log_path, [log_base])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
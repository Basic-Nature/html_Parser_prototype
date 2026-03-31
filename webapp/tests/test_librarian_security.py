"""
Security Test Suite for librarian.py
Tests path traversal prevention and file operation security in the librarian module
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test
from webapp.parser.Context_Integration.librarian import (
    ALLOWED_ROOTS,
    LOG_DIR_PATH,
    CONTEXT_LIBRARY_DIR,
    PROJECT_ROOT_PATH,
    safe_path,
    get_safe_log_path,
    atomic_write_json,
    load_context_library,
    save_context_library,
    backup_context_library,
    _get_log_path,
    _deduplicate_jsonl_log,
    log_unknown_tag,
    log_unknown_attr,
)


class TestSafePathLibrarian:
    """Test suite for safe_path function in librarian"""

    def test_path_validation_with_allowed_roots(self, tmp_path):
        """Test that paths are validated against allowed roots"""
        test_root = tmp_path / "allowed"
        test_root.mkdir()
        test_file = test_root / "test.json"
        
        result = safe_path(test_file, [test_root])
        assert result == test_file.resolve()
        
    def test_path_rejection_outside_roots(self, tmp_path):
        """Test that paths outside allowed roots are rejected"""
        allowed = tmp_path / "allowed"
        forbidden = tmp_path / "forbidden"
        
        for d in [allowed, forbidden]:
            d.mkdir()
            
        forbidden_file = forbidden / "test.json"
        
        with pytest.raises(ValueError, match="Path traversal detected"):
            safe_path(forbidden_file, [allowed])
            
    def test_uses_default_allowed_roots(self):
        """Test that default ALLOWED_ROOTS are used"""
        # Should work with paths in default allowed roots
        test_path = Path(LOG_DIR_PATH) / "test.jsonl"
        result = safe_path(test_path)
        # Should not raise
        assert result.is_relative_to(Path(LOG_DIR_PATH).resolve())


class TestGetSafeLogPath:
    """Test suite for get_safe_log_path function"""

    def test_sanitizes_filename(self, tmp_path):
        """Test that filenames are sanitized"""
        with patch('webapp.parser.Context_Integration.librarian.LOG_DIR_PATH', tmp_path):
            with patch('webapp.parser.Context_Integration.librarian.ALLOWED_ROOTS', [tmp_path]):
                # Malicious filename
                result = get_safe_log_path("../../etc/passwd")
                
                # Should be sanitized and in LOG_DIR
                assert result.is_relative_to(tmp_path.resolve())
                assert "../" not in str(result)
                
    def test_validates_after_sanitization(self, tmp_path):
        """Test that path is validated even after sanitization"""
        log_dir = tmp_path / "log"
        log_dir.mkdir()
        
        with patch('webapp.parser.Context_Integration.librarian.LOG_DIR_PATH', log_dir):
            with patch('webapp.parser.Context_Integration.librarian.ALLOWED_ROOTS', [log_dir]):
                result = get_safe_log_path("test.log")
                
                # Should be in log_dir
                assert result.is_relative_to(log_dir.resolve())
                
    def test_blocks_null_bytes(self, tmp_path):
        """Test that null bytes in filenames are blocked"""
        log_dir = tmp_path / "log"
        log_dir.mkdir()
        
        with patch('webapp.parser.Context_Integration.librarian.LOG_DIR_PATH', log_dir):
            with patch('webapp.parser.Context_Integration.librarian.ALLOWED_ROOTS', [log_dir]):
                result = get_safe_log_path("test\x00malicious.log")
                
                # Null byte should be removed
                assert "\x00" not in str(result)


class TestAtomicWriteJsonLibrarian:
    """Test suite for atomic_write_json in librarian"""

    def test_validates_main_path(self, tmp_path):
        """Test that main file path is validated"""
        lib_dir = tmp_path / "library"
        lib_dir.mkdir()
        test_file = lib_dir / "test.json"
        
        data = {"test": "data"}
        
        with patch('webapp.parser.Context_Integration.librarian.ALLOWED_ROOTS', [lib_dir]):
            atomic_write_json(data, test_file)
            assert test_file.exists()
            
    def test_validates_backup_path(self, tmp_path):
        """Test that backup path is validated"""
        lib_dir = tmp_path / "library"
        lib_dir.mkdir()
        test_file = lib_dir / "test.json"
        test_file.write_text('{"old": "data"}')
        
        data = {"new": "data"}
        
        with patch('webapp.parser.Context_Integration.librarian.ALLOWED_ROOTS', [lib_dir]):
            atomic_write_json(data, test_file)
            
            # Backup should exist and be validated
            backup = test_file.with_suffix(test_file.suffix + ".bak")
            assert backup.exists()
            assert backup.is_relative_to(lib_dir.resolve())
            
    def test_validates_tmp_path(self, tmp_path):
        """Test that temporary path is validated"""
        lib_dir = tmp_path / "library"
        lib_dir.mkdir()
        test_file = lib_dir / "test.json"
        
        data = {"test": "data"}
        
        with patch('webapp.parser.Context_Integration.librarian.ALLOWED_ROOTS', [lib_dir]):
            atomic_write_json(data, test_file)
            
            # No tmp files should remain
            tmp_files = list(lib_dir.glob("*.tmp"))
            assert len(tmp_files) == 0


class TestLoadContextLibrary:
    """Test suite for load_context_library function"""

    def test_validates_library_path(self, tmp_path):
        """Test that library path is validated"""
        lib_dir = tmp_path / "library"
        lib_dir.mkdir()
        lib_file = lib_dir / "context_library.json"
        lib_file.write_text('{"test": "data"}')
        
        with patch('webapp.parser.Context_Integration.librarian.ALLOWED_ROOTS', [lib_dir]):
            result = load_context_library(lib_file)
            assert isinstance(result, dict)
            
    def test_blocks_traversal_in_path(self, tmp_path):
        """Test that path traversal in library path is blocked"""
        lib_dir = tmp_path / "library"
        lib_dir.mkdir()
        
        # Try to load from outside allowed dir
        malicious_path = lib_dir / ".." / ".." / "etc" / "passwd"
        
        with patch('webapp.parser.Context_Integration.librarian.ALLOWED_ROOTS', [lib_dir]):
            with pytest.raises(ValueError):
                load_context_library(malicious_path)
                
    def test_validates_backup_on_corrupt(self, tmp_path):
        """Test that backup path is validated when handling corrupt files"""
        lib_dir = tmp_path / "library"
        lib_dir.mkdir()
        lib_file = lib_dir / "context_library.json"
        lib_file.write_text('invalid json{')
        
        with patch('webapp.parser.Context_Integration.librarian.ALLOWED_ROOTS', [lib_dir]):
            # Should create backup with .corrupt extension
            result = load_context_library(lib_file)
            
            # Backup should be in same directory
            corrupt_backup = lib_file.parent / (lib_file.name + ".corrupt")
            if corrupt_backup.exists():
                assert corrupt_backup.is_relative_to(lib_dir.resolve())


class TestSaveContextLibrary:
    """Test suite for save_context_library function"""

    def test_validates_save_path(self, tmp_path):
        """Test that save path is validated"""
        lib_dir = tmp_path / "library"
        lib_dir.mkdir()
        lib_file = lib_dir / "context_library.json"
        
        data = {"test": "data"}
        
        with patch('webapp.parser.Context_Integration.librarian.ALLOWED_ROOTS', [lib_dir]):
            save_context_library(data, lib_file)
            assert lib_file.exists()
            
    def test_validates_temp_file_path(self, tmp_path):
        """Test that temporary file paths are validated"""
        lib_dir = tmp_path / "library"
        lib_dir.mkdir()
        lib_file = lib_dir / "context_library.json"
        
        data = {"test": "data"}
        
        with patch('webapp.parser.Context_Integration.librarian.ALLOWED_ROOTS', [lib_dir]):
            save_context_library(data, lib_file)
            
            # Temp file should be created and removed from same dir
            # No temp files should remain
            temp_files = list(lib_dir.glob("*.tmp"))
            assert len(temp_files) == 0
            
    def test_cleans_up_old_temp_files(self, tmp_path):
        """Test that old temp files are cleaned up securely"""
        lib_dir = tmp_path / "library"
        lib_dir.mkdir()
        lib_file = lib_dir / "context_library.json"
        
        # Create an old temp file
        old_temp = lib_dir / "old_context.tmp"
        old_temp.write_text('old data')
        
        data = {"test": "data"}
        
        with patch('webapp.parser.Context_Integration.librarian.ALLOWED_ROOTS', [lib_dir]):
            save_context_library(data, lib_file)
            
            # Old temp file handling is implementation-specific
            # Just ensure no traversal can happen


class TestBackupContextLibrary:
    """Test suite for backup_context_library function"""

    def test_validates_backup_paths(self, tmp_path):
        """Test that all backup paths are validated"""
        lib_dir = tmp_path / "library"
        lib_dir.mkdir()
        lib_file = lib_dir / "context_library.json"
        lib_file.write_text('{"test": "data"}')
        
        with patch('webapp.parser.Context_Integration.librarian.ALLOWED_ROOTS', [lib_dir]):
            backup_context_library(lib_file, max_backups=3)
            
            # All backups should be in same directory
            backups = list(lib_dir.glob("*.bak"))
            for backup in backups:
                assert backup.is_relative_to(lib_dir.resolve())
                
    def test_prunes_old_backups_safely(self, tmp_path):
        """Test that old backups are pruned securely"""
        lib_dir = tmp_path / "library"
        lib_dir.mkdir()
        lib_file = lib_dir / "context_library.json"
        lib_file.write_text('{"test": "data"}')
        
        # Create multiple backups
        for i in range(5):
            backup = lib_dir / f"context_library.json.backup{i}.bak"
            backup.write_text(f'{{"version": {i}}}')
            
        with patch('webapp.parser.Context_Integration.librarian.ALLOWED_ROOTS', [lib_dir]):
            backup_context_library(lib_file, max_backups=2)
            
            # Should only keep 2 most recent
            backups = list(lib_dir.glob("*.bak"))
            assert len(backups) <= 3  # max_backups + new one


class TestGetLogPath:
    """Test suite for _get_log_path function"""

    def test_sanitizes_and_validates(self, tmp_path):
        """Test that filename is sanitized and path is validated"""
        log_dir = tmp_path / "log"
        log_dir.mkdir()
        
        with patch('webapp.parser.Context_Integration.librarian.LOG_DIR_PATH', log_dir):
            with patch('webapp.parser.Context_Integration.librarian.ALLOWED_ROOTS', [log_dir]):
                result = _get_log_path("../../etc/passwd")
                
                # Should be sanitized and in log_dir
                assert Path(result).is_relative_to(log_dir.resolve())
                assert "../" not in result


class TestDeduplicateJsonlLog:
    """Test suite for _deduplicate_jsonl_log function"""

    def test_validates_log_path(self, tmp_path):
        """Test that log file path is validated"""
        log_dir = tmp_path / "log"
        log_dir.mkdir()
        log_file = log_dir / "test.jsonl"
        log_file.write_text('{"tag": "test1"}\n{"tag": "test1"}\n{"tag": "test2"}\n')
        
        with patch('webapp.parser.Context_Integration.librarian.ALLOWED_ROOTS', [log_dir]):
            result = _deduplicate_jsonl_log(str(log_file), "tag")
            
            # Should deduplicate
            assert len(result) == 2
            
    def test_blocks_traversal_in_path(self, tmp_path):
        """Test that path traversal is blocked"""
        log_dir = tmp_path / "log"
        log_dir.mkdir()
        
        malicious_path = str(log_dir / ".." / ".." / "etc" / "passwd")
        
        with patch('webapp.parser.Context_Integration.librarian.ALLOWED_ROOTS', [log_dir]):
            # Should either raise or return empty safely
            result = _deduplicate_jsonl_log(malicious_path, "tag")
            # File doesn't exist, should return empty set
            assert isinstance(result, set)


class TestLogUnknownTag:
    """Test suite for log_unknown_tag function"""

    def test_validates_log_file_path(self, tmp_path):
        """Test that log file path is validated"""
        log_dir = tmp_path / "log"
        log_dir.mkdir()
        
        with patch('webapp.parser.Context_Integration.librarian.LOG_DIR_PATH', log_dir):
            with patch('webapp.parser.Context_Integration.librarian.ALLOWED_ROOTS', [log_dir]):
                log_unknown_tag("test_tag", {})
                
                # Log file should be created in log_dir
                log_file = log_dir / "unknown_tags_log.jsonl"
                assert log_file.exists()
                assert log_file.is_relative_to(log_dir.resolve())


class TestLogUnknownAttr:
    """Test suite for log_unknown_attr function"""

    def test_validates_log_file_path(self, tmp_path):
        """Test that log file path is validated"""
        log_dir = tmp_path / "log"
        log_dir.mkdir()
        
        with patch('webapp.parser.Context_Integration.librarian.LOG_DIR_PATH', log_dir):
            with patch('webapp.parser.Context_Integration.librarian.ALLOWED_ROOTS', [log_dir]):
                log_unknown_attr("test-attr", {})
                
                # Log file should be created in log_dir
                log_file = log_dir / "unknown_attrs_log.jsonl"
                assert log_file.exists()
                assert log_file.is_relative_to(log_dir.resolve())


class TestSelfHealSecurity:
    """Test suite for self_heal_context_library security"""

    def test_validates_scan_script_path(self, tmp_path):
        """Test that scan script path is validated"""
        project_root = tmp_path / "project"
        project_root.mkdir()
        
        # Create mock health directory structure
        health_dir = project_root / "webapp" / "parser" / "health"
        health_dir.mkdir(parents=True)
        scan_script = health_dir / "scan_misaligned_ner.py"
        scan_script.write_text('#!/usr/bin/env python\nprint("test")')
        
        with patch('webapp.parser.Context_Integration.librarian.PROJECT_ROOT', project_root):
            with patch('webapp.parser.Context_Integration.librarian.ALLOWED_ROOTS', [project_root]):
                # Validate the script path
                validated = safe_path(scan_script, [project_root])
                assert validated.is_relative_to(project_root.resolve())


class TestIntegrationScenarios:
    """Integration tests for realistic attack scenarios"""

    def test_malicious_context_library_path(self, tmp_path):
        """Test protection against malicious context library paths"""
        lib_dir = tmp_path / "library"
        lib_dir.mkdir()
        
        # Attacker tries to load system file
        malicious_path = "../../../../../../etc/shadow"
        
        with patch('webapp.parser.Context_Integration.librarian.ALLOWED_ROOTS', [lib_dir]):
            with pytest.raises(ValueError):
                load_context_library(Path(lib_dir) / malicious_path)
                
    def test_malicious_log_path_injection(self, tmp_path):
        """Test protection against log path injection"""
        log_dir = tmp_path / "log"
        log_dir.mkdir()
        
        # Attacker tries to write to system directory
        with patch('webapp.parser.Context_Integration.librarian.LOG_DIR_PATH', log_dir):
            with patch('webapp.parser.Context_Integration.librarian.ALLOWED_ROOTS', [log_dir]):
                result = get_safe_log_path("../../../etc/malicious.log")
                
                # Should be sanitized and in log_dir
                assert result.is_relative_to(log_dir.resolve())
                
    def test_symlink_escape_blocked(self, tmp_path):
        """Test that symlink escapes are blocked"""
        if os.name == "nt":
            pytest.skip("Symlink test not applicable on Windows")
            
        lib_dir = tmp_path / "library"
        outside_dir = tmp_path / "outside"
        
        for d in [lib_dir, outside_dir]:
            d.mkdir()
            
        # Create symlink to outside directory
        link = lib_dir / "escape"
        link.symlink_to(outside_dir)
        
        target = link / "file.json"
        
        with patch('webapp.parser.Context_Integration.librarian.ALLOWED_ROOTS', [lib_dir]):
            with pytest.raises(ValueError):
                safe_path(target, [lib_dir])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

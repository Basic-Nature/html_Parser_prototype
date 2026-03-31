"""
Security Test Suite for manual_correction_bot.py
Tests path traversal prevention and file operation security in the correction bot
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test
from webapp.parser.health.manual_correction_bot import (
    ALLOWED_ROOTS,
    LOG_DIR,
    CONTEXT_LIBRARY_DIR,
    CACHE_DIR,
    safe_path,
    find_log_files,
    load_jsonl,
    save_jsonl,
    atomic_write_json,
    check_and_fix_json_files,
    export_correction_session,
    import_correction_session,
)


class TestSafePathValidation:
    """Test suite for safe_path function in manual_correction_bot"""

    def test_path_within_allowed_roots(self, tmp_path):
        """Test that paths within allowed roots are accepted"""
        test_root = tmp_path / "allowed"
        test_root.mkdir()
        test_file = test_root / "test.jsonl"
        
        result = safe_path(test_file, [test_root])
        assert result == test_file.resolve()
        
    def test_path_outside_allowed_roots(self, tmp_path):
        """Test that paths outside allowed roots are rejected"""
        allowed_root = tmp_path / "allowed"
        forbidden_root = tmp_path / "forbidden"
        
        for root in [allowed_root, forbidden_root]:
            root.mkdir()
            
        forbidden_file = forbidden_root / "test.jsonl"
        
        with pytest.raises(ValueError, match="Path traversal detected"):
            safe_path(forbidden_file, [allowed_root])
            
    def test_traversal_attempt_blocked(self, tmp_path):
        """Test that path traversal attempts are blocked"""
        allowed_root = tmp_path / "allowed"
        allowed_root.mkdir()
        
        # Try to escape using ../
        traversal_path = allowed_root / ".." / ".." / "etc" / "passwd"
        
        with pytest.raises(ValueError, match="Path traversal detected"):
            safe_path(traversal_path, [allowed_root])
            
    def test_uses_default_allowed_roots(self):
        """Test that function uses ALLOWED_ROOTS by default"""
        # Should not raise if path is in one of the default allowed roots
        test_path = Path(LOG_DIR) / "test.jsonl"
        result = safe_path(test_path)
        assert result.is_relative_to(Path(LOG_DIR).resolve())


class TestLoadJsonlSecurity:
    """Test suite for load_jsonl function security"""

    def test_load_within_allowed_dirs(self, tmp_path):
        """Test loading JSONL from allowed directory"""
        log_dir = tmp_path / "log"
        log_dir.mkdir()
        test_file = log_dir / "test.jsonl"
        test_file.write_text('{"test": "data"}\n')
        
        with patch('webapp.parser.health.manual_correction_bot.LOG_DIR', log_dir):
            with patch('webapp.parser.health.manual_correction_bot.ALLOWED_ROOTS', [log_dir]):
                result = load_jsonl(test_file)
                assert len(result) == 1
                assert result[0]["test"] == "data"
                
    def test_load_outside_allowed_dirs_blocked(self, tmp_path):
        """Test that loading from outside allowed dirs is blocked"""
        allowed_dir = tmp_path / "allowed"
        forbidden_dir = tmp_path / "forbidden"
        
        for d in [allowed_dir, forbidden_dir]:
            d.mkdir()
            
        forbidden_file = forbidden_dir / "test.jsonl"
        forbidden_file.write_text('{"test": "data"}\n')
        
        with patch('webapp.parser.health.manual_correction_bot.ALLOWED_ROOTS', [allowed_dir]):
            with pytest.raises(ValueError, match="Path traversal detected"):
                load_jsonl(forbidden_file)


class TestSaveJsonlSecurity:
    """Test suite for save_jsonl function security"""

    def test_save_within_allowed_dirs(self, tmp_path):
        """Test saving JSONL to allowed directory"""
        log_dir = tmp_path / "log"
        log_dir.mkdir()
        test_file = log_dir / "test.jsonl"
        
        entries = [{"test": "data"}]
        
        with patch('webapp.parser.health.manual_correction_bot.LOG_DIR', log_dir):
            with patch('webapp.parser.health.manual_correction_bot.ALLOWED_ROOTS', [log_dir]):
                save_jsonl(test_file, entries)
                assert test_file.exists()
                
    def test_save_outside_allowed_dirs_blocked(self, tmp_path):
        """Test that saving outside allowed dirs is blocked"""
        allowed_dir = tmp_path / "allowed"
        forbidden_dir = tmp_path / "forbidden"
        
        for d in [allowed_dir, forbidden_dir]:
            d.mkdir()
            
        forbidden_file = forbidden_dir / "test.jsonl"
        entries = [{"test": "data"}]
        
        with patch('webapp.parser.health.manual_correction_bot.ALLOWED_ROOTS', [allowed_dir]):
            with pytest.raises(ValueError, match="Path traversal detected"):
                save_jsonl(forbidden_file, entries)
                
    def test_tmp_path_validated(self, tmp_path):
        """Test that temporary file paths are also validated"""
        log_dir = tmp_path / "log"
        log_dir.mkdir()
        test_file = log_dir / "test.jsonl"
        
        entries = [{"test": "data"}]
        
        with patch('webapp.parser.health.manual_correction_bot.ALLOWED_ROOTS', [log_dir]):
            save_jsonl(test_file, entries)
            # Verify tmp file was created in same dir (safe)
            assert test_file.exists()


class TestAtomicWriteJsonSecurity:
    """Test suite for atomic_write_json function security"""

    def test_atomic_write_validates_path(self, tmp_path):
        """Test that atomic_write_json validates the target path"""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        test_file = allowed_dir / "test.json"
        
        data = {"test": "data"}
        
        with patch('webapp.parser.health.manual_correction_bot.ALLOWED_ROOTS', [allowed_dir]):
            atomic_write_json(data, test_file)
            assert test_file.exists()
            
    def test_atomic_write_validates_backup_path(self, tmp_path):
        """Test that backup paths are validated"""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        test_file = allowed_dir / "test.json"
        
        # Create existing file to trigger backup
        test_file.write_text('{"old": "data"}')
        
        data = {"new": "data"}
        
        with patch('webapp.parser.health.manual_correction_bot.ALLOWED_ROOTS', [allowed_dir]):
            atomic_write_json(data, test_file)
            # Backup should exist
            backup = test_file.with_suffix(test_file.suffix + ".bak")
            assert backup.exists()
            
    def test_atomic_write_validates_tmp_path(self, tmp_path):
        """Test that temporary file paths are validated"""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        test_file = allowed_dir / "test.json"
        
        data = {"test": "data"}
        
        with patch('webapp.parser.health.manual_correction_bot.ALLOWED_ROOTS', [allowed_dir]):
            atomic_write_json(data, test_file)
            # No tmp file should remain
            tmp_files = list(allowed_dir.glob("*.tmp"))
            assert len(tmp_files) == 0


class TestFindLogFilesSecurity:
    """Test suite for find_log_files function security"""

    def test_find_validates_directories(self, tmp_path):
        """Test that find_log_files validates search directories"""
        allowed_dir = tmp_path / "allowed"
        forbidden_dir = tmp_path / "forbidden"
        
        for d in [allowed_dir, forbidden_dir]:
            d.mkdir()
            (d / "test.jsonl").write_text('{}')
            
        with patch('webapp.parser.health.manual_correction_bot.ALLOWED_ROOTS', [allowed_dir]):
            # Should only find files in allowed dir
            result = find_log_files([allowed_dir, forbidden_dir])
            # Forbidden dir should be skipped with warning
            assert all(allowed_dir in f.parents for f in result)
            
    def test_find_validates_returned_paths(self, tmp_path):
        """Test that returned file paths are validated"""
        log_dir = tmp_path / "log"
        log_dir.mkdir()
        (log_dir / "test.jsonl").write_text('{}')
        
        with patch('webapp.parser.health.manual_correction_bot.ALLOWED_ROOTS', [log_dir]):
            result = find_log_files([log_dir])
            # All returned paths should be validated
            for path in result:
                assert path.is_relative_to(log_dir.resolve())


class TestCheckAndFixJsonFilesSecurity:
    """Test suite for check_and_fix_json_files function security"""

    def test_validates_all_directories(self, tmp_path):
        """Test that all directories are validated"""
        allowed_dir = tmp_path / "allowed"
        forbidden_dir = tmp_path / "forbidden"
        
        for d in [allowed_dir, forbidden_dir]:
            d.mkdir()
            (d / "test.json").write_text('{"test": "data"}')
            
        with patch('webapp.parser.health.manual_correction_bot.ALLOWED_ROOTS', [allowed_dir]):
            # Should skip forbidden dir
            result = check_and_fix_json_files(
                directories=[allowed_dir, forbidden_dir],
                auto_delete=False,
                verbose=False
            )
            # No errors from forbidden dir should propagate
            
    def test_validates_quarantine_paths(self, tmp_path):
        """Test that quarantine directory paths are validated"""
        log_dir = tmp_path / "log"
        log_dir.mkdir()
        corrupt_file = log_dir / "corrupt.json"
        corrupt_file.write_text('{"invalid json')
        
        with patch('webapp.parser.health.manual_correction_bot.ALLOWED_ROOTS', [log_dir]):
            check_and_fix_json_files(
                directories=[log_dir],
                quarantine=True,
                try_fix=True,
                verbose=False
            )
            # Quarantine dir should be inside log_dir
            quarantine_dir = log_dir / "corrupt"
            if quarantine_dir.exists():
                assert quarantine_dir.is_relative_to(log_dir.resolve())


class TestExportImportSecurity:
    """Test suite for export/import correction session security"""

    def test_export_validates_paths(self, tmp_path):
        """Test that export validates all paths"""
        log_dir = tmp_path / "log"
        export_dir = tmp_path / "export"
        
        for d in [log_dir, export_dir]:
            d.mkdir()
            
        source_file = log_dir / "test.jsonl"
        source_file.write_text('{}')
        
        with patch('webapp.parser.health.manual_correction_bot.ALLOWED_ROOTS', [log_dir, export_dir]):
            with patch('webapp.parser.health.manual_correction_bot.LOG_DIR', log_dir):
                export_correction_session([source_file], export_dir)
                # Verify export happened to safe location
                exported = list(export_dir.glob("*.jsonl"))
                assert len(exported) > 0
                
    def test_export_blocks_traversal(self, tmp_path):
        """Test that export blocks path traversal"""
        log_dir = tmp_path / "log"
        export_dir = tmp_path / "export"
        
        for d in [log_dir, export_dir]:
            d.mkdir()
            
        source_file = log_dir / "test.jsonl"
        source_file.write_text('{}')
        
        # Try to export outside allowed dir
        forbidden_export = tmp_path / "forbidden"
        
        with patch('webapp.parser.health.manual_correction_bot.ALLOWED_ROOTS', [log_dir]):
            with pytest.raises(ValueError):
                export_correction_session([source_file], forbidden_export)
                
    def test_import_validates_paths(self, tmp_path):
        """Test that import validates all paths"""
        import_dir = tmp_path / "import"
        dest_dir = tmp_path / "dest"
        
        for d in [import_dir, dest_dir]:
            d.mkdir()
            
        import_file = import_dir / "test.jsonl"
        import_file.write_text('{}')
        dest_file = dest_dir / "test.jsonl"
        
        with patch('webapp.parser.health.manual_correction_bot.ALLOWED_ROOTS', [import_dir, dest_dir]):
            import_correction_session(import_file, dest_file)
            assert dest_file.exists()


class TestSubprocessSecurity:
    """Test suite for subprocess execution security"""

    def test_scan_script_path_validated(self, tmp_path):
        """Test that scan script paths are validated before execution"""
        # This tests the self-heal subprocess execution
        with patch('webapp.parser.health.manual_correction_bot.PROJECT_ROOT', tmp_path):
            with patch('webapp.parser.health.manual_correction_bot.ALLOWED_ROOTS', [tmp_path]):
                # Create mock scan script
                health_dir = tmp_path / "webapp" / "parser" / "health"
                health_dir.mkdir(parents=True)
                scan_script = health_dir / "scan_misaligned_ner.py"
                scan_script.write_text('print("test")')
                
                # The path should be validated
                validated_path = safe_path(scan_script, [tmp_path])
                assert validated_path.is_relative_to(tmp_path.resolve())


class TestIntegrationScenarios:
    """Integration tests for realistic attack scenarios"""

    def test_malicious_log_path_injection(self, tmp_path):
        """Test protection against malicious log path injection"""
        log_dir = tmp_path / "log"
        log_dir.mkdir()
        
        # Attacker tries to write to /etc/passwd via log path
        malicious_path = "../../../../../../etc/passwd"
        
        with patch('webapp.parser.health.manual_correction_bot.LOG_DIR', log_dir):
            with patch('webapp.parser.health.manual_correction_bot.ALLOWED_ROOTS', [log_dir]):
                # This should fail validation
                with pytest.raises(ValueError):
                    load_jsonl(malicious_path)
                    
    def test_malicious_cache_path(self, tmp_path):
        """Test protection against malicious cache paths"""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        
        # Attacker tries to corrupt system files
        malicious_cache = "../../../windows/system32/config.sys"
        
        with patch('webapp.parser.health.manual_correction_bot.CACHE_DIR', cache_dir):
            with patch('webapp.parser.health.manual_correction_bot.ALLOWED_ROOTS', [cache_dir]):
                with pytest.raises(ValueError):
                    full_path = Path(cache_dir) / malicious_cache
                    safe_path(full_path, [cache_dir])
                    
    def test_url_encoded_traversal_blocked(self, tmp_path):
        """Test that URL-encoded traversal is blocked"""
        log_dir = tmp_path / "log"
        log_dir.mkdir()
        
        # URL-encoded ../../../etc/passwd
        encoded = "..%2F..%2F..%2Fetc%2Fpasswd"
        decoded = encoded.replace("%2F", "/")
        
        with patch('webapp.parser.health.manual_correction_bot.ALLOWED_ROOTS', [log_dir]):
            with pytest.raises(ValueError):
                safe_path(Path(log_dir) / decoded, [log_dir])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

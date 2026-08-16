"""
Comprehensive tests for LocalStorageSync verification framework.

Tests staging, promotion, deduplication, and audit trail functionality.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timezone

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from webapp.parser.verification.local_dl_sync import LocalStorageSync


class TestLocalStorageSync(unittest.TestCase):
    """Tests for LocalStorageSync synchronization engine."""

    def setUp(self):
        """Create temporary verification directory for each test."""
        self.temp_dir = tempfile.mkdtemp(prefix="test_sync_")
        self.verification_dir = os.path.join(self.temp_dir, "verification")
        os.makedirs(self.verification_dir, exist_ok=True)
        self.sync = LocalStorageSync(self.verification_dir)

    def tearDown(self):
        """Clean up temporary directory after each test."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_test_file(self, content: str = None, suffix: str = ".csv") -> str:
        """Create a temporary test file and return its path."""
        if content is None:
            content = "candidate,votes,percent\nAlice,1000,50%\nBob,1000,50%\n"

        fd, temp_path = tempfile.mkstemp(suffix=suffix, text=True)
        try:
            os.write(fd, content.encode('utf-8'))
        finally:
            os.close(fd)
        return temp_path

    def test_sync_available(self):
        """Test that sync correctly detects available directories."""
        self.assertTrue(self.sync.is_available())

        # Test with non-existent directory (but create_if_missing creates it)
        import tempfile
        non_existent_parent = tempfile.mkdtemp()
        bad_dir = os.path.join(non_existent_parent, "nonexistent", "nested", "path")
        sync_bad = LocalStorageSync(bad_dir, create_if_missing=False)
        self.assertFalse(sync_bad.is_available())
        import shutil
        shutil.rmtree(non_existent_parent, ignore_errors=True)

    def test_stage_dl2_file(self):
        """Test staging a file into DL2 (unverified)."""
        source_file = self._create_test_file()
        try:
            file_id = self.sync.stage_dl2_file(
                source_file,
                metadata={
                    "source_url": "https://example.org/results",
                    "state": "Virginia",
                    "county": "Arlington",
                }
            )

            self.assertIsNotNone(file_id)
            self.assertIsInstance(file_id, str)
            self.assertTrue(len(file_id) > 0)

            # Verify file exists in DL2
            dl2_dir = os.path.join(self.verification_dir, "dl2")
            self.assertTrue(os.path.isdir(dl2_dir))
            staged_file = os.path.join(dl2_dir, f"{file_id}.csv")
            self.assertTrue(os.path.isfile(staged_file))
        finally:
            os.unlink(source_file)

    def test_deduplication(self):
        """Test content-based deduplication detection."""
        content = "candidate,votes\nAlice,100\nBob,100\n"
        file1 = self._create_test_file(content)
        file2 = self._create_test_file(content + "\nExtra,line\n")  # Different content

        try:
            # Stage first file
            file_id1 = self.sync.stage_dl2_file(
                file1,
                metadata={"state": "Virginia", "county": "Arlington"}
            )

            # Stage second file with different content
            file_id2 = self.sync.stage_dl2_file(
                file2,
                metadata={"state": "Virginia", "county": "Fairfax"}
            )

            # Both should be different file IDs
            self.assertNotEqual(file_id1, file_id2)

            # Test find_duplicates with a known hash
            # Get hash of first file
            metadata = self.sync._load_metadata()
            hash_to_find = metadata.get("file_hashes", {}).get(file_id1, {}).get("hash")
            if hash_to_find:
                duplicates_list = self.sync.find_duplicates(hash_to_find)
                # Should find at least the first file
                self.assertIn(file_id1, duplicates_list)
        finally:
            os.unlink(file1)
            os.unlink(file2)

    def test_promote_to_dl1(self):
        """Test promotion from DL2 (unverified) to DL1 (verified)."""
        source_file = self._create_test_file()
        try:
            # Stage in DL2
            file_id = self.sync.stage_dl2_file(
                source_file,
                metadata={"state": "Virginia", "county": "Arlington"}
            )

            # Verify it's in DL2
            dl2_file = os.path.join(self.verification_dir, "dl2", f"{file_id}.csv")
            self.assertTrue(os.path.isfile(dl2_file))

            # Promote to DL1
            promotion_record = self.sync.promote_to_dl1(
                file_id,
                verifier_principal="analyst@election.gov",
                verification_notes="Verified against official results"
            )

            self.assertIsNotNone(promotion_record)
            self.assertEqual(promotion_record.get("file_id"), file_id)
            self.assertEqual(promotion_record.get("verifier_principal"), "analyst@election.gov")
            self.assertIn("promoted_at", promotion_record)  # API uses promoted_at not timestamp

            # Verify file is now in DL1
            dl1_file = os.path.join(self.verification_dir, "dl1", f"{file_id}.csv")
            self.assertTrue(os.path.isfile(dl1_file))

            # Verify promotion is logged
            history = self.sync.get_promotion_history(limit=1)
            self.assertTrue(len(history) > 0)
            last_promotion = history[0]
            self.assertEqual(last_promotion.get("file_id"), file_id)
        finally:
            os.unlink(source_file)

    def test_promotion_history(self):
        """Test that promotion history is properly tracked."""
        # Stage and promote multiple files
        file_ids = []
        source_files = []

        for i in range(3):
            content = f"candidate,votes\nCandidate{i},100\n"
            source_file = self._create_test_file(content)
            source_files.append(source_file)

            file_id = self.sync.stage_dl2_file(
                source_file,
                metadata={"state": "Virginia", "file": f"File{i}"}
            )
            file_ids.append(file_id)

            self.sync.promote_to_dl1(
                file_id,
                verifier_principal=f"analyst{i}@election.gov",
                verification_notes=f"Verified file {i}"
            )

        try:
            # Verify all promotions are logged
            history = self.sync.get_promotion_history(limit=10)
            self.assertEqual(len(history), 3)

            # Verify order (most recent first) and fields
            for i, record in enumerate(history):
                self.assertIn(record.get("file_id"), file_ids)
                self.assertIn("promoted_at", record)  # API uses promoted_at
                self.assertIn("verifier_principal", record)
        finally:
            for source_file in source_files:
                os.unlink(source_file)

    def test_list_dl2_samples(self):
        """Test listing unverified DL2 samples."""
        file_ids = []
        source_files = []

        for i in range(2):
            source_file = self._create_test_file(f"candidate,votes\nC{i},100\n")
            source_files.append(source_file)
            file_id = self.sync.stage_dl2_file(source_file, metadata={"state": "VA"})
            file_ids.append(file_id)

        try:
            # List DL2 samples
            samples = self.sync.list_dl2_samples(limit=10)
            self.assertEqual(len(samples), 2)

            # Verify structure (API uses 'hash' not 'content_hash')
            for sample in samples:
                self.assertIn("file_id", sample)
                self.assertIn("hash", sample)  # API uses 'hash'
                self.assertIn("created_at", sample)  # API uses created_at not staged_at
                self.assertIn(sample["file_id"], file_ids)
        finally:
            for source_file in source_files:
                os.unlink(source_file)

    def test_list_dl1_approved(self):
        """Test listing verified DL1 approved samples."""
        # Stage and promote a file
        source_file = self._create_test_file()
        try:
            file_id = self.sync.stage_dl2_file(source_file, metadata={"state": "VA"})
            self.sync.promote_to_dl1(file_id, verifier_principal="analyst@test.org")

            # List DL1 approved
            approved = self.sync.list_dl1_approved(limit=10)
            self.assertEqual(len(approved), 1)

            # Verify structure (API doesn't include verifier_principal in list output)
            record = approved[0]
            self.assertEqual(record["file_id"], file_id)
            self.assertIn("hash", record)  # API uses 'hash'
            self.assertIn("approved_at", record)  # API uses 'approved_at'
            # Note: verifier_principal is stored in promotion_history, not in list output
        finally:
            os.unlink(source_file)

    def test_storage_stats(self):
        """Test storage statistics calculation."""
        source_file = self._create_test_file()
        try:
            file_id = self.sync.stage_dl2_file(source_file, metadata={"state": "VA"})

            # Get stats before promotion (API uses nested structure)
            stats_before = self.sync.get_storage_stats()
            self.assertEqual(stats_before["dl2"]["file_count"], 1)
            self.assertEqual(stats_before["dl1"]["file_count"], 0)
            self.assertGreater(stats_before["dl2"]["total_size_bytes"], 0)

            # Promote and check stats again
            # IMPORTANT: Files are COPIED to DL1, not MOVED (forensic preservation)
            self.sync.promote_to_dl1(file_id, verifier_principal="analyst@test.org")
            stats_after = self.sync.get_storage_stats()
            # DL2 count remains 1 (file stays for audit trail)
            self.assertEqual(stats_after["dl2"]["file_count"], 1)
            # DL1 count is now 1 (copy created)
            self.assertEqual(stats_after["dl1"]["file_count"], 1)
            self.assertGreater(stats_after["dl1"]["total_size_bytes"], 0)
        finally:
            os.unlink(source_file)

    def test_promotion_safety_checks(self):
        """Test that promotion validates file existence."""
        # Try to promote non-existent file
        with self.assertRaises(FileNotFoundError):
            self.sync.promote_to_dl1(
                "nonexistent_file_id",
                verifier_principal="analyst@test.org"
            )

    def test_metadata_persistence(self):
        """Test that metadata is persisted correctly."""
        source_file = self._create_test_file()
        try:
            file_id = self.sync.stage_dl2_file(
                source_file,
                metadata={
                    "state": "Virginia",
                    "county": "Arlington",
                    "contest": "President",
                    "source_url": "https://example.org/results",
                }
            )

            # Create new sync instance (simulates server restart)
            sync2 = LocalStorageSync(self.verification_dir)
            samples = sync2.list_dl2_samples(limit=10)

            # Verify metadata persisted
            self.assertTrue(len(samples) > 0)
            found = next((s for s in samples if s["file_id"] == file_id), None)
            self.assertIsNotNone(found)
        finally:
            os.unlink(source_file)


class TestComputeFileHash(unittest.TestCase):
    """Tests for static file hashing utility."""

    def test_compute_file_hash(self):
        """Test SHA256 file hash computation."""
        content = "test content"
        fd, temp_path = tempfile.mkstemp(text=True)
        try:
            os.write(fd, content.encode('utf-8'))
            os.close(fd)

            hash_val = LocalStorageSync.compute_file_hash(temp_path)
            self.assertIsNotNone(hash_val)
            self.assertEqual(len(hash_val), 64)  # SHA256 hex is 64 chars

            # Compute again - should be identical (content addressable)
            hash_val2 = LocalStorageSync.compute_file_hash(temp_path)
            self.assertEqual(hash_val, hash_val2)
        finally:
            os.unlink(temp_path)

    def test_hash_different_for_different_content(self):
        """Test that different content produces different hashes."""
        fd1, path1 = tempfile.mkstemp(text=True)
        fd2, path2 = tempfile.mkstemp(text=True)

        try:
            os.write(fd1, b"content1")
            os.write(fd2, b"content2")
            os.close(fd1)
            os.close(fd2)

            hash1 = LocalStorageSync.compute_file_hash(path1)
            hash2 = LocalStorageSync.compute_file_hash(path2)
            self.assertNotEqual(hash1, hash2)
        finally:
            os.unlink(path1)
            os.unlink(path2)


if __name__ == "__main__":
    unittest.main()

"""
Local DL1/DL2 File System Sync Implementation

Manages bidirectional synchronization between:
- DL2 (Unverified AI-extracted data)
- DL1 (Verified ground truth data)

All operations are local filesystem-based with no external dependencies.
"""

import hashlib
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import orjson


class LocalStorageSync:
    """
    Local filesystem-based DL1/DL2 synchronization.

    Manages:
    - File staging from extraction → DL2
    - Promotion from DL2 → DL1 (on approval)
    - Deduplication via content hashing
    - Immutable promotion audit trail
    """

    def __init__(self, verification_dir: str | Path, create_if_missing: bool = True):
        """
        Initialize local storage sync.

        Args:
            verification_dir: Root directory for verification/sync storage
            create_if_missing: Create directories if they don't exist
        """
        self.verification_dir = Path(verification_dir)
        self.dl2_dir = self.verification_dir / "dl2"
        self.dl1_dir = self.verification_dir / "dl1"
        self.metadata_path = self.verification_dir / "sync_metadata.json"
        self.promotion_log_path = self.verification_dir / "promotion_history.jsonl"
        self._lock = threading.RLock()

        if create_if_missing:
            self.verification_dir.mkdir(parents=True, exist_ok=True)
            self.dl2_dir.mkdir(parents=True, exist_ok=True)
            self.dl1_dir.mkdir(parents=True, exist_ok=True)

    def is_available(self) -> bool:
        """Check if storage directories are accessible."""
        try:
            return (
                self.verification_dir.exists()
                and self.dl2_dir.exists()
                and self.dl1_dir.exists()
            )
        except Exception:
            return False

    @staticmethod
    def compute_file_hash(file_path: str | Path, algorithm: str = "sha256") -> str:
        """Compute cryptographic hash of file content."""
        file_path = Path(file_path)
        hasher = hashlib.new(algorithm)
        try:
            with open(file_path, "rb") as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            raise ValueError(f"Failed to hash {file_path}: {e}")

    def _load_metadata(self) -> Dict[str, Any]:
        """Load sync metadata from disk."""
        if not self.metadata_path.exists():
            return {
                "version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "file_hashes": {},  # file_id → {hash, timestamp, location}
                "dedup_index": {},  # hash → [file_id, ...]
                "promotion_index": {},  # file_id → promotion_record
            }

        try:
            with open(self.metadata_path, "rb") as f:
                return orjson.loads(f.read())
        except Exception:
            return {
                "version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "file_hashes": {},
                "dedup_index": {},
                "promotion_index": {},
            }

    def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save sync metadata to disk (atomic)."""
        tmp_path = self.metadata_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "wb") as f:
                f.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(self.metadata_path)
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            raise ValueError(f"Failed to save metadata: {e}")

    def list_dl2_samples(self, limit: int | None = None) -> List[Dict[str, Any]]:
        """List unverified samples in DL2."""
        files = []
        try:
            for file_path in sorted(self.dl2_dir.glob("*.csv")):
                metadata = self._load_metadata()
                file_id = file_path.stem
                file_hash = self.compute_file_hash(file_path)

                entry = {
                    "file_id": file_id,
                    "filename": file_path.name,
                    "size_bytes": file_path.stat().st_size,
                    "hash": file_hash,
                    "created_at": datetime.fromtimestamp(
                        file_path.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                    "promoted": file_id in metadata.get("promotion_index", {}),
                }
                files.append(entry)

                if limit and len(files) >= limit:
                    break
        except Exception:
            pass

        return files

    def list_dl1_approved(self, limit: int | None = None) -> List[Dict[str, Any]]:
        """List verified/approved samples in DL1."""
        files = []
        try:
            for file_path in sorted(self.dl1_dir.glob("*.csv")):
                file_id = file_path.stem
                file_hash = self.compute_file_hash(file_path)

                entry = {
                    "file_id": file_id,
                    "filename": file_path.name,
                    "size_bytes": file_path.stat().st_size,
                    "hash": file_hash,
                    "approved_at": datetime.fromtimestamp(
                        file_path.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                }
                files.append(entry)

                if limit and len(files) >= limit:
                    break
        except Exception:
            pass

        return files

    def stage_dl2_file(
        self,
        source_path: str | Path,
        file_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        """
        Stage a new file into DL2.

        Args:
            source_path: Path to source file
            file_id: Optional file identifier (auto-generated if not provided)
            metadata: Optional metadata dict

        Returns:
            file_id of staged file
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        if not file_id:
            # Auto-generate ID from timestamp + hash
            file_hash = self.compute_file_hash(source_path)
            file_id = f"dl2_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{file_hash[:8]}"

        dest_path = self.dl2_dir / f"{file_id}.csv"
        if dest_path.exists():
            raise FileExistsError(f"File already exists in DL2: {file_id}")

        with self._lock:
            # Copy file
            shutil.copy2(source_path, dest_path)

            # Compute hash
            file_hash = self.compute_file_hash(dest_path)

            # Update metadata
            sync_metadata = self._load_metadata()
            sync_metadata["file_hashes"][file_id] = {
                "hash": file_hash,
                "location": "dl2",
                "staged_at": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {},
            }

            # Update dedup index
            dedup = sync_metadata.setdefault("dedup_index", {})
            if file_hash not in dedup:
                dedup[file_hash] = []
            if file_id not in dedup[file_hash]:
                dedup[file_hash].append(file_id)

            self._save_metadata(sync_metadata)

        return file_id

    def promote_to_dl1(
        self,
        file_id: str,
        verifier_principal: str | None = None,
        verification_notes: str | None = None,
    ) -> Dict[str, Any]:
        """
        Promote an approved file from DL2 to DL1.

        Args:
            file_id: File ID in DL2
            verifier_principal: Principal who approved
            verification_notes: Approval notes

        Returns:
            Promotion record
        """
        dl2_path = self.dl2_dir / f"{file_id}.csv"
        if not dl2_path.exists():
            raise FileNotFoundError(f"File not found in DL2: {file_id}")

        dl1_path = self.dl1_dir / f"{file_id}.csv"
        if dl1_path.exists():
            raise FileExistsError(f"File already exists in DL1: {file_id}")

        with self._lock:
            # Copy to DL1
            shutil.copy2(dl2_path, dl1_path)

            # Create promotion record
            promotion_record = {
                "file_id": file_id,
                "source_location": "dl2",
                "dest_location": "dl1",
                "promoted_at": datetime.now(timezone.utc).isoformat(),
                "verifier_principal": verifier_principal,
                "verification_notes": verification_notes,
                "source_hash": self.compute_file_hash(dl2_path),
                "dest_hash": self.compute_file_hash(dl1_path),
            }

            # Update metadata
            sync_metadata = self._load_metadata()
            sync_metadata["promotion_index"][file_id] = promotion_record

            # Mark as promoted in file_hashes
            if file_id in sync_metadata["file_hashes"]:
                sync_metadata["file_hashes"][file_id]["promoted"] = True
                sync_metadata["file_hashes"][file_id]["promoted_at"] = (
                    promotion_record["promoted_at"]
                )

            self._save_metadata(sync_metadata)

            # Append to promotion history log
            self._append_promotion_log(promotion_record)

        return promotion_record

    def _append_promotion_log(self, record: Dict[str, Any]) -> None:
        """Append promotion record to JSONL history log."""
        try:
            with open(self.promotion_log_path, "ab") as f:
                f.write(orjson.dumps(record) + b"\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            pass

    def get_promotion_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent promotion history."""
        records = []
        try:
            if self.promotion_log_path.exists():
                with open(self.promotion_log_path, "rb") as f:
                    for line in f:
                        if line.strip():
                            try:
                                record = orjson.loads(line)
                                records.append(record)
                            except Exception:
                                pass
        except Exception:
            pass

        # Return most recent first
        return sorted(records, key=lambda r: r.get("promoted_at", ""), reverse=True)[
            :limit
        ]

    def get_promotion_by_id(self, file_id: str) -> Dict[str, Any] | None:
        """Get promotion record for a specific file."""
        metadata = self._load_metadata()
        return metadata.get("promotion_index", {}).get(file_id)

    def find_duplicates(self, file_hash: str) -> List[str]:
        """Find all file IDs with matching content hash."""
        metadata = self._load_metadata()
        return metadata.get("dedup_index", {}).get(file_hash, [])

    def delete_file(self, file_id: str, location: str = "dl2") -> bool:
        """Delete a file from DL2 or DL1."""
        if location == "dl2":
            target_dir = self.dl2_dir
        elif location == "dl1":
            target_dir = self.dl1_dir
        else:
            raise ValueError(f"Invalid location: {location}")

        file_path = target_dir / f"{file_id}.csv"
        if not file_path.exists():
            return False

        with self._lock:
            file_path.unlink()

            # Update metadata
            metadata = self._load_metadata()
            if file_id in metadata.get("file_hashes", {}):
                del metadata["file_hashes"][file_id]

            self._save_metadata(metadata)

        return True

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        dl2_files = list(self.dl2_dir.glob("*.csv"))
        dl1_files = list(self.dl1_dir.glob("*.csv"))

        dl2_size = sum(f.stat().st_size for f in dl2_files)
        dl1_size = sum(f.stat().st_size for f in dl1_files)

        metadata = self._load_metadata()
        promoted_count = len(metadata.get("promotion_index", {}))

        return {
            "dl2": {
                "file_count": len(dl2_files),
                "total_size_bytes": dl2_size,
            },
            "dl1": {
                "file_count": len(dl1_files),
                "total_size_bytes": dl1_size,
            },
            "total_promoted": promoted_count,
            "dedup_groups": len(metadata.get("dedup_index", {})),
        }

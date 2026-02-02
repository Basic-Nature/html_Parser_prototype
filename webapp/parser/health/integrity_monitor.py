"""
integrity_monitor.py

Context-aware integrity monitoring system with:
- Download cache deduplication (15-minute TTL)
- Async file integrity verification (SHA-256)
- HuggingFace NLP integration (replacing OpenAI)
- PyTorch neural network for health scoring
- Session flag prioritization
- Context library persistence
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import orjson

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..config import LOG_DIR, OUTPUT_DIR, PROJECT_ROOT
from ..Context_Integration.librarian import atomic_write_json, clean_for_json
from ..utils.logger_singleton import logger

# Constants
DOWNLOAD_CACHE_DIR = Path(LOG_DIR) / "download_cache"
DOWNLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_TTL_SECONDS = 15 * 60  # 15 minutes
SESSION_STORAGE_QUOTA = 5 * 1024 * 1024  # 5MB
DOWNLOAD_CACHE_QUOTA = 50 * 1024 * 1024  # 50MB
INTEGRITY_LOG = Path(LOG_DIR) / "integrity_monitor.jsonl"
CONFIDENCE_THRESHOLD = 0.7

# Cache state
_download_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = asyncio.Lock()


class IntegrityNeuralNetwork(nn.Module if TORCH_AVAILABLE else object):
    """
    PyTorch neural network for health scoring with confidence thresholds.
    Predicts integrity score (0-1) based on session context features.
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for IntegrityNeuralNetwork")
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x
    
    def predict_with_confidence(self, features: Any) -> Tuple[float, float]:
        """
        Predict integrity score with confidence.
        Returns (score, confidence) where both are in [0, 1].
        Expects features to be a torch.Tensor when torch is available.
        """
        with torch.no_grad():
            self.eval()
            score = self.forward(features).item()
            # Confidence based on distance from decision boundary (0.5)
            confidence = 1.0 - abs(score - 0.5) * 2
            return score, confidence


class HuggingFaceNLPAnalyzer:
    """
    HuggingFace-based NLP analyzer for session flag assessment.
    Replaces OpenAI with local/private transformer models.
    """
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.ner_pipeline = None
        self._initialized = False
        
    def _lazy_init(self):
        """Lazy initialization to avoid loading models at import time."""
        if self._initialized or not TRANSFORMERS_AVAILABLE:
            return
            
        try:
            # Use lightweight, privacy-focused models
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            # NER pipeline for entity extraction
            self.ner_pipeline = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple"
            )
            
            self._initialized = True
            logger.info("[IntegrityMonitor] HuggingFace NLP models loaded successfully")
        except Exception as e:
            logger.error(f"[IntegrityMonitor] Failed to load HuggingFace models: {e}")
            
    def analyze_session_flags(self, session_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze session context to prioritize integrity flags.
        Returns: {
            "priority_score": float,  # 0-1
            "entities": List[Dict],
            "risk_factors": List[str],
            "confidence": float
        }
        """
        self._lazy_init()
        
        if not self._initialized:
            return {
                "priority_score": 0.5,
                "entities": [],
                "risk_factors": ["NLP unavailable"],
                "confidence": 0.0
            }
            
        try:
            # Extract text from context
            context_text = " ".join([
                str(session_context.get("contest", "")),
                str(session_context.get("state", "")),
                str(session_context.get("county", "")),
                str(session_context.get("handler", ""))
            ])
            
            # Entity extraction
            entities = self.ner_pipeline(context_text) if context_text.strip() else []
            
            # Risk factor detection (heuristic + NER)
            risk_factors = []
            
            # Check for suspicious patterns
            suspicious_keywords = ["test", "demo", "fake", "sample", "invalid"]
            for keyword in suspicious_keywords:
                if keyword in context_text.lower():
                    risk_factors.append(f"suspicious_keyword:{keyword}")
                    
            # Check for missing critical fields
            if not session_context.get("state"):
                risk_factors.append("missing_state")
            if not session_context.get("county"):
                risk_factors.append("missing_county")
                
            # Calculate priority score
            base_score = 0.5
            entity_boost = min(len(entities) * 0.05, 0.3)
            risk_penalty = len(risk_factors) * 0.1
            
            priority_score = max(0.0, min(1.0, base_score + entity_boost - risk_penalty))
            confidence = 0.8 if entities else 0.5
            
            return {
                "priority_score": priority_score,
                "entities": entities,
                "risk_factors": risk_factors,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"[IntegrityMonitor] NLP analysis failed: {e}")
            return {
                "priority_score": 0.5,
                "entities": [],
                "risk_factors": [f"analysis_error: {str(e)}"],
                "confidence": 0.0
            }


class IntegrityMonitor:
    """
    Main integrity monitoring coordinator.
    Manages download cache, file verification, and health checks.
    """
    
    def __init__(self, context_library_path: Optional[Path] = None):
        self.context_library_path = context_library_path or Path(PROJECT_ROOT) / "webapp" / "parser" / "Context_Integration" / "Context_Library" / "context_library.json"
        self.nlp_analyzer = HuggingFaceNLPAnalyzer()
        self.health_network = None
        
        # Initialize neural network if available
        if TORCH_AVAILABLE:
            try:
                self.health_network = IntegrityNeuralNetwork()
                logger.info("[IntegrityMonitor] Integrity neural network initialized")
            except Exception as e:
                logger.error(f"[IntegrityMonitor] Failed to init neural network: {e}")
                
    async def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._compute_hash_sync, file_path)
        
    def _compute_hash_sync(self, file_path: Path) -> str:
        """Synchronous hash computation."""
        hasher = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"[IntegrityMonitor] Hash computation failed for {file_path}: {e}")
            return ""
            
    async def verify_download_integrity(
        self,
        file_path: Path,
        expected_hash: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify file integrity with async hash computation.
        Returns: {
            "valid": bool,
            "hash": str,
            "size": int,
            "verified_at": float,
            "session_id": str
        }
        """
        if not file_path.exists():
            return {"valid": False, "error": "file_not_found"}
            
        file_hash = await self.compute_file_hash(file_path)
        file_size = file_path.stat().st_size
        
        valid = True
        if expected_hash and file_hash != expected_hash:
            valid = False
            logger.warning(f"[IntegrityMonitor] Hash mismatch for {file_path.name}: expected {expected_hash}, got {file_hash}")
            
        result = {
            "valid": valid,
            "hash": file_hash,
            "size": file_size,
            "verified_at": time.time(),
            "session_id": session_id or "unknown"
        }
        
        # Log to integrity monitor
        self._log_integrity_event({
            "type": "file_verification",
            "file": str(file_path.name),
            "result": result
        })
        
        return result
        
    async def get_or_cache_download(
        self,
        file_name: str,
        principal: str,
        session_id: str,
        file_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Get download from cache or compute hash and cache it.
        Implements 15-minute TTL and deduplication across sessions with same principal.
        
        Returns: {
            "cache_hit": bool,
            "hash": str,
            "size": int,
            "cached_at": float,
            "ttl_expires_at": float,
            "sessions": List[str]  # All sessions sharing this download
        }
        """
        async with _cache_lock:
            cache_key = f"{principal}:{file_name}"
            now = time.time()
            
            # Check if cached and not expired
            if cache_key in _download_cache:
                cached = _download_cache[cache_key]
                if now < cached["ttl_expires_at"]:
                    # Update session list if new session
                    if session_id not in cached["sessions"]:
                        cached["sessions"].append(session_id)
                    return {
                        "cache_hit": True,
                        **cached
                    }
                else:
                    # Expired, remove from cache
                    del _download_cache[cache_key]
                    
            # Compute hash and cache
            if not file_path:
                file_path = Path(OUTPUT_DIR) / file_name
                
            if not file_path.exists():
                return {"cache_hit": False, "error": "file_not_found"}
                
            file_hash = await self.compute_file_hash(file_path)
            file_size = file_path.stat().st_size
            
            cache_entry = {
                "hash": file_hash,
                "size": file_size,
                "cached_at": now,
                "ttl_expires_at": now + DOWNLOAD_TTL_SECONDS,
                "sessions": [session_id],
                "principal": principal
            }
            
            _download_cache[cache_key] = cache_entry
            
            # Enforce cache quota
            await self._enforce_cache_quota()
            
            return {
                "cache_hit": False,
                **cache_entry
            }
            
    async def _enforce_cache_quota(self):
        """
        Enforce download cache quota (50MB) using LRU eviction.
        Must be called while holding _cache_lock.
        """
        total_size = sum(entry["size"] for entry in _download_cache.values())
        
        if total_size <= DOWNLOAD_CACHE_QUOTA:
            return
            
        # Sort by cached_at (oldest first)
        sorted_entries = sorted(
            _download_cache.items(),
            key=lambda x: x[1]["cached_at"]
        )
        
        # Evict oldest until under 80% quota
        target_size = int(DOWNLOAD_CACHE_QUOTA * 0.8)
        for key, entry in sorted_entries:
            if total_size <= target_size:
                break
            del _download_cache[key]
            total_size -= entry["size"]
            logger.info(f"[IntegrityMonitor] Evicted cache entry: {key} ({entry['size']} bytes)")
            
    def assess_session_health(
        self,
        session_context: Dict[str, Any],
        flags: List[str]
    ) -> Dict[str, Any]:
        """
        Assess session health using NLP + neural network.
        Returns: {
            "health_score": float,  # 0-1
            "confidence": float,
            "priority": str,  # "high", "medium", "low"
            "nlp_analysis": Dict,
            "flags": List[str],
            "recommendations": List[str]
        }
        """
        # NLP analysis
        nlp_result = self.nlp_analyzer.analyze_session_flags(session_context)
        
        # Neural network prediction (if available)
        health_score = 0.5
        nn_confidence = 0.0
        
        if self.health_network and TORCH_AVAILABLE:
            try:
                # Build feature vector from context
                features = self._build_feature_vector(session_context, flags, nlp_result)
                health_score, nn_confidence = self.health_network.predict_with_confidence(features)
            except Exception as e:
                logger.error(f"[IntegrityMonitor] Neural network prediction failed: {e}")
                
        # Combine NLP and NN confidence
        combined_confidence = (nlp_result["confidence"] + nn_confidence) / 2
        
        # Determine priority
        if health_score >= 0.7 and combined_confidence >= CONFIDENCE_THRESHOLD:
            priority = "high"
        elif health_score >= 0.5:
            priority = "medium"
        else:
            priority = "low"
            
        # Generate recommendations
        recommendations = []
        if health_score < 0.5:
            recommendations.append("Review session context for missing critical fields")
        if nlp_result["risk_factors"]:
            recommendations.append(f"Address risk factors: {', '.join(nlp_result['risk_factors'])}")
        if combined_confidence < CONFIDENCE_THRESHOLD:
            recommendations.append("Low confidence - manual review recommended")
            
        result = {
            "health_score": health_score,
            "confidence": combined_confidence,
            "priority": priority,
            "nlp_analysis": nlp_result,
            "flags": flags,
            "recommendations": recommendations,
            "timestamp": time.time()
        }
        
        # Persist to context library if high priority
        if priority == "high":
            self._persist_to_context_library(session_context, result)
            
        return result
        
    def _build_feature_vector(
        self,
        session_context: Dict[str, Any],
        flags: List[str],
        nlp_result: Dict[str, Any]
    ) -> Any:
        """
        Build feature vector for neural network from session data.
        Returns 128-dimensional tensor (torch.Tensor when torch is available).
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for feature vector")
            
        # Initialize feature vector
        features = torch.zeros(128)
        
        # Binary features (0-31): field presence
        features[0] = 1.0 if session_context.get("state") else 0.0
        features[1] = 1.0 if session_context.get("county") else 0.0
        features[2] = 1.0 if session_context.get("contest") else 0.0
        features[3] = 1.0 if session_context.get("year") else 0.0
        features[4] = 1.0 if session_context.get("handler") else 0.0
        
        # Numeric features (32-63): counts and scores
        features[32] = len(flags) / 10.0  # Normalize flag count
        features[33] = nlp_result.get("priority_score", 0.5)
        features[34] = nlp_result.get("confidence", 0.5)
        features[35] = len(nlp_result.get("entities", [])) / 10.0
        features[36] = len(nlp_result.get("risk_factors", [])) / 5.0
        
        # Text embeddings would go in 64-127 (simplified here)
        # In production, use HuggingFace embeddings of contest/state/county text
        
        return features.unsqueeze(0)  # Add batch dimension
        
    def _persist_to_context_library(
        self,
        session_context: Dict[str, Any],
        health_result: Dict[str, Any]
    ):
        """
        Persist high-priority health results to context_library.json.
        This is the local safe stage for validated data.
        """
        try:
            # Load existing library
            if self.context_library_path.exists():
                with open(self.context_library_path, "rb") as f:
                    library = orjson.loads(f.read())
            else:
                library = {}
                
            # Add to integrity_checks section
            if "integrity_checks" not in library:
                library["integrity_checks"] = []
                
            entry = {
                "session_id": session_context.get("session_id"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "health_score": health_result["health_score"],
                "confidence": health_result["confidence"],
                "priority": health_result["priority"],
                "context_summary": {
                    "state": session_context.get("state"),
                    "county": session_context.get("county"),
                    "contest": session_context.get("contest"),
                    "handler": session_context.get("handler")
                },
                "nlp_entities": health_result["nlp_analysis"].get("entities", []),
                "risk_factors": health_result["nlp_analysis"].get("risk_factors", []),
                "recommendations": health_result["recommendations"]
            }
            
            library["integrity_checks"].append(entry)
            
            # Keep only last 100 entries to prevent bloat
            library["integrity_checks"] = library["integrity_checks"][-100:]
            
            # Atomic write
            atomic_write_json(self.context_library_path, clean_for_json(library))
            
            logger.info(f"[IntegrityMonitor] Persisted high-priority health check to context library")
            
        except Exception as e:
            logger.error(f"[IntegrityMonitor] Failed to persist to context library: {e}")
            
    def _log_integrity_event(self, event: Dict[str, Any]):
        """Log integrity event to JSONL file."""
        try:
            event["timestamp"] = time.time()
            with open(INTEGRITY_LOG, "ab") as f:
                f.write(orjson.dumps(event) + b"\n")
        except Exception as e:
            logger.error(f"[IntegrityMonitor] Failed to log event: {e}")


# Singleton instance
_monitor_instance: Optional[IntegrityMonitor] = None

def get_integrity_monitor() -> IntegrityMonitor:
    """Get or create singleton IntegrityMonitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = IntegrityMonitor()
    return _monitor_instance

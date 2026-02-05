"""
Quality Assurance Module: Data Classification & Verification Pipeline

Implements DL1/DL2 classification for election data:
- DL1 = Unverified (freshly extracted)
- DL2 = Verified (human approved + QA passed)

Provides:
- data_classifier.py: Classification logic + automated QA checks
- qa_endpoints.py: Flask blueprint with REST API
"""

from .data_classifier import (
    ActionType,
    ClassificationResult,
    DatasetMetadata,
    DLStatus,
    QAIssue,
    QAIssueType,
    classify_as_dl1,
    get_dataset_lineage,
    get_dl2_inventory,
    get_pending_dl2_reviews,
    promote_to_dl2,
)
from .qa_endpoints import qa_bp

__all__ = [
    "DLStatus",
    "QAIssueType",
    "ActionType",
    "QAIssue",
    "ClassificationResult",
    "DatasetMetadata",
    "classify_as_dl1",
    "promote_to_dl2",
    "get_pending_dl2_reviews",
    "get_dl2_inventory",
    "get_dataset_lineage",
    "qa_bp",
]

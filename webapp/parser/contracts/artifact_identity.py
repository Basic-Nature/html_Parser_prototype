"""Pure artifact identity handoff contract.

This module defines how an already-known immutable-content SHA-256 may be
handed between acquisition/orchestration code and parser consumers.

It does not:
- read files;
- hash content;
- inspect paths or URLs;
- invoke parsers, OCR, geometry, persistence, or network services.

Missing identity is represented by the absence of an ArtifactIdentityHandoff
object. Consumers must not manufacture an identity from a path, URL, or parser
observation.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._contract_validation import require_sha256


ARTIFACT_IDENTITY_HANDOFF_CONTRACT = "artifact_identity_handoff_v1"
ARTIFACT_IDENTITY_ALGORITHM = "sha256"
ARTIFACT_IDENTITY_SEMANTICS = "SHA256_OF_IMMUTABLE_CONTENT_BYTES"
ARTIFACT_IDENTITY_CANONICAL_HEX_CASE = "lowercase"
MISSING_ARTIFACT_IDENTITY_POLICY = (
    "PRESERVE_UNKNOWN_DO_NOT_HASH_FOR_INSTRUMENTATION"
)
INVALID_ARTIFACT_IDENTITY_POLICY = (
    "REJECT_INVALID_FORMAT_NO_INFERENCE"
)


@dataclass(frozen=True)
class ArtifactIdentityHandoff:
    """Validated content identity supplied by an upstream owner."""

    document_sha256: str
    algorithm: str = ARTIFACT_IDENTITY_ALGORITHM
    semantics: str = ARTIFACT_IDENTITY_SEMANTICS

    def __post_init__(self) -> None:
        normalized_sha256 = require_sha256(
            "document_sha256",
            self.document_sha256,
        )

        if self.algorithm != ARTIFACT_IDENTITY_ALGORITHM:
            raise ValueError(
                "algorithm must be exactly 'sha256'"
            )

        if self.semantics != ARTIFACT_IDENTITY_SEMANTICS:
            raise ValueError(
                "semantics must be exactly "
                "'SHA256_OF_IMMUTABLE_CONTENT_BYTES'"
            )

        # Hex case is representation-only canonicalization, not inferred
        # identity. The validated bytes identity remains unchanged.
        object.__setattr__(
            self,
            "document_sha256",
            normalized_sha256,
        )

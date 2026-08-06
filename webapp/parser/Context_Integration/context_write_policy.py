from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ContextWriteKind(str, Enum):
    """Allowed persistence destinations for context-related data."""

    NONE = "none"
    RUNTIME = "runtime"
    EVIDENCE = "evidence"
    LEARNED = "learned"
    CANONICAL = "canonical"


@dataclass(frozen=True)
class ContextWritePolicy:
    """
    Controls which persistence operations are permitted.

    Parser and enrichment code should normally use the default policy, which
    allows runtime/evidence output but blocks learned and canonical promotion.
    """

    allow_runtime: bool = True
    allow_evidence: bool = True
    allow_learned: bool = False
    allow_canonical: bool = False

    def permits(self, write_kind: ContextWriteKind) -> bool:
        if write_kind is ContextWriteKind.NONE:
            return True
        if write_kind is ContextWriteKind.RUNTIME:
            return self.allow_runtime
        if write_kind is ContextWriteKind.EVIDENCE:
            return self.allow_evidence
        if write_kind is ContextWriteKind.LEARNED:
            return self.allow_learned
        if write_kind is ContextWriteKind.CANONICAL:
            return self.allow_canonical
        return False


DEFAULT_CONTEXT_WRITE_POLICY = ContextWritePolicy()

REVIEW_CONTEXT_WRITE_POLICY = ContextWritePolicy(
    allow_runtime=True,
    allow_evidence=True,
    allow_learned=True,
    allow_canonical=False,
)

# Reserved for the future explicit promotion service.
# ContextCoordinator currently rejects canonical writes even under this policy.
ADMIN_CONTEXT_WRITE_POLICY = ContextWritePolicy(
    allow_runtime=True,
    allow_evidence=True,
    allow_learned=True,
    allow_canonical=True,
)
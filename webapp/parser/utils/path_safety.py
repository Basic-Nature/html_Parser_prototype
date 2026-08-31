"""Pure filesystem path-containment helpers.

These helpers establish path ownership boundaries only. A path is a locator,
not an artifact identity.
"""

from __future__ import annotations

import os
from typing import Any


def is_path_within_root(
    candidate_path: str,
    root_path: str,
    *,
    path_module: Any = os.path,
) -> bool:
    """Return True only when candidate_path resolves inside root_path.

    The comparison is path-component aware rather than string-prefix based.
    It fails closed for malformed paths and cross-drive comparisons.
    """
    try:
        resolved_root = path_module.normcase(
            path_module.realpath(root_path)
        )
        resolved_candidate = path_module.normcase(
            path_module.realpath(candidate_path)
        )
        return (
            path_module.commonpath(
                [resolved_root, resolved_candidate]
            )
            == resolved_root
        )
    except (OSError, TypeError, ValueError):
        return False

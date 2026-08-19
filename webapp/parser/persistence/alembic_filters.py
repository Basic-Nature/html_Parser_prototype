"""Alembic autogenerate ownership filters.

ElectionPulse owns tables declared in Base.metadata. The database may also
contain extension, source-lane, workflow, or operator-managed tables.

Autogenerate must never infer that a database-only table should be dropped
merely because it is absent from ORM metadata. Intentional table removals must
be expressed by an explicit hand-written migration.

Modeled tables still compare normally, so column/type/index/constraint drift
remains visible.
"""

from __future__ import annotations

from typing import Any


def include_object(
    obj: Any,
    name: str | None,
    type_: str,
    reflected: bool,
    compare_to: Any,
) -> bool:
    """Exclude only reflected tables that have no metadata counterpart."""

    if (
        type_ == "table"
        and reflected
        and compare_to is None
    ):
        return False

    return True

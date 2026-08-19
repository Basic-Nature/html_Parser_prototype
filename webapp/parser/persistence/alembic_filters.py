"""Alembic autogenerate ownership filters.

ElectionPulse owns tables declared in Base.metadata. PostgreSQL extensions
such as PostGIS/TIGER may install additional objects in the same database.
Those extension-owned objects must never be interpreted as ElectionPulse
drop candidates during autogenerate.
"""

from __future__ import annotations

from typing import Any


EXTENSION_SCHEMAS = frozenset({"tiger", "tiger_data", "topology"})
EXTENSION_TABLES = frozenset({"spatial_ref_sys"})


def _object_schema(obj: Any) -> str | None:
    schema = getattr(obj, "schema", None)
    if schema:
        return str(schema)

    table = getattr(obj, "table", None)
    if table is None:
        return None

    table_schema = getattr(table, "schema", None)
    return str(table_schema) if table_schema else None


def _object_table_name(
    obj: Any,
    name: str | None,
    type_: str,
) -> str | None:
    if type_ == "table":
        return str(name) if name is not None else None

    table = getattr(obj, "table", None)
    if table is None:
        return None

    table_name = getattr(table, "name", None)
    return str(table_name) if table_name is not None else None


def include_object(
    obj: Any,
    name: str | None,
    type_: str,
    reflected: bool,
    compare_to: Any,
) -> bool:
    """Exclude only reflected objects owned by known DB extensions."""

    if not reflected or compare_to is not None:
        return True

    schema = _object_schema(obj)
    table_name = _object_table_name(obj, name, type_)

    if schema in EXTENSION_SCHEMAS:
        return False

    if table_name in EXTENSION_TABLES:
        return False

    return True

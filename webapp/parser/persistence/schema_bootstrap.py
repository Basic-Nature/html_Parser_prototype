from __future__ import annotations

"""
Explicit compatibility bootstrap for the Election Pulse application ORM schema.

Alembic remains the authority for versioned schema evolution.

This module preserves existing ``Base.metadata.create_all`` behavior
during architectural migration so production startup, health recovery,
and legacy callers can delegate to one implementation.

Importing this module does not create an engine, connect to a database,
or execute DDL. Schema creation occurs only when
``ensure_application_schema_compat`` is explicitly called.
"""

from sqlalchemy import inspect
from sqlalchemy.engine import Engine

from ..utils.models import Base


def verify_application_schema_compat(engine: Engine) -> None:
    """Verify the ORM schema without executing DDL.

    Runtime startup and health checks use this read-only contract. Missing
    application tables fail closed so Alembic remains the sole production DDL
    authority. Explicit bootstrap/admin tooling may still call
    ``ensure_application_schema_compat`` deliberately.
    """
    inspector = inspect(engine)
    missing: list[str] = []

    for table in Base.metadata.tables.values():
        schema = table.schema

        if inspector.has_table(table.name, schema=schema):
            continue

        qualified = (
            f"{schema}.{table.name}"
            if schema
            else table.name
        )
        missing.append(qualified)

    if missing:
        raise RuntimeError(
            "Application schema is not Alembic-ready; missing ORM tables: "
            + ", ".join(sorted(missing))
        )


def ensure_application_schema_compat(engine: Engine) -> None:
    """Create any missing tables represented by the current ORM metadata.

    This is a compatibility initializer, not a migration mechanism.
    Versioned schema changes belong in Alembic migrations.
    """

    Base.metadata.create_all(engine)

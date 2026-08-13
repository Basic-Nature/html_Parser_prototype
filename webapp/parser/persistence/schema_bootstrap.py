from __future__ import annotations

"""
Compatibility bootstrap for the Election Pulse application ORM schema.

Alembic remains the authority for versioned schema evolution.

This module preserves existing ``Base.metadata.create_all`` behavior
during architectural migration so production startup, health recovery,
and legacy callers can delegate to one implementation.

Importing this module does not create an engine, connect to a database,
or execute DDL. Schema creation occurs only when
``ensure_application_schema_compat`` is explicitly called.
"""

from sqlalchemy.engine import Engine

from ..utils.models import Base


def ensure_application_schema_compat(engine: Engine) -> None:
    """Create any missing tables represented by the current ORM metadata.

    This is a compatibility initializer, not a migration mechanism.
    Versioned schema changes belong in Alembic migrations.
    """

    Base.metadata.create_all(engine)

import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# Import our models for autogenerate
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from webapp.parser.utils.models import Base
from webapp.parser.persistence.alembic_filters import include_object

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


_PRODUCTION_LIKE_DEPLOY_ENVS = frozenset(
    {
        "production",
        "prod",
        "staging",
        "stage",
    }
)


def _is_production_like_execution() -> bool:
    """Return True when Alembic is running in a production-like host context."""
    deploy_env = os.getenv("DEPLOY_ENV", "").strip().lower()

    if deploy_env in _PRODUCTION_LIKE_DEPLOY_ENVS:
        return True

    # Azure App Service injects these host markers into the runtime.
    return bool(
        os.getenv("WEBSITE_SITE_NAME")
        or os.getenv("WEBSITE_INSTANCE_ID")
    )


def _resolve_database_url() -> str:
    """Resolve Alembic's database target with fail-closed production authority."""
    explicit_url = os.getenv("DATABASE_URL", "").strip()
    production_like = _is_production_like_execution()

    if production_like and not explicit_url:
        raise RuntimeError(
            "Alembic target authority violation: production-like execution "
            "requires an explicit DATABASE_URL."
        )

    url = explicit_url or config.get_main_option("sqlalchemy.url")

    if not url or not url.strip():
        raise RuntimeError(
            "Alembic target authority violation: no database URL is configured."
        )

    normalized_url = url.strip()

    if production_like and normalized_url.lower().startswith("sqlite"):
        raise RuntimeError(
            "Alembic target authority violation: SQLite is not allowed for "
            "production-like execution."
        )

    return normalized_url

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    import os
    url = _resolve_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        include_object=include_object,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    import os
    
    # Get database URL from environment or config
    database_url = _resolve_database_url()
    
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = database_url
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

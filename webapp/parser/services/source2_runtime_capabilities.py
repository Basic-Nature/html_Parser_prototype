"""Runtime-only source capability check. No client-supplied URL or source set."""
from __future__ import annotations
import os
from .source2_capability_scope import (MD, UTAH, MD_COMPACT, MD_CSV, LEGACY_COMPACT,
    LEGACY_CSV, SourceCapabilities, enabled, source_capabilities)
from .public_ballot_lens_policy import (configured_public_registry_pilot_source_id,
    configured_public_registry_execution_source_ids, public_registry_parse_feature_enabled)


def compact_registration_enabled(environ=None):
    env = os.environ if environ is None else environ
    # Keep the installed primary compact flag's historical literal semantics.
    return env.get(LEGACY_COMPACT, '').lower() in {'1','true','yes'} or enabled(env, MD_COMPACT)


def csv_endpoint_enabled(environ=None):
    env = os.environ if environ is None else environ
    return enabled(env, LEGACY_CSV) or (enabled(env, MD_COMPACT) and enabled(env, MD_CSV))


def _resolve(source_id):
    from webapp.parser.config import URL_LIST_FILE
    from .source_registry_runtime import resolve_public_registry_source
    return resolve_public_registry_source(URL_LIST_FILE, source_id)


def capabilities_for_source(source_id: str) -> SourceCapabilities:
    """Re-resolve approved execution authority at each disclosure boundary.

    A failed lookup is denial, not fallback to a caller or cached source list.
    No source URL is returned, logged, or accepted here.
    """
    env = dict(os.environ)
    deny = SourceCapabilities()
    try:
        if not (compact_registration_enabled(env) or csv_endpoint_enabled(env)):
            return deny
        if not public_registry_parse_feature_enabled(env):
            return deny
        if source_id == MD and env.get("SOURCE_REGISTRY_AUTHORITY_MODE", "legacy_file").strip().lower() != "durable_db":
            return deny
        primary = configured_public_registry_pilot_source_id(env)
        if source_id == MD and primary != UTAH:
            return deny
        configured = configured_public_registry_execution_source_ids(env)
        if source_id not in configured or primary not in configured:
            return deny
        # For the existing source preserve exact old flag behavior.
        env[LEGACY_COMPACT] = 'true' if env.get(LEGACY_COMPACT, '').lower() in {'1','true','yes'} else 'false'
        approved = []
        for sid in dict.fromkeys((primary, source_id)):
            obj = _resolve(sid)
            if (getattr(obj, 'registry_source_id', None) != sid or
                getattr(obj, 'registry_category', None) != 'curated'):
                return deny
            approved.append(sid)
        return source_capabilities(source_id=source_id, primary_id=primary,
            approved_execution_ids=tuple(approved), environ=env)
    except Exception:
        return deny

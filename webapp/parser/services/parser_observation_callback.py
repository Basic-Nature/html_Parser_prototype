"""Dormant transport-neutral delivery seam for parser observation bundles.

This helper knows only how to project the published noncanonical parser
observation bundle and synchronously pass it to an explicitly supplied
callable. It does not know about stores, sockets, routes, persistence,
sessions, public runtime activation, canonical output, or finalization.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from webapp.parser.contracts.table_pipeline import TablePipelineResult
from webapp.parser.services.parser_observation_bundle import (
    project_parser_observation_bundle,
)

ParserObservationEmitFunc = Callable[[dict[str, Any]], Any]


def emit_parser_observation_bundle_if_requested(
    result: TablePipelineResult,
    *,
    parser_observation_emit_func: ParserObservationEmitFunc | None = None,
) -> bool:
    """Deliver one bundle only when an explicit callback is supplied."""
    if parser_observation_emit_func is None:
        return False
    if not callable(parser_observation_emit_func):
        raise TypeError("parser_observation_emit_func must be callable")

    payload = project_parser_observation_bundle(result)
    parser_observation_emit_func(payload)
    return True

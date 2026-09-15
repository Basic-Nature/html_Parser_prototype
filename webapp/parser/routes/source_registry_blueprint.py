"""Inert Source Registry control-plane blueprint.

The blueprint is intentionally NOT registered by W20D. Its handlers remain
fail-closed while the mutation feature flag is false.
"""
from __future__ import annotations

from flask import Blueprint, jsonify

from webapp.parser.services.source_registry_governance import (
    SourceRegistryMutationDisabled,
    assert_mutation_feature_enabled,
)

source_registry_blueprint = Blueprint(
    "source_registry_control_plane_v1",
    __name__,
    url_prefix="/api/source-registry/v1",
)


@source_registry_blueprint.get("/status")
def source_registry_status():
    return jsonify({
        "contract": "source_registry_control_plane_v1",
        "registered_by_w20d": False,
        "authority_mode_default": "legacy_file",
        "mutations_enabled_default": False,
    })


def _mutation_disabled_response():
    try:
        assert_mutation_feature_enabled()
    except SourceRegistryMutationDisabled:
        return jsonify({
            "error": "source_registry_mutations_disabled",
            "message": "Source Registry mutation authority is not active.",
        }), 503
    return jsonify({
        "error": "source_registry_route_not_activated",
        "message": "Source Registry mutation route is not activated.",
    }), 503


@source_registry_blueprint.post("/proposals")
def create_source_registry_proposal():
    return _mutation_disabled_response()


@source_registry_blueprint.post("/proposals/<proposal_id>/review")
def review_source_registry_proposal(proposal_id: str):
    del proposal_id
    return _mutation_disabled_response()


@source_registry_blueprint.post("/proposals/<proposal_id>/publish")
def publish_source_registry_proposal(proposal_id: str):
    del proposal_id
    return _mutation_disabled_response()


@source_registry_blueprint.post("/bindings/<binding_id>/quarantine")
def quarantine_source_registry_binding(binding_id: str):
    del binding_id
    return _mutation_disabled_response()

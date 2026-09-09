from __future__ import annotations

from pathlib import Path
import re
from uuid import uuid4

from flask import Flask

from webapp.parser.routes.workflow_reviewer_blueprint import (
    create_workflow_reviewer_blueprint,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = REPO_ROOT / "webapp" / "Smart_Elections_Parser_Webapp.py"


def test_reviewer_blueprint_dispatch_contract():
    app = Flask(__name__)
    item_id = uuid4()
    comparison_id = uuid4()

    app.config["_WORKFLOW_REVIEWER_ROUTE_HANDLERS"] = {
        "api_workflow_v1_resolve_discrepancies": (
            lambda item_id, comparison_id: (
                {
                    "success": True,
                    "item_id": str(item_id),
                    "comparison_id": str(comparison_id),
                },
                200,
            )
        ),
    }
    app.register_blueprint(create_workflow_reviewer_blueprint())
    client = app.test_client()

    response = client.post(
        (
            f"/api/workflow/v1/reviewer/items/{item_id}/"
            f"comparisons/{comparison_id}/resolve"
        )
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["item_id"] == str(item_id)
    assert payload["comparison_id"] == str(comparison_id)

    assert client.get(
        (
            f"/api/workflow/v1/reviewer/items/{item_id}/"
            f"comparisons/{comparison_id}/resolve"
        )
    ).status_code == 405


def test_composition_root_reviewer_resolution_authority_contract():
    source = APP_PATH.read_text(encoding="utf-8")

    assert "create_workflow_reviewer_blueprint" in source
    assert "CAP_DISCREPANCY_RESOLVE" in source
    assert "def _workflow_reviewer_authority(required_capability: str):" in source
    compact = re.sub(r"\s+", "", source)
    assert "_workflow_reviewer_authority(CAP_DISCREPANCY_RESOLVE)" in compact
    assert "WORKFLOW_REVIEWER_MUTATIONS_ENABLED" in source
    assert (
        'os.environ.get("WORKFLOW_REVIEWER_MUTATIONS_ENABLED", "false")'
        in source
    )
    assert "_WORKFLOW_REVIEWER_RESOLUTION_REQUEST_KEYS" in source
    for key in (
        "expected_row_version",
        "resolution_code",
        "resolution_notes",
    ):
        assert f'"{key}"' in source
    assert (
        "body_keys != _WORKFLOW_REVIEWER_RESOLUTION_REQUEST_KEYS"
        in source
    )
    assert "resolve_workflow_comparison_discrepancies(" in source
    assert '"workflow_reviewer_mutations_disabled"' in source
    assert 'app.config["_WORKFLOW_REVIEWER_ROUTE_HANDLERS"]' in source

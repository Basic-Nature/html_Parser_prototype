# W15L public Workflow search-boundary regression contracts.

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from webapp.parser.services.workflow_reader import (
    read_workflow_facets,
    read_workflow_items,
    read_workflow_stats,
)
from webapp.parser.utils.models import Base, WorkflowItem


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_RUNTIME = (
    REPO_ROOT
    / "webapp"
    / "parser"
    / "services"
    / "public_read_runtime.py"
)


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _seed_search_rows(session) -> None:
    session.add_all(
        [
            WorkflowItem(
                lifecycle_state="active",
                current_stage="independent_acquisition",
                stage_condition="in_progress",
                priority=5,
                election_year=2024,
                state="Arizona",
                jurisdiction_name="Pima",
                jurisdiction_type="county",
                contest="President",
                source_race_id="AZ-2024-PRES",
                source_url=(
                    "https://hidden-only-token.invalid/"
                    "official-source"
                ),
                workflow_metadata={},
            ),
            WorkflowItem(
                lifecycle_state="blocked",
                current_stage="source_intake",
                stage_condition="pending",
                priority=3,
                election_year=2024,
                state="Texas",
                jurisdiction_name="Tarrant",
                jurisdiction_type="county",
                contest="US Senate",
                source_race_id="TX-2024-SEN",
                source_url="https://example.invalid/tx",
                workflow_metadata={},
            ),
        ]
    )
    session.commit()


def _facet_total(payload: dict) -> int:
    state_rows = payload["facets"]["state"]
    return sum(int(row["count"]) for row in state_rows)


def test_internal_default_preserves_source_url_search(db_session) -> None:
    _seed_search_rows(db_session)

    payload = read_workflow_items(
        db_session,
        {"search": "hidden-only-token"},
    )

    assert payload["pagination"]["total"] == 1
    assert payload["items"][0]["scope"]["source_race_id"] == "AZ-2024-PRES"


def test_public_safe_items_search_excludes_withheld_source_url(
    db_session,
) -> None:
    _seed_search_rows(db_session)

    payload = read_workflow_items(
        db_session,
        {"search": "hidden-only-token"},
        include_source_url_search=False,
    )

    assert payload["pagination"]["total"] == 0
    assert payload["items"] == []


@pytest.mark.parametrize(
    "visible_term",
    [
        "President",
        "Pima",
        "AZ-2024-PRES",
    ],
)
def test_public_safe_items_search_keeps_visible_fields(
    db_session,
    visible_term,
) -> None:
    _seed_search_rows(db_session)

    payload = read_workflow_items(
        db_session,
        {"search": visible_term},
        include_source_url_search=False,
    )

    assert payload["pagination"]["total"] == 1
    assert payload["items"][0]["scope"]["source_race_id"] == "AZ-2024-PRES"


def test_public_safe_stats_and_facets_exclude_withheld_source_url(
    db_session,
) -> None:
    _seed_search_rows(db_session)
    params = {"search": "hidden-only-token"}

    stats = read_workflow_stats(
        db_session,
        params,
        include_source_url_search=False,
    )
    facets = read_workflow_facets(
        db_session,
        params,
        include_source_url_search=False,
    )

    assert stats["total"] == 0
    assert stats["action_counts"] == {
        "blocked": 0,
        "ready_for_publication": 0,
        "published": 0,
    }
    assert _facet_total(facets) == 0


def test_public_runtime_opts_all_three_reads_into_safe_search_scope() -> None:
    tree = ast.parse(PUBLIC_RUNTIME.read_text(encoding="utf-8"))
    expected = {
        "read_workflow_items",
        "read_workflow_facets",
        "read_workflow_stats",
    }
    seen: dict[str, list[bool]] = {name: [] for name in expected}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in expected:
            continue

        value = None
        for keyword in node.keywords:
            if keyword.arg == "include_source_url_search":
                value = ast.literal_eval(keyword.value)
                break
        seen[node.func.id].append(value is False)

    assert set(seen) == expected
    for name, safe_calls in seen.items():
        assert safe_calls == [True], (
            f"{name} must have exactly one public call with "
            "include_source_url_search=False"
        )

from __future__ import annotations

import json

from unittest.mock import MagicMock

import pytest

from webapp.parser.Context_Integration.context_write_policy import (
    DEFAULT_CONTEXT_WRITE_POLICY,
    ContextWriteKind,
    ContextWritePolicy,
)


def test_default_policy_blocks_learned_and_canonical() -> None:
    assert DEFAULT_CONTEXT_WRITE_POLICY.permits(ContextWriteKind.NONE)
    assert DEFAULT_CONTEXT_WRITE_POLICY.permits(ContextWriteKind.RUNTIME)
    assert DEFAULT_CONTEXT_WRITE_POLICY.permits(ContextWriteKind.EVIDENCE)
    assert not DEFAULT_CONTEXT_WRITE_POLICY.permits(ContextWriteKind.LEARNED)
    assert not DEFAULT_CONTEXT_WRITE_POLICY.permits(ContextWriteKind.CANONICAL)


def test_custom_policy_can_allow_learned() -> None:
    policy = ContextWritePolicy(
        allow_runtime=True,
        allow_evidence=True,
        allow_learned=True,
        allow_canonical=False,
    )

    assert policy.permits(ContextWriteKind.LEARNED)
    assert not policy.permits(ContextWriteKind.CANONICAL)


def test_organize_and_enrich_does_not_persist_by_default() -> None:
    from webapp.parser.Context_Integration.context_coordinator import (
        ContextCoordinator,
    )

    coordinator = ContextCoordinator.__new__(ContextCoordinator)
    coordinator.organizer = MagicMock()
    coordinator.organizer.organize_context.return_value = {
        "organized": {"contests": []},
        "summary": {},
    }
    coordinator.organizer.apply_keyword_priority_hints.side_effect = lambda value: value

    coordinator._build_enrichment_plan = MagicMock(
        return_value={"routes": []}
    )
    coordinator._enrich_contests_with_nlp = MagicMock()
    coordinator._log_enrichment_snapshot = MagicMock()
    coordinator._persist_organized_context = MagicMock()
    coordinator.last_raw_context = None
    coordinator.organized = {}

    result = coordinator.organize_and_enrich({"source_type": "html"})

    assert result == {"contests": []}
    coordinator._persist_organized_context.assert_not_called()


def test_organize_and_enrich_records_explicit_evidence() -> None:
    from webapp.parser.Context_Integration.context_coordinator import (
        ContextCoordinator,
    )

    coordinator = ContextCoordinator.__new__(ContextCoordinator)
    coordinator.organizer = MagicMock()
    coordinator.organizer.organize_context.return_value = {
        "organized": {"contests": []},
        "summary": {},
    }
    coordinator.organizer.apply_keyword_priority_hints.side_effect = lambda value: value

    coordinator._build_enrichment_plan = MagicMock(
        return_value={"routes": []}
    )
    coordinator._enrich_contests_with_nlp = MagicMock()
    coordinator._log_enrichment_snapshot = MagicMock()
    coordinator._persist_organized_context = MagicMock(return_value=True)
    coordinator.last_raw_context = None
    coordinator.organized = {}

    coordinator.organize_and_enrich(
        {"source_type": "html"},
        write_kind=ContextWriteKind.EVIDENCE,
    )

    coordinator._persist_organized_context.assert_called_once()


def test_default_policy_rejects_canonical_write() -> None:
    from webapp.parser.Context_Integration.context_coordinator import (
        ContextCoordinator,
    )

    coordinator = ContextCoordinator.__new__(ContextCoordinator)
    coordinator.organizer = MagicMock()
    coordinator.organizer.organize_context.return_value = {
        "organized": {},
        "summary": {},
    }
    coordinator._build_enrichment_plan = MagicMock(
        return_value={"routes": []}
    )
    coordinator._enrich_contests_with_nlp = MagicMock()
    coordinator._log_enrichment_snapshot = MagicMock()
    coordinator.last_raw_context = None
    coordinator.organized = {}

    with pytest.raises(PermissionError):
        coordinator.organize_and_enrich(
            {},
            write_kind=ContextWriteKind.CANONICAL,
        )


def test_append_context_evidence_writes_reviewable_jsonl(tmp_path) -> None:
    from webapp.parser.Context_Integration.context_organizer import (
        ContextOrganizer,
    )

    organizer = ContextOrganizer.__new__(ContextOrganizer)
    evidence_path = tmp_path / "context_evidence.jsonl"

    success = organizer.append_context_evidence(
        {
            "metadata": {
                "state": "new_york",
                "county": "rockland",
                "confidence": 0.91,
            },
            "contests": [{"title": "Member of Assembly"}],
        },
        raw_context={
            "session_id": "session-123",
            "source_url": "https://example.test/results",
            "source_type": "html",
        },
        path=evidence_path,
    )

    assert success is True

    lines = evidence_path.read_bytes().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])

    assert record["type"] == "context_observation"
    assert record["status"] == "pending_review"
    assert record["source"] == "parser_enrichment"
    assert record["jurisdiction"] == {
        "state": "new_york",
        "county": "rockland",
    }
    assert record["provenance"]["session_id"] == "session-123"
    assert record["provenance"]["source_url"] == "https://example.test/results"
    assert record["observation"]["contests"][0]["title"] == "Member of Assembly"
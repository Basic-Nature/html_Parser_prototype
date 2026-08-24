"""C2G 2.0 process-local ephemeral inspection store contract."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from webapp.parser.services.ephemeral_pipeline_inspection import (
    ProcessLocalInspectionStore,
    ProcessLocalTopologyAttestation,
    STORE_AUTHORITY,
)
from webapp.parser.services.pipeline_inspection import (
    INSPECTION_AUTHORITY,
    INSPECTION_CONTRACT,
)


@dataclass
class FakeClock:
    value: float = 1000.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _topology(
    *,
    instances: int = 1,
    workers: int = 1,
) -> ProcessLocalTopologyAttestation:
    return ProcessLocalTopologyAttestation(
        app_service_instance_capacity=instances,
        gunicorn_workers=workers,
        evidence_ref="C2G_1_9:test",
    )


def _payload() -> dict:
    return {
        "contract": INSPECTION_CONTRACT,
        "authority": {
            "inspection": INSPECTION_AUTHORITY,
            "canonical": False,
            "write_kind": "none",
        },
        "stage": "interpreted",
        "source_provenance": {
            "source_type": "csv",
            "source_sha256": "a" * 64,
            "artifact_id": None,
            "evidence_ref": "fixture://store",
            "location": None,
            "source_uri_included": False,
            "source_metadata_included": False,
        },
        "summary": {
            "header_count": 1,
            "row_count": 3,
            "transformation_count": 1,
            "warning_count": 0,
        },
        "completeness": {
            "state": "unknown",
            "expected_count": None,
            "observed_count": None,
            "missing_count": None,
            "null_value_count": None,
            "is_complete": None,
            "notes": [],
        },
        "transformations": [
            {
                "sequence": 0,
                "from_stage": "interpreted",
                "to_stage": "interpreted",
                "operation": "vote_method_header_canonicalization",
                "rule_source": (
                    "Context_Integration.Context_Library.constants."
                    "BALLOT_NAME_CANON_MAP"
                ),
                "confidence": None,
                "evidence_refs": [],
                "details": {
                    "before_header": "election day",
                    "after_header": "Election Day",
                    "vote_value_mutation": False,
                    "unknown_example": None,
                    "confirmed_zero_example": 0,
                    "signed_example": -4,
                },
            }
        ],
        "warnings": [],
        "rows_included": False,
        "headers_included": False,
        "automatic_timestamp": False,
    }


def test_store_authority_is_explicitly_noncanonical_and_process_local() -> None:
    assert STORE_AUTHORITY == "noncanonical_process_local_ephemeral_evidence"

    topology = _topology()
    assert topology.process_local_safe is True


@pytest.mark.parametrize(
    ("instances", "workers"),
    [(2, 1), (1, 2), (2, 2), (0, 1), (1, 0)],
)
def test_store_refuses_non_single_process_topology(
    instances: int,
    workers: int,
) -> None:
    with pytest.raises(RuntimeError):
        ProcessLocalInspectionStore(
            topology=_topology(instances=instances, workers=workers),
        )


def test_store_requires_topology_evidence_ref() -> None:
    with pytest.raises(ValueError):
        ProcessLocalInspectionStore(
            topology=ProcessLocalTopologyAttestation(
                app_service_instance_capacity=1,
                gunicorn_workers=1,
                evidence_ref="",
            ),
        )


def test_put_get_preserves_null_zero_and_signed_evidence() -> None:
    store = ProcessLocalInspectionStore(topology=_topology())
    payload = _payload()

    store.put(
        session_id="session-1",
        principal="reviewer@example.test",
        payload=payload,
    )

    loaded = store.get(
        session_id="session-1",
        principal="reviewer@example.test",
    )

    assert loaded is not None
    details = loaded["transformations"][0]["details"]
    assert details["unknown_example"] is None
    assert details["confirmed_zero_example"] == 0
    assert details["signed_example"] == -4


def test_store_isolates_same_session_from_wrong_principal() -> None:
    store = ProcessLocalInspectionStore(topology=_topology())
    store.put(
        session_id="session-1",
        principal="principal-a",
        payload=_payload(),
    )

    assert (
        store.get(
            session_id="session-1",
            principal="principal-b",
        )
        is None
    )


def test_store_isolates_different_session() -> None:
    store = ProcessLocalInspectionStore(topology=_topology())
    store.put(
        session_id="session-1",
        principal="principal-a",
        payload=_payload(),
    )

    assert (
        store.get(
            session_id="session-2",
            principal="principal-a",
        )
        is None
    )


def test_store_copy_on_write_prevents_caller_mutation() -> None:
    store = ProcessLocalInspectionStore(topology=_topology())
    payload = _payload()

    store.put(
        session_id="session-1",
        principal="principal-a",
        payload=payload,
    )
    payload["transformations"][0]["details"]["after_header"] = "MUTATED"

    loaded = store.get(
        session_id="session-1",
        principal="principal-a",
    )
    assert loaded["transformations"][0]["details"]["after_header"] == "Election Day"


def test_store_copy_on_read_prevents_retrieval_mutation() -> None:
    store = ProcessLocalInspectionStore(topology=_topology())
    store.put(
        session_id="session-1",
        principal="principal-a",
        payload=_payload(),
    )

    first = store.get(
        session_id="session-1",
        principal="principal-a",
    )
    first["transformations"][0]["details"]["after_header"] = "MUTATED"

    second = store.get(
        session_id="session-1",
        principal="principal-a",
    )
    assert second["transformations"][0]["details"]["after_header"] == "Election Day"


def test_store_expires_at_ttl_boundary() -> None:
    clock = FakeClock()
    store = ProcessLocalInspectionStore(
        topology=_topology(),
        ttl_seconds=30,
        clock=clock,
    )
    store.put(
        session_id="session-1",
        principal="principal-a",
        payload=_payload(),
    )

    clock.advance(29.999)
    assert store.get(session_id="session-1", principal="principal-a") is not None

    clock.advance(0.001)
    assert store.get(session_id="session-1", principal="principal-a") is None


def test_store_max_entry_bound_evicts_oldest_session() -> None:
    clock = FakeClock()
    store = ProcessLocalInspectionStore(
        topology=_topology(),
        max_entries=2,
        clock=clock,
    )

    for index in range(3):
        store.put(
            session_id=f"session-{index}",
            principal="principal-a",
            payload=_payload(),
        )
        clock.advance(1)

    assert store.size() == 2
    assert store.get(session_id="session-0", principal="principal-a") is None
    assert store.get(session_id="session-1", principal="principal-a") is not None
    assert store.get(session_id="session-2", principal="principal-a") is not None


def test_store_rejects_canonical_payload() -> None:
    payload = _payload()
    payload["authority"]["canonical"] = True

    store = ProcessLocalInspectionStore(topology=_topology())

    with pytest.raises(ValueError):
        store.put(
            session_id="session-1",
            principal="principal-a",
            payload=payload,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update({"rows": [{"votes": 1}]}),
        lambda payload: payload.update({"rows_included": True}),
        lambda payload: payload.update({"headers": ["Candidate"]}),
        lambda payload: payload.update({"headers_included": True}),
        lambda payload: payload["source_provenance"].update(
            {"source_uri_included": True}
        ),
        lambda payload: payload["source_provenance"].update(
            {"source_uri": "https://example.test/private"}
        ),
        lambda payload: payload["source_provenance"].update(
            {"source_metadata_included": True}
        ),
        lambda payload: payload["source_provenance"].update(
            {"metadata": {"secret": "not-allowed"}}
        ),
    ],
)
def test_store_rejects_forbidden_exposure_fields(mutation) -> None:
    payload = _payload()
    mutation(payload)

    store = ProcessLocalInspectionStore(topology=_topology())

    with pytest.raises(ValueError):
        store.put(
            session_id="session-1",
            principal="principal-a",
            payload=payload,
        )


def test_store_delete_requires_matching_principal() -> None:
    store = ProcessLocalInspectionStore(topology=_topology())
    store.put(
        session_id="session-1",
        principal="principal-a",
        payload=_payload(),
    )

    assert store.delete(session_id="session-1", principal="principal-b") is False
    assert store.delete(session_id="session-1", principal="principal-a") is True
    assert store.get(session_id="session-1", principal="principal-a") is None


def test_store_has_no_global_singleton_side_effect() -> None:
    import webapp.parser.services.ephemeral_pipeline_inspection as module

    assert not hasattr(module, "inspection_store")
    assert not hasattr(module, "store")
    assert not hasattr(module, "default_store")
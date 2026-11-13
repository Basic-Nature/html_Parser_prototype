"""P6C5B2B shadow-only bridge between the public runtime and the B1 broker.

No Socket.IO routes, browser activation, or worker registration occur here.
Actual runtime wiring must be separately qualified and explicitly authorized.
"""
from __future__ import annotations
from typing import Any


def bind_compact_broker_to_runtime(runtime: Any, broker: Any, server_peer: Any) -> None:
    """Install a server-provided capture callback for an approved CSV run."""
    source = runtime.source_projection
    scope = {
        "state": str(source["state"]),
        "year": str(source["year"]),
        "contest": str(source["contest"]),
        "registry_source_id": str(runtime.registry_source_id),
    }

    def capture(*, headers: list[str], rows: list[dict[str, object]],
                source_sha256: str, quarantine: dict[str, object]) -> dict[str, object]:
        event = broker.begin(
            server_peer, scope=scope, source_sha256=source_sha256,
            headers=headers, rows=rows, quarantine=quarantine,
        )
        # Only server-finalized output is staged; failure of optional CSV
        # delivery must never turn a valid precinct page run into a false PASS.
        from .public_csv_delivery import feature_enabled
        if feature_enabled():
            from .public_compact_csv_handoff import STAGED_COMPACT_CSV
            try:
                STAGED_COMPACT_CSV.stage(
                    snapshot_id=event['manifest']['snapshot_id'],
                    session_id=server_peer.server_session_id,
                    source_id=server_peer.registry_source_id,
                    source_sha256=source_sha256,
                    headers=headers, rows=rows,
                )
            except (ValueError, RuntimeError):
                # The run remains a valid partial memory-only preview, but no
                # CSV capability is minted for this snapshot.
                pass
        return event

    runtime.enable_compact_capture(capture)

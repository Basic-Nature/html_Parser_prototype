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
        return broker.begin(
            server_peer, scope=scope, source_sha256=source_sha256,
            headers=headers, rows=rows, quarantine=quarantine,
        )

    runtime.enable_compact_capture(capture)

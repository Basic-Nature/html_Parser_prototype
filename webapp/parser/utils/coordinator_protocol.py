from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@runtime_checkable
class CoordinatorProtocol(Protocol):
    """Minimal protocol describing the Coordinator API relied on by table_builder.

    The real ContextCoordinator exposes a much larger surface area, but the
    builder pipeline only needs lightweight scoring and entity extraction hooks.
    Tests provide dummy coordinators that satisfy this contract.
    """

    def score_header(self, header: str, context: Mapping[str, Any] | None = None) -> float:
        ...

    def extract_entities(self, text: str) -> Sequence[tuple[Any, Any]]:
        ...

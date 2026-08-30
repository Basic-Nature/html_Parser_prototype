from __future__ import annotations

import threading
from typing import Any, Callable

from webapp.parser.utils.logger_singleton import logger

MonitorCallable = Callable[..., None]


class AlertMonitorService:
    """Process-local owner for the Alert-table polling thread."""

    def __init__(
        self,
        *,
        poll_interval: float = 10.0,
        monitor_callable: MonitorCallable | None = None,
    ) -> None:
        self.poll_interval = max(float(poll_interval), 0.01)
        self._monitor_callable = monitor_callable
        self._lock = threading.RLock()
        self.stop_event = threading.Event()
        self.state: dict[str, Any] = {}
        self.thread: threading.Thread | None = None

    def _resolve_monitor(self) -> MonitorCallable:
        if self._monitor_callable is not None:
            return self._monitor_callable

        # Lazy by design: constructing/importing this service does not import
        # the integrity/ML stack. The real monitor is resolved only inside the
        # started worker thread.
        from webapp.parser.Context_Integration.Integrity_check import (
            monitor_db_for_alerts,
        )
        return monitor_db_for_alerts

    def _run(self) -> None:
        try:
            self._resolve_monitor()(
                poll_interval=self.poll_interval,
                stop_event=self.stop_event,
                state=self.state,
            )
        except Exception as exc:
            self.state.update(
                {
                    "running": False,
                    "db_available": False,
                    "last_failure_stage": "service_uncaught_exception",
                    "last_error_type": type(exc).__name__,
                    "last_error_message": str(exc),
                }
            )
            logger.error(
                "[ALERT MONITOR SERVICE] Uncaught monitor exception: %s",
                exc,
                exc_info=True,
            )

    def start(self) -> threading.Thread:
        """Start once per process; repeated calls reuse the live thread."""
        with self._lock:
            if self.thread is not None and self.thread.is_alive():
                return self.thread

            self.stop_event = threading.Event()
            self.state = {}
            thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="electionpulse-alert-monitor",
            )
            self.thread = thread
            thread.start()
            return thread

    def stop(self, timeout: float = 2.0) -> bool:
        """Signal the owned poller and join it briefly."""
        with self._lock:
            thread = self.thread
            self.stop_event.set()

        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=max(float(timeout), 0.0))

        stopped = not (thread is not None and thread.is_alive())
        if stopped:
            with self._lock:
                if self.thread is thread:
                    self.thread = None
        return stopped

    def health(self) -> dict[str, Any]:
        with self._lock:
            thread = self.thread
            return {
                **dict(self.state),
                "thread_alive": bool(thread and thread.is_alive()),
                "stop_requested": self.stop_event.is_set(),
                "poll_interval": self.poll_interval,
            }

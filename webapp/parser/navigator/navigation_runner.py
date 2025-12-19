from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
import threading
from typing import Any, Dict, List, Optional

from ..utils.logger_singleton import logger
from ..utils.browser_utils import autoscroll_until_stable
from ..utils.html_scanner import scan_html_for_context
from .keyword_bias import load_keyword_bias
from .navigation_recipes import NavigationRecipeStore, DEFAULT_RECIPE_PATH


DEFAULT_BIAS_CUTOFF = 0.55


@dataclass
class NavigationResult:
    executed: bool
    script_id: Optional[str] = None
    context_updates: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    telemetry: Optional[List[Dict[str, Any]]] = None


class NavigationInstructionRunner:
    """Executes context-aware navigation scripts loaded from orjson recipes."""

    def __init__(
        self,
        store: NavigationRecipeStore | None = None,
        *,
        autoscroll_fn=autoscroll_until_stable,
        scan_fn=scan_html_for_context,
        max_parallel_workers: int = 4,
    ) -> None:
        self.store = store or NavigationRecipeStore(DEFAULT_RECIPE_PATH)
        self.autoscroll_fn = autoscroll_fn
        self.scan_fn = scan_fn
        self.max_parallel_workers = max_parallel_workers
        self._context_cache: Dict[str, Any] = {}
        self._page_lock = threading.RLock()
        self._trace_lock = threading.RLock()

    def run(
        self,
        page,
        *,
        context: Dict[str, Any] | None = None,
        coordinator=None,
        session_id: str | None = None,
    ) -> NavigationResult:
        context = dict(context or {})
        target_url = context.get("url") or getattr(page, "url", None)
        candidates = self.store.candidates_for(context)
        trace: List[Dict[str, Any]] = []
        if not candidates:
            return NavigationResult(False, context_updates={}, telemetry=trace)

        self._apply_keyword_bias(page, context, session_id, trace)

        for script in candidates:
            if not self._script_matches(script, page, target_url):
                continue
            result = self._execute_script(script, page, context, coordinator, session_id, trace)
            metadata = {}
            if script.get("description"):
                metadata["description"] = script["description"]
            return NavigationResult(True, script.get("id"), result, metadata or None, trace or None)
        return NavigationResult(False, context_updates={}, telemetry=trace or None)

    def _script_matches(self, script, page, target_url) -> bool:
        match = script.get("match") or {}
        if not isinstance(match, dict):
            return True
        url_contains: List[str] = [s for s in match.get("url_contains", []) if isinstance(s, str)]
        if url_contains and target_url:
            lowered = target_url.lower()
            if not any(substr.lower() in lowered for substr in url_contains):
                return False
        dom_markers: List[str] = [s for s in match.get("dom_markers", []) if isinstance(s, str)]
        if dom_markers:
            try:
                html_source = page.content()
            except Exception:
                html_source = ""
            lowered_html = html_source.lower() if html_source else ""
            if not lowered_html:
                return False
            if not all(marker.lower() in lowered_html for marker in dom_markers):
                return False
        return True

    def _execute_script(self, script, page, context, coordinator, session_id, trace):
        context_updates: Dict[str, Any] = {}
        for step in script.get("steps", []):
            updates = self._execute_step(step, page, context, coordinator, session_id, trace)
            if updates:
                context_updates.update(updates)
        post_context = script.get("post_context")
        if isinstance(post_context, dict):
            context_updates.update(post_context)
        return context_updates

    def _execute_step(self, step, page, context, coordinator, session_id, trace):
        action = (step or {}).get("action")
        if not isinstance(action, str):
            return None
        action = action.lower()
        try:
            if action == "wait_for_selector":
                selector = step.get("selector")
                timeout = step.get("timeout_ms")
                if selector:
                    with self._page_lock:
                        page.wait_for_selector(selector, timeout=timeout)
                    self._record_trace(trace, action, "ok", selector=selector, timeout_ms=timeout)
            elif action == "wait_for_load_state":
                state = step.get("state") or "networkidle"
                timeout = step.get("timeout_ms")
                with self._page_lock:
                    page.wait_for_load_state(state=state, timeout=timeout)
                self._record_trace(trace, action, "ok", state=state, timeout_ms=timeout)
            elif action == "click":
                selector = step.get("selector")
                if selector:
                    with self._page_lock:
                        page.click(selector, timeout=step.get("timeout_ms"))
                    wait_after = step.get("wait_after_ms")
                    if wait_after:
                        with self._page_lock:
                            page.wait_for_timeout(wait_after)
                    self._record_trace(trace, action, "ok", selector=selector, wait_after_ms=wait_after)
            elif action == "wait":
                timeout = step.get("timeout_ms", 0)
                if timeout:
                    with self._page_lock:
                        page.wait_for_timeout(timeout)
                    self._record_trace(trace, action, "ok", timeout_ms=timeout)
            elif action == "fill":
                selector = step.get("selector")
                value = step.get("value", "")
                if selector:
                    with self._page_lock:
                        page.fill(selector, value)
                    self._record_trace(trace, action, "ok", selector=selector, value=value)
            elif action == "run_js":
                expression = step.get("expression")
                if expression:
                    with self._page_lock:
                        page.evaluate(expression)
                    self._record_trace(trace, action, "ok", expression=expression)
            elif action == "autoscroll":
                max_time = step.get("max_time_ms")
                metrics: Dict[str, Any] = {}
                with self._page_lock:
                    self.autoscroll_fn(
                        page,
                        max_total_time=max_time,
                        session_id=session_id,
                        metrics=metrics,
                    )
                allowed_keys = {
                    k: v
                    for k, v in (metrics or {}).items()
                    if k in {"scroll_attempts", "tables_seen", "elapsed_ms", "selector_hits", "no_new_tables_iters", "stable_frames", "scroll_depth"}
                }
                self._record_trace(trace, action, "ok", max_time_ms=max_time, **allowed_keys)
            elif action == "scan_context":
                scan_kwargs = step.get("kwargs") or {}
                with self._page_lock:
                    result = self.scan_fn(
                        target_url=context.get("url") or getattr(page, "url", None),
                        page=page,
                        coordinator=coordinator,
                        session_id=session_id,
                        context_cache=self._context_cache,
                        **scan_kwargs,
                    )
                projected = self._project_scan_result(result, step)
                summary_keys = list((projected or {}).keys())
                self._record_trace(trace, action, "ok", projected_keys=summary_keys)
                return projected
            elif action == "set_context":
                values = step.get("values")
                if isinstance(values, dict):
                    self._record_trace(trace, action, "ok", keys=list(values.keys()))
                    return values
            elif action == "parallel":
                updates = self._run_parallel(step.get("steps") or [], page, context, coordinator, session_id, trace)
                merged_keys = list((updates or {}).keys())
                self._record_trace(trace, action, "ok", step_count=len(step.get("steps") or []), merged_keys=merged_keys)
                return updates
            else:
                logger.debug({
                    "level": "DEBUG",
                    "type": "navigation",
                    "message": f"Unknown navigation action: {action}",
                })
                self._record_trace(trace, action, "skipped", reason="unknown_action")
        except Exception as exc:
            logger.warning({
                "level": "WARNING",
                "type": "navigation",
                "message": f"Navigation step '{action}' failed: {exc}",
            })
            self._record_trace(trace, action, "error", error=str(exc))
        return None

    def _run_parallel(self, steps, page, context, coordinator, session_id, trace):
        if not steps:
            return None
        outputs: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(len(steps), self.max_parallel_workers)) as executor:
            futures = [
                executor.submit(self._execute_step, step, page, context, coordinator, session_id, trace)
                for step in steps
            ]
            wait(futures)
            for future in futures:
                try:
                    result = future.result()
                except Exception:
                    result = None
                if result:
                    outputs.append(result)
        merged: Dict[str, Any] = {}
        for update in outputs:
            merged.update(update)
        return merged or None

    def _project_scan_result(self, scan_result, step):
        if not isinstance(scan_result, dict):
            return None
        projection = step.get("project")
        if projection and isinstance(projection, dict):
            updates: Dict[str, Any] = {}
            for target, source in projection.items():
                value = self._extract_path(scan_result, source)
                if value is not None:
                    updates[target] = value
            return updates or None
        store_key = step.get("store_key") or "scan_context"
        return {store_key: scan_result}

    @staticmethod
    def _extract_path(payload: Dict[str, Any], path: str):
        if not path:
            return None
        current: Any = payload
        for segment in path.split('.'):
            if isinstance(current, dict):
                current = current.get(segment)
            else:
                return None
            if current is None:
                return None
        return current

    def _record_trace(self, trace, action: str, status: str, **details):
        if trace is None:
            return
        entry: Dict[str, Any] = {"action": action, "status": status}
        if details:
            entry["details"] = details
        with self._trace_lock:
            trace.append(entry)

    def _apply_keyword_bias(self, page, context, session_id, trace) -> None:
        bias_entries = load_keyword_bias()
        if not bias_entries:
            return
        bias_cutoff = float(context.get("navigation_bias_threshold", DEFAULT_BIAS_CUTOFF) or DEFAULT_BIAS_CUTOFF)
        try:
            html_lower = (page.content() or "").lower()
        except Exception:
            return
        seen_selectors = set()
        for entry in bias_entries:
            selector = entry.get("selector")
            phrases = entry.get("phrases") or []
            confidence = entry.get("confidence", 0.0) or 0.0
            max_wait_ms = entry.get("max_wait_ms")
            autoscroll_ms = entry.get("autoscroll_ms")
            if not selector or selector in seen_selectors:
                continue
            if confidence < bias_cutoff:
                continue
            if not any(p in html_lower for p in phrases):
                continue
            seen_selectors.add(selector)
            try:
                with self._page_lock:
                    handle = page.query_selector(selector)
                    if handle:
                        handle.click()
                        if max_wait_ms:
                            page.wait_for_timeout(max_wait_ms)
                        if autoscroll_ms:
                            self.autoscroll_fn(
                                page,
                                max_total_time=autoscroll_ms,
                                session_id=session_id,
                            )
                        self._record_trace(
                            trace,
                            "keyword_bias",
                            "ok",
                            selector=selector,
                            phrases=phrases,
                            confidence=confidence,
                        )
                    else:
                        self._record_trace(
                            trace,
                            "keyword_bias",
                            "skipped",
                            selector=selector,
                            reason="selector_not_found",
                        )
            except Exception as exc:
                self._record_trace(
                    trace,
                    "keyword_bias",
                    "error",
                    selector=selector,
                    error=str(exc),
                )


__all__ = ["NavigationInstructionRunner", "NavigationResult"]

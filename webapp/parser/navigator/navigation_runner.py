from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..utils.browser_utils import (
    SCROLL_METRIC_KEYS,
    autoscroll_until_stable,
    safe_click_with_retry,
    safe_get_attribute,
    safe_inner_text,
)
from ..utils.html_scanner import scan_html_for_context
from ..utils.logger_singleton import logger
from .keyword_bias import load_keyword_bias
from .navigation_recipes import DEFAULT_RECIPE_PATH, NavigationRecipeStore

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
                optional = bool(step.get("optional"))
                if selector:
                    if self._is_enhanced_voting_page(page, target_url=context.get("url") or getattr(page, "url", None)) and self._should_soft_skip_selector_failure(selector):
                        self._record_trace(
                            trace,
                            action,
                            "skipped",
                            selector=selector,
                            reason="enhanced_voting_generic_selector",
                        )
                        return None
                    with self._page_lock:
                        wait_ok = False
                        last_wait_error = None
                        for candidate in self._selector_candidates(selector):
                            try:
                                page.wait_for_selector(candidate, timeout=timeout)
                                wait_ok = True
                                break
                            except Exception as wait_exc:
                                last_wait_error = wait_exc
                        if not wait_ok:
                            if optional:
                                self._record_trace(trace, action, "skipped", selector=selector, reason="optional_step_not_found")
                                return None
                            if self._should_soft_skip_selector_failure(selector) and self._has_results_ready(page):
                                self._record_trace(trace, action, "skipped", selector=selector, reason="results_already_ready")
                                return None
                            raise RuntimeError(last_wait_error or f"wait_for_selector failed for {selector}")
                    self._record_trace(trace, action, "ok", selector=selector, timeout_ms=timeout)
            elif action == "wait_for_load_state":
                state = step.get("state") or "networkidle"
                timeout = step.get("timeout_ms")
                with self._page_lock:
                    page.wait_for_load_state(state=state, timeout=timeout)
                self._record_trace(trace, action, "ok", state=state, timeout_ms=timeout)
            elif action == "click":
                selector = step.get("selector")
                optional = bool(step.get("optional"))
                if selector:
                    if self._is_enhanced_voting_page(page, target_url=context.get("url") or getattr(page, "url", None)) and self._should_soft_skip_selector_failure(selector):
                        self._record_trace(
                            trace,
                            action,
                            "skipped",
                            selector=selector,
                            reason="enhanced_voting_generic_selector",
                        )
                        return None
                    with self._page_lock:
                        click_ok = False
                        for candidate in self._selector_candidates(selector):
                            click_ok = safe_click_with_retry(
                                page=page,
                                selector=candidate,
                                max_retries=4,
                                timeout=step.get("timeout_ms") or 8000,
                                delay_between=0.25,
                                scroll_into_view=True,
                                soft_fail=optional,
                                session_id=session_id,
                                logger=logger,
                            )
                            if click_ok:
                                break
                        if not click_ok:
                            click_ok = self._click_by_text_discovery(page, selector, session_id)
                        if not click_ok:
                            if optional:
                                self._record_trace(trace, action, "skipped", selector=selector, reason="optional_step_not_found")
                                return None
                            if self._should_soft_skip_selector_failure(selector) and self._has_results_ready(page):
                                self._record_trace(trace, action, "skipped", selector=selector, reason="results_already_ready")
                                return None
                            raise RuntimeError(f"click failed for selector: {selector}")
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
                    if k in SCROLL_METRIC_KEYS
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

    def _selector_candidates(self, selector: str) -> List[str]:
        candidates = [selector]
        lowered = selector.lower()
        if "county" in lowered:
            candidates.extend([
                "button:has-text('County'), [role='tab']:has-text('County'), a:has-text('County'), [aria-label*='County' i], [title*='County' i]",
                "text=/.*County.*/i",
            ])
        if "precinct" in lowered:
            candidates.extend([
                "button:has-text('Precinct'), [role='tab']:has-text('Precinct'), a:has-text('Precinct'), [aria-label*='Precinct' i], [title*='Precinct' i]",
                "text=/.*Precinct.*/i",
            ])
        seen = set()
        deduped: List[str] = []
        for cand in candidates:
            if cand and cand not in seen:
                seen.add(cand)
                deduped.append(cand)
        return deduped

    @staticmethod
    def _should_soft_skip_selector_failure(selector: str) -> bool:
        lowered = (selector or "").lower()
        return ("county" in lowered) or ("precinct" in lowered)

    def _is_enhanced_voting_page(self, page, target_url: Optional[str] = None) -> bool:
        try:
            url = str(target_url or getattr(page, "url", None) or "")
        except Exception:
            url = ""
        lowered_url = url.lower()
        if any(marker in lowered_url for marker in ["enhancedvoting", "enhanced-voting", "enhanced voting", "rockland"]):
            return True

        try:
            html_source = page.content() or ""
            lowered_html = html_source.lower()
            if any(marker in lowered_html for marker in [
                "view results by election district",
                "results by election district",
                "vote method",
                "enhanced voting",
            ]):
                return True
        except Exception:
            pass

        return False

    def _has_results_ready(self, page) -> bool:
        try:
            checks = [
                "table, [role='table']",
                "a[href*='export'], a[href$='.csv'], a[href$='.json'], a[href$='.pdf'], a[href$='.xlsx'], a[href$='.xls']",
                "button:has-text('Download'), a:has-text('Download')",
            ]
            for selector in checks:
                try:
                    loc = page.locator(selector)
                    if hasattr(loc, "count") and loc.count() > 0:
                        return True
                except Exception:
                    continue
            return False
        except Exception:
            return False

    def _click_by_text_discovery(self, page, selector: str, session_id: Optional[str]) -> bool:
        lowered = (selector or "").lower()
        tokens: List[str] = []
        if "county" in lowered:
            tokens.append("county")
        if "precinct" in lowered:
            tokens.append("precinct")
        if not tokens:
            return False
        try:
            elements = page.query_selector_all("button, [role='tab'], a, [aria-label], [title]") or []
        except Exception:
            elements = []
        for element in elements:
            try:
                text_value = (safe_inner_text(element, logger) or "").strip().lower()
                aria_value = (safe_get_attribute(element, "aria-label", logger) or "").strip().lower()
                title_value = (safe_get_attribute(element, "title", logger) or "").strip().lower()
                haystack = " ".join([text_value, aria_value, title_value])
                if any(token in haystack for token in tokens):
                    if safe_click_with_retry(
                        page=page,
                        element=element,
                        max_retries=3,
                        timeout=7000,
                        delay_between=0.2,
                        scroll_into_view=True,
                        session_id=session_id,
                        logger=logger,
                    ):
                        return True
            except Exception:
                continue
        return False

    def _run_parallel(self, steps, page, context, coordinator, session_id, trace):
        if not steps:
            return None
        outputs: List[Dict[str, Any]] = []
        for step in steps:
            try:
                result = self._execute_step(step, page, context, coordinator, session_id, trace)
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
        bias_cutoff = float(context.get("navigation_bias_threshold", DEFAULT_BIAS_CUTOFF))
        try:
            html_lower = (page.content() or "").lower()
        except Exception:
            return
        seen_selectors = set()
        for entry in bias_entries:
            selector = entry.get("selector")
            phrases = entry.get("phrases") or []
            confidence = float(entry.get("confidence", 0.0))
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
                        clicked = safe_click_with_retry(
                            page=page,
                            element=handle,
                            max_retries=4,
                            timeout=max_wait_ms or 8000,
                            delay_between=0.25,
                            scroll_into_view=True,
                            session_id=session_id,
                            logger=logger,
                        )
                        if not clicked:
                            raise RuntimeError(f"keyword bias click failed: {selector}")
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

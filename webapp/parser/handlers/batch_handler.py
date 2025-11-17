from __future__ import annotations

import copy
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..utils.logger_singleton import logger, prompt
from ..utils.shared_logic import safe_get, safe_lower, safe_parse, safe_strip
from ..utils.user_prompt import PromptCancelled


def _normalize_label(value: Optional[str]) -> str:
    """Lightweight normalization for contest labels used in matching."""
    if value is None:
        return ""
    try:
        return safe_strip(safe_lower(value))
    except Exception:
        return str(value).strip().lower()


class BatchProcessor:
    """Coordinates batch execution for multiple contest selections."""

    def __init__(
        self,
        *,
        coordinator: Any,
        handler: Any,
        page: Any,
        base_context: Optional[dict],
        selected_races: Sequence[Any],
        initial_result: Optional[Tuple[List[str], List[Dict[str, Any]], str, Dict[str, Any]]],
        session_id: Optional[str],
        target_url: str,
        output_dir: str,
        processed_info: Any,
        ai_analyze_results: Callable[[List[str], List[Dict[str, Any]], str, Dict[str, Any]], None],
        stream_results: Callable[[List[str], List[Dict[str, Any]], str, Dict[str, Any]], None],
        mark_url_processed: Callable[..., None],
        max_workers: int = 2,
    ) -> None:
        self.coordinator = coordinator
        self.handler = handler
        self.page = page
        self.session_id = session_id
        self.target_url = target_url
        self.output_dir = output_dir
        self.processed_info = processed_info
        self.ai_analyze_results = ai_analyze_results
        self.stream_results = stream_results
        self.mark_url_processed = mark_url_processed
        self.max_workers = max(1, int(max_workers or 1))
        self.batch_id = str(uuid.uuid4())
        self.prompt = prompt
        self.initial_result = (
            initial_result if isinstance(initial_result, tuple) and len(initial_result) == 4 else None
        )

        self.original_races = self._prepare_races(selected_races)
        self.pending_races = [copy.deepcopy(r) for r in self.original_races]
        self._initial_match_index = self._find_initial_match_index()

        original_count = len(self.original_races)
        if original_count:
            if self.initial_result and self._initial_match_index is None:
                self.total_expected = original_count + 1
            else:
                self.total_expected = original_count
        else:
            self.total_expected = 1 if self.initial_result else 0

        self.results: List[Tuple[List[str], List[Dict[str, Any]], str, Dict[str, Any]]] = []
        self._analysis_executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=self.max_workers) if self.max_workers > 1 else None
        )
        self._analysis_futures: List[Future] = []

        if isinstance(base_context, dict):
            try:
                self.base_context = copy.deepcopy(base_context)
            except Exception:
                self.base_context = dict(base_context)
        else:
            self.base_context = {}
        # Ensure batch metadata is discoverable for downstream consumers
        self.base_context.setdefault("batch_mode", True)
        self.base_context.setdefault("selected_races", self.original_races)

    def run(self) -> List[Tuple[List[str], List[Dict[str, Any]], str, Dict[str, Any]]]:
        start = time.time()
        success_count = 0
        failure_count = 0
        processed_counter = 0

        if self.prompt and self.session_id:
            try:
                self.prompt.clear_queued_responses(self.session_id)
            except Exception:
                pass

        logger.info({
            "level": "INFO",
            "type": "batch",
            "message": f"[Batch] Starting batch run with {self.total_expected} target(s).",
            "session_id": self.session_id,
            "url": self.target_url,
            "batch_id": self.batch_id,
            "races": [self._race_label(r) for r in self.original_races],
        })

        if self.initial_result:
            race = None
            if self._initial_match_index is not None and 0 <= self._initial_match_index < len(self.pending_races):
                race = self.pending_races.pop(self._initial_match_index)
            if self._emit_result(self.initial_result, race, index=1, initial=True):
                success_count += 1
            else:
                failure_count += 1
            processed_counter += 1

        for idx, race in enumerate(self.pending_races, start=processed_counter + 1):
            if not race:
                failure_count += 1
                continue
            try:
                if self._process_single_race(race, idx):
                    success_count += 1
                else:
                    failure_count += 1
            except PromptCancelled:
                logger.warning({
                    "level": "WARNING",
                    "type": "batch",
                    "message": "[Batch] Prompt cancelled by user; skipping remaining contests.",
                    "session_id": self.session_id,
                    "url": self.target_url,
                })
                failure_count += 1
                break
            except Exception as exc:
                failure_count += 1
                logger.error({
                    "level": "ERROR",
                    "type": "batch",
                    "message": f"[Batch] Contest processing failed: {exc}",
                    "session_id": self.session_id,
                    "url": self.target_url,
                }, exc_info=True)

        self._await_postprocessing()

        status = self._compute_status(success_count, failure_count)
        duration = round(time.time() - start, 3)
        self._mark_processed(status, success_count, failure_count, duration)

        logger.info({
            "level": "INFO",
            "type": "batch",
            "message": f"[Batch] Completed with status={status} (success={success_count}, failures={failure_count}).",
            "session_id": self.session_id,
            "url": self.target_url,
            "batch_id": self.batch_id,
            "duration_sec": duration,
        })
        return self.results

    # --- Internal helpers ---

    def _prepare_races(self, races: Sequence[Any]) -> List[Dict[str, Any]]:
        prepared: List[Dict[str, Any]] = []
        for race in races or []:
            if isinstance(race, dict):
                try:
                    prepared.append(copy.deepcopy(race))
                except Exception:
                    prepared.append(dict(race))
            else:
                prepared.append({"value": race})
        return prepared

    def _find_initial_match_index(self) -> Optional[int]:
        if not self.initial_result:
            return None
        _, _, contest, _ = self.initial_result
        contest_norm = _normalize_label(contest)
        if not contest_norm:
            return None
        for idx, race in enumerate(self.pending_races):
            label = self._race_label(race)
            if _normalize_label(label) == contest_norm:
                return idx
        return None

    def _process_single_race(self, race: Dict[str, Any], batch_index: int) -> bool:
        label = self._race_label(race) or f"#{batch_index}"
        logger.info({
            "level": "INFO",
            "type": "batch",
            "message": f"[Batch] Processing contest '{label}' ({batch_index}/{max(self.total_expected, batch_index)}).",
            "session_id": self.session_id,
            "url": self.target_url,
        })
        self._queue_prompt_responses(race)
        context = self._build_context(race, batch_index)

        result = safe_parse(
            self.handler,
            self.page,
            self.coordinator,
            context,
            session_id=self.session_id,
            logger=logger,
        )

        if not isinstance(result, tuple) or len(result) != 4:
            logger.error({
                "level": "ERROR",
                "type": "batch",
                "message": f"[Batch] Handler returned invalid result for '{label}'.",
                "session_id": self.session_id,
                "url": self.target_url,
            })
            return False

        return self._emit_result(result, race, index=batch_index)

    def _build_context(self, race: Dict[str, Any], batch_index: int) -> Dict[str, Any]:
        context = {}
        if isinstance(self.base_context, dict):
            try:
                context = copy.deepcopy(self.base_context)
            except Exception:
                context = dict(self.base_context)
        context["batch_mode"] = True
        context["batch_index"] = batch_index
        context["selected_race"] = race
        context.setdefault("selected_races", self.original_races)
        if self.session_id and "session_id" not in context:
            context["session_id"] = self.session_id

        overrides = {}
        for key in ("context_overrides", "overrides", "context"):
            value = race.get(key)
            if isinstance(value, dict):
                overrides.update(value)
        if overrides:
            context.update(overrides)

        hint = self._race_label(race)
        if hint and "contest_hint" not in context:
            context["contest_hint"] = hint
        return context

    def _queue_prompt_responses(self, race: Dict[str, Any]) -> None:
        if not self.prompt or not self.session_id:
            return

        selection_value = None
        for key in ("selection", "indices", "index", "value"):
            if key in race:
                selection_value = race.get(key)
                break
        if selection_value is not None:
            self.prompt.queue_response(
                self.session_id,
                self._format_prompt_value(selection_value),
                matcher=self._contest_matcher,
                consume=True,
            )

        for entry in race.get("prompt_responses") or []:
            if not isinstance(entry, dict):
                continue
            if "value" not in entry:
                continue
            matcher = self._build_matcher(entry.get("matcher"))
            self.prompt.queue_response(
                self.session_id,
                self._format_prompt_value(entry.get("value")),
                matcher=matcher,
                consume=bool(entry.get("consume", True)),
            )

    def _emit_result(
        self,
        result: Tuple[List[str], List[Dict[str, Any]], str, Dict[str, Any]],
        race: Optional[Dict[str, Any]],
        *,
        index: int,
        initial: bool = False,
    ) -> bool:
        headers, data, contest, metadata = result
        if not headers or not data or not contest:
            logger.error({
                "level": "ERROR",
                "type": "batch",
                "message": "[Batch] Result missing required fields; skipping post-processing.",
                "session_id": self.session_id,
                "url": self.target_url,
            })
            return False

        metadata_out = dict(metadata or {})
        batch_meta = metadata_out.get("batch") or {}
        batch_meta.update({
            "batch_id": self.batch_id,
            "index": index,
            "initial": initial,
            "total": max(self.total_expected, index),
        })
        label = self._race_label(race)
        if label:
            batch_meta.setdefault("label", label)
            metadata_out.setdefault("contest_label", label)
        metadata_out["batch"] = batch_meta
        metadata_out.setdefault("batch_mode", True)
        metadata_out.setdefault("output_dir", metadata_out.get("output_dir") or self.output_dir)
        metadata_out.setdefault("source_url", metadata_out.get("source_url") or self.target_url)
        metadata_out.setdefault("session_id", self.session_id)
        if race and "selected_race" not in metadata_out:
            metadata_out["selected_race"] = race
        if race:
            race_meta = race.get("metadata") or {}
            if isinstance(race_meta, dict) and race_meta.get("bundle_mode") == "aggregate":
                metadata_out.setdefault("bundle_mode", "aggregate")
                metadata_out.setdefault("bundle_metadata", race_meta)
                if race_meta.get("bundle_size"):
                    metadata_out.setdefault("bundle_size", race_meta.get("bundle_size"))
                if race_meta.get("bundle_key"):
                    metadata_out.setdefault("bundle_key", race_meta.get("bundle_key"))

        try:
            if self._analysis_executor:
                self._analysis_futures.append(
                    self._analysis_executor.submit(
                        self.ai_analyze_results,
                        headers,
                        data,
                        contest,
                        metadata_out,
                        target_url=self.target_url,
                        session_id=self.session_id,
                    )
                )
                self._analysis_futures.append(
                    self._analysis_executor.submit(
                        self.stream_results,
                        headers,
                        data,
                        contest,
                        metadata_out,
                        target_url=self.target_url,
                        session_id=self.session_id,
                    )
                )
            else:
                self.ai_analyze_results(
                    headers,
                    data,
                    contest,
                    metadata_out,
                    target_url=self.target_url,
                    session_id=self.session_id,
                )
                self.stream_results(
                    headers,
                    data,
                    contest,
                    metadata_out,
                    target_url=self.target_url,
                    session_id=self.session_id,
                )
        except Exception as exc:
            logger.error({
                "level": "ERROR",
                "type": "batch",
                "message": f"[Batch] Post-processing failed: {exc}",
                "session_id": self.session_id,
                "url": self.target_url,
            }, exc_info=True)

        self.results.append((headers, data, contest, metadata_out))
        return True

    def _await_postprocessing(self) -> None:
        if not self._analysis_executor:
            return
        for future in self._analysis_futures:
            try:
                future.result()
            except Exception as exc:
                logger.error({
                    "level": "ERROR",
                    "type": "batch",
                    "message": f"[Batch] Post-processing future failed: {exc}",
                    "session_id": self.session_id,
                    "url": self.target_url,
                }, exc_info=True)
        self._analysis_executor.shutdown(wait=True)

    def _compute_status(self, success: int, failures: int) -> str:
        if failures and not success:
            return "fail"
        if failures:
            return "partial"
        return "success"

    def _mark_processed(self, status: str, success: int, failures: int, duration: float) -> None:
        if not callable(self.mark_url_processed):
            return
        try:
            self.mark_url_processed(
                self.target_url,
                status=status,
                session_id=self.session_id,
                batch_id=self.batch_id,
                batch_total=self.total_expected,
                batch_success=success,
                batch_failures=failures,
                batch_duration=duration,
            )
        except Exception as exc:
            logger.warning({
                "level": "WARNING",
                "type": "batch",
                "message": f"[Batch] Failed to update processed cache: {exc}",
                "session_id": self.session_id,
                "url": self.target_url,
            })

    def _contest_matcher(self, context: Any) -> bool:
        if isinstance(context, dict):
            kind = context.get("kind")
            if kind in {"contest", "selector", "contest_menu"}:
                return True
            if context.get("options"):
                return True
        return False

    def _build_matcher(self, matcher_spec: Any) -> Optional[Callable[[Any], bool]]:
        if callable(matcher_spec):
            return matcher_spec
        if matcher_spec is None:
            return None
        if isinstance(matcher_spec, str):
            target = matcher_spec.lower()

            def _match(ctx: Any) -> bool:
                text = ""
                if isinstance(ctx, dict):
                    text = (ctx.get("message") or ctx.get("prompt") or "")
                else:
                    text = str(ctx or "")
                return target in text.lower()

            return _match
        if isinstance(matcher_spec, dict):
            expected_kind = matcher_spec.get("kind")
            contains = matcher_spec.get("message_contains")

            def _match(ctx: Any) -> bool:
                if expected_kind:
                    if not isinstance(ctx, dict) or ctx.get("kind") != expected_kind:
                        return False
                if contains:
                    text = ""
                    if isinstance(ctx, dict):
                        text = (ctx.get("message") or ctx.get("prompt") or "")
                    else:
                        text = str(ctx or "")
                    if contains.lower() not in text.lower():
                        return False
                return True

            return _match
        return None

    def _format_prompt_value(self, value: Any) -> str:
        if isinstance(value, (list, tuple, set)):
            return ",".join(str(v) for v in value)
        return str(value)

    def _race_label(self, race: Optional[Dict[str, Any]]) -> str:
        if not isinstance(race, dict):
            return ""
        return (
            safe_get(race, "title")
            or safe_get(race, "contest")
            or safe_get(race, "label")
            or safe_get(race, "value")
            or ""
        )


from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
from urllib.parse import urlparse

import orjson

from ..config import LOG_DIR

DEFAULT_RECIPE_PATH = Path(__file__).with_name("navigation_recipes.orjson")
DEFAULT_LEARNED_LOG = Path(LOG_DIR) / "navigation_learning_log.jsonl"


class NavigationRecipeStore:
    """Loads and filters navigation recipes stored in orjson format."""

    def __init__(
        self,
        recipe_path: str | Path | None = None,
        *,
        auto_reload: bool = True,
        learned_log_path: str | Path | None = None,
        learned_enabled: bool = True,
        learned_min_actions: int = 2,
        learned_min_ok_ratio: float = 0.8,
        learned_max_entries: int = 2000,
    ) -> None:
        self.recipe_path = Path(recipe_path or DEFAULT_RECIPE_PATH)
        self.auto_reload = auto_reload
        self.learned_log_path = Path(learned_log_path or DEFAULT_LEARNED_LOG)
        self.learned_enabled = learned_enabled
        self.learned_min_actions = int(learned_min_actions)
        self.learned_min_ok_ratio = float(learned_min_ok_ratio)
        self.learned_max_entries = int(learned_max_entries)
        self._cache: List[Dict[str, Any]] = []
        self._mtime: float | None = None
        self._learned_cache: List[Dict[str, Any]] = []
        self._learned_mtime: float | None = None
        self._lock = threading.RLock()

    def _maybe_reload_locked(self) -> None:
        if not self.recipe_path.exists():
            self._cache = []
            self._mtime = None
            return
        mtime = self.recipe_path.stat().st_mtime
        needs_reload = not self._cache or self._mtime is None
        if self.auto_reload and not needs_reload:
            needs_reload = self._mtime != mtime
        if not needs_reload and self._mtime == mtime:
            return
        try:
            raw = self.recipe_path.read_bytes()
            data = orjson.loads(raw or b"[]")
            if isinstance(data, list):
                self._cache = [item for item in data if isinstance(item, dict)]
            else:
                self._cache = []
            self._mtime = mtime
        except FileNotFoundError:
            self._cache = []
            self._mtime = None

    def load(self) -> List[Dict[str, Any]]:
        with self._lock:
            self._maybe_reload_locked()
            return list(self._cache)

    def load_learned(self) -> List[Dict[str, Any]]:
        if not self.learned_enabled:
            return []
        with self._lock:
            self._maybe_reload_learned_locked()
            return list(self._learned_cache)

    def iter_recipes(self) -> Iterable[Dict[str, Any]]:
        yield from self.load()

    @staticmethod
    def _normalize(value: str | None, *, lower: bool = False) -> str:
        if not isinstance(value, str):
            return ""
        normalized = value.strip()
        return normalized.lower() if lower else normalized.upper()

    def _match_list(self, candidates: Sequence[str] | str | None, value: str, *, lower: bool) -> bool:
        if not candidates:
            return True
        if isinstance(candidates, str):
            candidates = [candidates]
        normalized_value = value.lower() if lower else value.upper()
        for candidate in candidates:
            normalized_candidate = (candidate or "").strip()
            normalized_candidate = normalized_candidate.lower() if lower else normalized_candidate.upper()
            if not normalized_candidate:
                continue
            if normalized_candidate == normalized_value:
                return True
        return False

    def candidates_for(self, context: Dict[str, Any] | None) -> List[Dict[str, Any]]:
        context = context or {}
        state = self._normalize(context.get("state"))
        county = self._normalize(context.get("county"), lower=True)
        selected: List[Dict[str, Any]] = []
        for recipe in self.iter_recipes():
            match = recipe.get("match") or {}
            if not isinstance(match, dict):
                match = {}
            state_ok = self._match_list(match.get("state") or match.get("states"), state, lower=False)
            county_ok = self._match_list(match.get("county") or match.get("counties"), county, lower=True)
            if state_ok and county_ok:
                selected.append(recipe)
        for recipe in self.load_learned():
            match = recipe.get("match") or {}
            if not isinstance(match, dict):
                match = {}
            state_ok = self._match_list(match.get("state") or match.get("states"), state, lower=False)
            county_ok = self._match_list(match.get("county") or match.get("counties"), county, lower=True)
            if state_ok and county_ok:
                selected.append(recipe)
        return selected

    def add_or_update(self, recipe: Dict[str, Any]) -> None:
        """Persist a new or updated recipe back to disk."""

        if not isinstance(recipe, dict):
            return
        recipe_id = str(recipe.get("id") or "").strip() or None
        with self._lock:
            self._maybe_reload_locked()
            updated = False
            if recipe_id:
                for idx, existing in enumerate(self._cache):
                    if str(existing.get("id")) == recipe_id:
                        self._cache[idx] = recipe
                        updated = True
                        break
            if not updated:
                self._cache.append(recipe)
            self._write_locked()

    def _write_locked(self) -> None:
        self.recipe_path.parent.mkdir(parents=True, exist_ok=True)
        payload = orjson.dumps(self._cache, option=orjson.OPT_INDENT_2)
        self.recipe_path.write_bytes(payload)
        self._mtime = self.recipe_path.stat().st_mtime

    def _maybe_reload_learned_locked(self) -> None:
        if not self.learned_log_path.exists():
            self._learned_cache = []
            self._learned_mtime = None
            return
        mtime = self.learned_log_path.stat().st_mtime
        needs_reload = not self._learned_cache or self._learned_mtime is None
        if self.auto_reload and not needs_reload:
            needs_reload = self._learned_mtime != mtime
        if not needs_reload and self._learned_mtime == mtime:
            return
        self._learned_cache = self._build_learned_recipes()
        self._learned_mtime = mtime

    def _build_learned_recipes(self) -> List[Dict[str, Any]]:
        recipes: List[Dict[str, Any]] = []
        if not self.learned_log_path.exists():
            return recipes
        try:
            with self.learned_log_path.open("rb") as handle:
                for line in handle:
                    if len(recipes) >= self.learned_max_entries:
                        break
                    entry = self._parse_log_line(line)
                    if not entry:
                        continue
                    recipe = self._entry_to_recipe(entry)
                    if recipe:
                        recipes.append(recipe)
        except Exception:
            return recipes
        return recipes

    @staticmethod
    def _parse_log_line(line: bytes) -> Dict[str, Any] | None:
        if not line or not line.strip():
            return None
        try:
            data = orjson.loads(line)
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def _entry_to_recipe(self, entry: Dict[str, Any]) -> Dict[str, Any] | None:
        if not entry.get("success"):
            return None
        telemetry = entry.get("telemetry") or []
        if not isinstance(telemetry, list) or len(telemetry) < self.learned_min_actions:
            return None
        ok_steps = [t for t in telemetry if isinstance(t, dict) and t.get("status") == "ok"]
        ok_ratio = (len(ok_steps) / max(len(telemetry), 1)) if telemetry else 0.0
        if ok_ratio < self.learned_min_ok_ratio:
            return None
        steps = self._telemetry_to_steps(ok_steps)
        if not steps:
            return None

        context_after = entry.get("context_after") or {}
        context_before = entry.get("context_before") or {}
        state = context_after.get("state") or context_before.get("state")
        county = context_after.get("county") or context_before.get("county")
        url = (
            (entry.get("metadata") or {}).get("page_url")
            or context_after.get("url")
            or context_before.get("url")
        )
        url_domain = (entry.get("metadata") or {}).get("url_domain")
        if not url_domain and isinstance(url, str):
            try:
                parsed = urlparse(url)
                url_domain = parsed.hostname or None
            except Exception:
                url_domain = None

        recipe_id = entry.get("script_id") or "learned"
        match: Dict[str, Any] = {}
        if state:
            match["state"] = state
        if county:
            match["county"] = county
        if url_domain:
            match["url_contains"] = [url_domain]

        return {
            "id": f"learned::{recipe_id}",
            "description": "Learned navigation replay",
            "match": match,
            "steps": steps,
            "metadata": {
                "source": "learned",
                "ok_ratio": ok_ratio,
                "action_count": len(telemetry),
            },
        }

    @staticmethod
    def _telemetry_to_steps(telemetry: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        steps: List[Dict[str, Any]] = []
        for entry in telemetry:
            action = entry.get("action")
            details = entry.get("details") or {}
            if action == "wait_for_selector":
                selector = details.get("selector")
                if selector:
                    step = {"action": action, "selector": selector}
                    if details.get("timeout_ms") is not None:
                        step["timeout_ms"] = details.get("timeout_ms")
                    steps.append(step)
            elif action == "wait_for_load_state":
                step = {"action": action, "state": details.get("state") or "networkidle"}
                if details.get("timeout_ms") is not None:
                    step["timeout_ms"] = details.get("timeout_ms")
                steps.append(step)
            elif action == "click":
                selector = details.get("selector")
                if selector:
                    step = {"action": action, "selector": selector}
                    if details.get("timeout_ms") is not None:
                        step["timeout_ms"] = details.get("timeout_ms")
                    if details.get("wait_after_ms") is not None:
                        step["wait_after_ms"] = details.get("wait_after_ms")
                    steps.append(step)
            elif action == "wait":
                if details.get("timeout_ms") is not None:
                    steps.append({"action": action, "timeout_ms": details.get("timeout_ms")})
            elif action == "fill":
                selector = details.get("selector")
                if selector is not None:
                    steps.append({"action": action, "selector": selector, "value": details.get("value", "")})
            elif action == "run_js":
                expression = details.get("expression")
                if expression:
                    steps.append({"action": action, "expression": expression})
            elif action == "autoscroll":
                step = {"action": action}
                if details.get("max_time_ms") is not None:
                    step["max_time_ms"] = details.get("max_time_ms")
                steps.append(step)
            elif action == "scan_context":
                steps.append({"action": action})
            elif action == "set_context":
                if details.get("keys"):
                    steps.append({"action": action, "values": {}})
            elif action == "parallel":
                steps.append({"action": action})
        return steps


__all__ = ["NavigationRecipeStore", "DEFAULT_RECIPE_PATH"]

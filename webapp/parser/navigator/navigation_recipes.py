from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import orjson

DEFAULT_RECIPE_PATH = Path(__file__).with_name("navigation_recipes.orjson")


class NavigationRecipeStore:
    """Loads and filters navigation recipes stored in orjson format."""

    def __init__(
        self,
        recipe_path: str | Path | None = None,
        *,
        auto_reload: bool = True,
    ) -> None:
        self.recipe_path = Path(recipe_path or DEFAULT_RECIPE_PATH)
        self.auto_reload = auto_reload
        self._cache: List[Dict[str, Any]] = []
        self._mtime: float | None = None
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


__all__ = ["NavigationRecipeStore", "DEFAULT_RECIPE_PATH"]
